"""CREMA-D: file discovery, target selection, and LongCat-AudioDiT voice cloning.

CREMA-D has 91 actors reading the same 12 sentences in six emotions, which makes same-speaker MCD
possible: clone each actor's voice from a prompt of that actor saying a *different* sentence,
then score against the actor's real recording of the target sentence. The "addon" and
"crossclone" experiments share the exact target selection below (same seed, same order), so
the voice-A clones reproduce between them.
"""
import glob
import os
import random
import re

import librosa
import numpy as np
import pandas as pd
import torch

from .model import import_audiodit

CREMA_TEXT = {
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.", "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.", "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.", "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.", "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.", "WSI": "We'll stop in a couple of minutes.",
}
CREMA_EMO = {"ANG": "anger", "DIS": "disgust", "FEA": "fear", "HAP": "joy", "NEU": "neutral", "SAD": "sadness"}
PAT = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})\.wav$")


def find_cremad(cremad_dir=None) -> pd.DataFrame:
    """Every CREMA-D clip of a known sentence and emotion. With cremad_dir=None, searches
    /kaggle/input for an AudioWAV folder, then for any file matching CREMA-D's naming scheme."""
    cremad_files = []
    roots = [cremad_dir] if cremad_dir else glob.glob("/kaggle/input/**/AudioWAV", recursive=True)
    for r in roots:
        cremad_files += glob.glob(os.path.join(r, "*.wav"))
    if not cremad_files:   # fall back to any matching file name anywhere under /kaggle/input
        cremad_files = [p for p in glob.glob("/kaggle/input/**/*.wav", recursive=True) if PAT.match(os.path.basename(p))]
    rows = []
    for p in cremad_files:
        m = PAT.match(os.path.basename(p))
        if m and m.group(3) in CREMA_EMO and m.group(2) in CREMA_TEXT:
            rows.append({"path": p, "actor": m.group(1), "sent": m.group(2), "emo_code": m.group(3),
                         "emotion": CREMA_EMO[m.group(3)], "level": m.group(4)})
    crema = pd.DataFrame(rows)
    print(f"CREMA-D files found: {len(crema)} | actors {crema['actor'].nunique() if len(crema) else 0}")
    return crema


def _single_take_pool(crema):
    return crema[(crema.level == "XX") & (crema.sent != "IEO")]      # one take per actor/sentence/emotion


def select_targets(crema: pd.DataFrame, cfg) -> pd.DataFrame:
    """Per sampled actor and emotion: a target sentence (`ref_same`), a prompt by the same actor
    saying a different sentence, and a different actor's recording of the target (`ref_other`)."""
    targets = []
    if len(crema):
        rng = random.Random(cfg.eval_seed)
        pool = _single_take_pool(crema)
        actors = sorted(pool.actor.unique()); rng.shuffle(actors)
        for a in actors[:cfg.cremad_actors]:
            for emo in CREMA_EMO.values():
                own = pool[(pool.actor == a) & (pool.emotion == emo)]
                if len(own) < 2:
                    continue
                tgt = own.sample(1, random_state=rng.randint(0, 10**6)).iloc[0]
                prm = own[own.sent != tgt.sent].sample(1, random_state=rng.randint(0, 10**6)).iloc[0]
                other = pool[(pool.actor != a) & (pool.sent == tgt.sent) & (pool.emotion == emo)]
                if not len(other):
                    continue
                oth = other.sample(1, random_state=rng.randint(0, 10**6)).iloc[0]
                targets.append({"key": f"{a}_{tgt.sent}_{tgt.emo_code}", "actor": a, "emotion": emo,
                                "text": CREMA_TEXT[tgt.sent], "ref_same": tgt.path,
                                "prompt_path": prm.path, "prompt_text": CREMA_TEXT[prm.sent],
                                "ref_other": oth.path, "other_actor": oth.actor})
    targets = pd.DataFrame(targets)
    print(f"targets: {len(targets)} ({targets['actor'].nunique() if len(targets) else 0} actors)")
    return targets


def add_other_actor_prompts(targets: pd.DataFrame, crema: pd.DataFrame, cfg) -> pd.DataFrame:
    """Prompt for actor B: B saying a different sentence in the same emotion. Targets without
    one are dropped."""
    rng = random.Random(cfg.eval_seed + 7)
    pool = _single_take_pool(crema)
    pB_path, pB_text = [], []
    for _, r in targets.iterrows():
        tgt_sent = [k for k, v in CREMA_TEXT.items() if v == r.text][0]
        cand = pool[(pool.actor == r.other_actor) & (pool.emotion == r.emotion) & (pool.sent != tgt_sent)]
        row = cand.sample(1, random_state=rng.randint(0, 10**6)).iloc[0] if len(cand) else None
        pB_path.append(row.path if row is not None else None)
        pB_text.append(CREMA_TEXT[row.sent] if row is not None else None)
    targets = targets.copy()
    targets["prompt_B_path"], targets["prompt_B_text"] = pB_path, pB_text
    targets = targets[targets.prompt_B_path.notna()].reset_index(drop=True)
    print(f"{len(targets)} targets with a usable prompt from the other actor")
    return targets


def install_extras(*pkgs):
    import subprocess
    import sys
    for pkg in pkgs:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)


@torch.no_grad()
def generate_cloned(model, tokenizer, cfg, device, text, prompt_path, prompt_text, cfg_strength):
    """Voice cloning exactly as LongCat-AudioDiT's inference.py does it: prompt text is prepended
    to the target text, duration follows the repository's estimator, and the model strips the
    prompt from the returned waveform."""
    import_audiodit(cfg)                       # puts the LongCat repo on sys.path
    from utils import approx_duration_from_text, normalize_text   # LongCat-AudioDiT's own helpers

    model.eval()
    pw, _ = librosa.load(prompt_path, sr=cfg.sample_rate, mono=True)
    pw_t = torch.from_numpy(pw).float().view(1, 1, -1).to(device)
    _, prompt_frames = model.encode_prompt_audio(pw_t)
    prompt_time = prompt_frames * cfg.latent_hop / cfg.sample_rate
    t_norm, p_norm = normalize_text(text), normalize_text(prompt_text)
    max_d = model.config.max_wav_duration
    dur = approx_duration_from_text(t_norm, max_duration=max_d - prompt_time)
    ratio = float(np.clip(prompt_time / max(approx_duration_from_text(p_norm, max_duration=max_d), 1e-3), 1.0, 1.5))
    total = min(int(dur * ratio * cfg.sample_rate // cfg.latent_hop) + prompt_frames,
                int(max_d * cfg.sample_rate // cfg.latent_hop))
    enc = tokenizer([f"{p_norm} {t_norm}"], padding="longest", return_tensors="pt").to(device)
    out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, prompt_audio=pw_t,
                duration=total, steps=cfg.cfm_steps, cfg_strength=cfg_strength, guidance_method="cfg")
    return out.waveform.squeeze().float().cpu().numpy()


def load16(path):
    """A clip at 16 kHz mono float32 — the input every scorer in these experiments takes."""
    return librosa.load(path, sr=16000, mono=True)[0].astype(np.float32)
