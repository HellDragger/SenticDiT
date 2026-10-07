"""Speech generation and the generate-then-score loop every evaluation stage is built on."""
import pandas as pd
import soundfile as sf
import torch

from .data import calibrated_duration_frames
from .metrics import compute_health_metrics


def bracket_prompt(text, emotion):
    return f"[{emotion.upper()}] {text}"


# Natural-language instructions the model never saw in training (it was only ever trained on
# `[EMOTION] text`) — used by the conditioning-generalisation test.
PARAPHRASE_TEMPLATES = {
    "neutral": "Say this in a neutral, matter-of-fact tone: {text}",
    "joy": "Say this in a happy, joyful tone: {text}",
    "sadness": "Say this in a sad, sorrowful tone: {text}",
    "anger": "Say this in an angry, furious tone: {text}",
    "fear": "Say this in a fearful, frightened tone: {text}",
    "disgust": "Say this in a disgusted, repulsed tone: {text}",
    "surprise": "Say this in a surprised, shocked tone: {text}",
}


def paraphrase_prompt(text, emotion):
    return PARAPHRASE_TEMPLATES.get(emotion, "{text}").format(text=text)


@torch.no_grad()
def generate_speech(model, tokenizer, text, cfg, device, emotion=None,
                     out_path=None, steps=None, cfg_strength=None):
    model.eval()
    prompt = f"[{emotion.upper()}] {text}" if emotion else text
    enc = tokenizer([prompt], padding="longest", return_tensors="pt").to(device)
    duration = calibrated_duration_frames(text, cfg)
    output = model(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        duration=duration,
        steps=steps or cfg.cfm_steps,
        cfg_strength=cfg_strength if cfg_strength is not None else cfg.cfg_strength,
        guidance_method="cfg",
    )
    wav = output.waveform.squeeze().float().cpu().numpy()
    if out_path:
        sf.write(out_path, wav, cfg.sample_rate)
    return wav


def generate_and_score(model, tokenizer, df, cfg, device, emotions=None, n_per_emotion=None,
                        cfg_strength=None, steps=None, prompt_fn=bracket_prompt, return_wavs=False):
    """Generate every (or `n_per_emotion` sampled) held-out utterance and score its health.

    n_per_emotion=None uses every held-out row for that emotion. `utt_id` and `ref_wav_path` let
    every arm be paired to the same utterance, which is what paired_bootstrap_diff needs.
    """
    emotions = emotions or cfg.emotions
    cfg_strength = cfg.cfg_strength if cfg_strength is None else cfg_strength
    steps = steps or cfg.cfm_steps
    model.eval()
    rows = []
    for emotion in emotions:
        pool = df[df["emotion"] == emotion]
        if len(pool) == 0:
            continue
        if n_per_emotion is None:
            sampled = pool
        else:
            sampled = pool.sample(n=n_per_emotion, replace=len(pool) < n_per_emotion,
                                   random_state=cfg.seed)
        for idx, r in sampled.iterrows():
            text = str(r["utterance"]).strip()
            if not text:
                continue
            prompt = prompt_fn(text, emotion)
            enc = tokenizer([prompt], padding="longest", return_tensors="pt").to(device)
            duration = calibrated_duration_frames(text, cfg)
            with torch.no_grad():
                out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                            duration=duration, steps=steps, cfg_strength=cfg_strength,
                            guidance_method="cfg")
            wav = out.waveform.squeeze().float().cpu().numpy()
            metrics = compute_health_metrics(wav, cfg.sample_rate, cfg)
            metrics.update({"emotion": emotion, "utt_id": idx, "text": text, "prompt": prompt,
                            "n_chars": len(text), "ref_wav_path": r.get("wav_path")})
            if return_wavs:
                metrics["wav"] = wav
            rows.append(metrics)
    return pd.DataFrame(rows)


def evaluate_holdout(model, tokenizer, holdout_df, cfg, device, cfg_strength=None,
                     prompt_fn=bracket_prompt, return_wavs=False):
    """The standard full held-out pass used by every arm."""
    return generate_and_score(model, tokenizer, holdout_df, cfg, device,
                              emotions=cfg.emotions, n_per_emotion=cfg.eval_samples_per_emotion,
                              cfg_strength=cfg_strength, steps=cfg.cfm_steps,
                              prompt_fn=prompt_fn, return_wavs=return_wavs)
