"""Experiment "crossclone": separating speaker from channel in MCD (senticdit_crossclone.ipynb).

The add-on cloned each actor from that actor's own prompt, so speaker identity and recording
channel were copied together. Here every target sentence is generated twice — from a prompt by
its own actor A and from a prompt by a different actor B recorded in the same booth — and each
output is scored against both actors' real recordings of the sentence. A 2x2 with the channel
held fixed:

|                       | reference A | reference B |
|-----------------------|-------------|-------------|
| voice cloned from A   | match       | mismatch    |
| voice cloned from B   | mismatch    | match       |

The matched-vs-mismatched difference is MCD's sensitivity to speaker identity at matched channel.
Speaker embeddings confirm which voice each clone carries.

Inputs: CREMA-D (`ejlok1/cremad`). No adapter and no MELD needed. Internet on. ~35 min on 2x T4.
"""
import gc
import time
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
import torch

from ..context import RunContext
from ..cremad import add_other_actor_prompts, find_cremad, generate_cloned, install_extras, load16, \
    select_targets
from ..evaluation.speaker import embed_normalized, load_ecapa, wavlm_embeddings
from ..stats import paired_bootstrap_diff
from ..utils import reseed, section, zip_outputs


def generate(ctx, targets, wav_dir, cfg_strength):
    cfg = ctx.cfg
    t0 = time.time()
    reseed(cfg, "addon_cremad_cloned")          # same tag as the add-on run: voice-A clones reproduce exactly
    for _, r in targets.iterrows():
        sf.write(wav_dir / f"cloneA_{r.key}.wav",
                 generate_cloned(ctx.model, ctx.tokenizer, cfg, ctx.device, r.text, r.prompt_path,
                                 r.prompt_text, cfg_strength), cfg.sample_rate)
    reseed(cfg, "crossclone_B")
    for _, r in targets.iterrows():
        sf.write(wav_dir / f"cloneB_{r.key}.wav",
                 generate_cloned(ctx.model, ctx.tokenizer, cfg, ctx.device, r.text, r.prompt_B_path,
                                 r.prompt_B_text, cfg_strength), cfg.sample_rate)
    print(f"generated {2 * len(targets)} clips in {(time.time() - t0) / 60:.1f} min")


def score(ctx, targets, wav_dir, out_dir):
    """MCD in the 2x2, and speaker embeddings (ECAPA if SpeechBrain imports, and WavLM)."""
    from pymcd.mcd import Calculate_MCD

    cfg = ctx.cfg
    calc = {m: Calculate_MCD(MCD_mode=m) for m in ["dtw", "plain"]}
    rows = []
    for _, r in targets.iterrows():
        for voice in ["A", "B"]:
            g = str(wav_dir / f"clone{voice}_{r.key}.wav")
            for refname, ref in [("A", r.ref_same), ("B", r.ref_other)]:
                rec = {"key": r.key, "emotion": r.emotion, "voice": voice, "ref": refname,
                       "match": voice == refname,
                       "dur_ratio": librosa.get_duration(path=g) / librosa.get_duration(path=ref)}
                for m, c in calc.items():
                    rec[f"mcd_{m}"] = c.calculate_mcd(ref, g)
                rows.append(rec)
    cc = pd.DataFrame(rows); cc.to_csv(out_dir / "crossclone_mcd.csv", index=False)

    paths = {}
    for _, r in targets.iterrows():
        paths[("cloneA", r.key)] = str(wav_dir / f"cloneA_{r.key}.wav")
        paths[("cloneB", r.key)] = str(wav_dir / f"cloneB_{r.key}.wav")
        paths[("realA", r.key)] = r.ref_same
        paths[("realB", r.key)] = r.ref_other
    keys = list(paths)
    emb = {}
    try:
        emb["ECAPA"] = embed_normalized(load_ecapa(ctx.device), [load16(paths[k]) for k in keys])
    except Exception as e:
        print(f"ECAPA unavailable ({e})")
    gc.collect(); torch.cuda.empty_cache()
    emb["WavLM"] = wavlm_embeddings([load16(paths[k]) for k in keys], cfg.speaker_model_id, ctx.device)
    idx = {k: i for i, k in enumerate(keys)}
    simrows = []
    for name, E in emb.items():
        for _, r in targets.iterrows():
            for voice in ["A", "B"]:
                for ref in ["A", "B"]:
                    simrows.append({"model": name, "key": r.key, "voice": voice, "ref": ref,
                                    "sim": float(E[idx[(f"clone{voice}", r.key)]] @ E[idx[(f"real{ref}", r.key)]])})
    sims = pd.DataFrame(simrows); sims.to_csv(out_dir / "crossclone_speaker_sim.csv", index=False)
    return cc, sims


def results(cc, sims, cfg):
    print("Mean MCD (dB) in the 2x2, rows = cloned voice, columns = reference speaker:")
    for m in ["dtw", "plain"]:
        print(f"\n  {m}")
        print(cc.pivot_table(index="voice", columns="ref", values=f"mcd_{m}", aggfunc="mean").round(2).to_string())

    # speaker effect with channel held fixed: per target, mean(matched) - mean(mismatched)
    for m in ["dtw", "plain"]:
        per = cc.groupby(["key", "match"])[f"mcd_{m}"].mean().unstack()
        r = paired_bootstrap_diff(per[False], per[True], n_boot=cfg.n_boot_diff)
        print(f"\n  {m}: mismatched minus matched speaker = {r['diff']:+.2f} dB "
              f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}], n={r['n']}")
        for voice in ["A", "B"]:
            d = cc[cc.voice == voice].pivot_table(index="key", columns="ref", values=f"mcd_{m}")
            own, other = ("A", "B") if voice == "A" else ("B", "A")
            rr = paired_bootstrap_diff(d[other], d[own], n_boot=cfg.n_boot_diff)
            print(f"     voice {voice}: other-speaker ref minus own-speaker ref {rr['diff']:+.2f} [{rr['ci_low']:+.2f}, {rr['ci_high']:+.2f}]")

    print("\nSpeaker similarity of each clone to each actor's real recording:")
    print(sims.pivot_table(index=["model", "voice"], columns="ref", values="sim", aggfunc="mean").round(3).to_string())
    print("\nIf each clone is most similar to the actor it was prompted from, the clones carry the intended")
    print("speaker, and the MCD difference above is the speaker effect at matched channel.")


def run(cfg):
    ctx = RunContext(cfg)
    ctx.load_model(snapshot=False)       # MELD is not needed: cloning uses LongCat's own duration estimator

    out_dir = Path(cfg.output_dir) / "crossclone"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wavs"; wav_dir.mkdir(exist_ok=True)
    cfg_strength = float(cfg.cfg_strength)
    install_extras("speechbrain", "jiwer")

    section("Targets (same selection and seed as the add-on run)")
    crema = find_cremad(cfg.cremad_dir)
    targets = select_targets(crema, cfg)
    targets.to_csv(out_dir / "cremad_targets.csv", index=False)
    targets = add_other_actor_prompts(targets, crema, cfg)
    targets.to_csv(out_dir / "crossclone_targets.csv", index=False)

    section("Generation")
    generate(ctx, targets, wav_dir, cfg_strength)
    ctx.free_model()

    section("Scoring: MCD in the 2x2, and speaker embeddings")
    cc, sims = score(ctx, targets, wav_dir, out_dir)
    section("Results")
    results(cc, sims, cfg)

    if cfg.zip_results:
        zip_outputs(Path(cfg.output_dir).parent / "crossclone_results", Path(cfg.output_dir), ["crossclone"])
    return cc, sims
