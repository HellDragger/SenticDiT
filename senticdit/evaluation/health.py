"""Generation-health evaluation stages: zero-shot control, CFG sweep, full held-out evaluation,
conditioning generalisation, and demo samples."""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from ..generation import evaluate_holdout, generate_and_score, generate_speech, paraphrase_prompt
from ..stats import describe_diff, min_detectable_effect, paired_bootstrap_diff, wilson_ci
from ..utils import reseed, show_audio


def zero_shot_baseline(ctx):
    """The untouched pretrained model, before any LoRA is attached, on the same held-out set with
    the same prompts and metrics. Without this row the paper never shows LoRA fine-tuning beat
    simply prompting the base model. WAVs are written to disk so later stages can reuse them."""
    cfg = ctx.cfg
    if not cfg.run_zero_shot_baseline:
        print("cfg.run_zero_shot_baseline=False — skipping the pretrained control.")
        return None

    reseed(cfg, "zeroshot")
    zs_dir = Path(cfg.output_dir) / "zeroshot_wavs"
    zs_dir.mkdir(parents=True, exist_ok=True)

    zeroshot_df = evaluate_holdout(ctx.model, ctx.tokenizer, ctx.holdout_df, cfg, ctx.device,
                                   return_wavs=True)

    wav_files = []
    for pos, (_, r) in enumerate(zeroshot_df.iterrows()):
        p = zs_dir / f"zs_{pos:04d}_{r['emotion']}.wav"
        sf.write(p, r["wav"], cfg.sample_rate)
        wav_files.append(str(p))
    zeroshot_df["wav_file"] = wav_files
    zeroshot_df.drop(columns=["wav"]).to_csv(
        f"{cfg.output_dir}/health_metrics_zeroshot.csv", index=False)

    print(f"Zero-shot baseline regenerated this session "
          f"(force_regenerate_zeroshot={cfg.force_regenerate_zeroshot}).")
    print("  If these figures match a previous run to 4 decimals, the arm was NOT regenerated —")
    print("  check that no cached CSV is being read and that reseed() ran.\n")
    print(f"Zero-shot baseline: {len(zeroshot_df)} utterances "
          f"({zeroshot_df['emotion'].nunique()} emotions)\n")
    print(zeroshot_df.groupby("emotion")[
        ["completion_ratio", "duration_fit", "f0_median", "f0_iqr"]].median().round(3))
    print(f"\nOctave-error flags: {int(zeroshot_df['f0_octave_flag'].sum())}/{len(zeroshot_df)}")
    return zeroshot_df


def cfg_sweep(ctx) -> float:
    """Sweep classifier-free guidance strength, then report it as a null.

    Two v3 runs of this sweep produced ANTI-correlated rankings (Spearman rho = -0.40 on both
    completion_ratio and f0_std); per-clip noise through a 16-step Euler solve swamps the
    between-scale effect at any affordable sample size. So the scale is fixed by fiat.
    """
    cfg = ctx.cfg
    reseed(cfg, "cfg_sweep")
    sweep_rows = []
    for strength in cfg.cfg_sweep_values:
        df_s = generate_and_score(ctx.model, ctx.tokenizer, ctx.holdout_df, cfg, ctx.device,
                                   emotions=cfg.emotions, n_per_emotion=cfg.cfg_sweep_per_emotion,
                                   cfg_strength=strength, steps=cfg.cfm_steps)
        df_s["cfg_strength"] = strength
        sweep_rows.append(df_s)

    cfg_sweep_df = pd.concat(sweep_rows, ignore_index=True)
    sweep_summary = cfg_sweep_df.groupby("cfg_strength")[
        ["completion_ratio", "f0_std", "rms"]].mean().round(3)
    print(sweep_summary.to_string())

    print("\n--- Pairwise tests against the reference scale (paired by utterance) ---")
    ref_scale = cfg.cfg_strength
    ref_df = cfg_sweep_df[cfg_sweep_df["cfg_strength"] == ref_scale]
    for strength in cfg.cfg_sweep_values:
        if strength == ref_scale:
            continue
        alt = cfg_sweep_df[cfg_sweep_df["cfg_strength"] == strength]
        mg = ref_df.merge(alt, on="utt_id", suffixes=("_ref", "_alt"))
        for col in ["completion_ratio", "f0_std"]:
            r = paired_bootstrap_diff(mg[f"{col}_ref"], mg[f"{col}_alt"], n_boot=cfg.n_boot_diff)
            if r.get("n"):
                print(f"  {col:16s} {describe_diff(r, f'cfg {ref_scale}', f'cfg {strength}')}")

    # No hardcoded override. If nothing is separable, keep the configured default and say so.
    candidates = sweep_summary[sweep_summary["f0_std"] < cfg.f0_std_sanity_bound]
    swept_best = float(candidates["completion_ratio"].idxmax()) if len(candidates) \
        else float(sweep_summary["completion_ratio"].idxmax())
    best_cfg_strength = float(cfg.cfg_strength)

    print(f"\nSweep's nominal argmax: cfg={swept_best}")
    print(f"Scale used for final evaluation: cfg={best_cfg_strength} (cfg.cfg_strength)")
    print("  Fixed by fiat, not selected. Across two independent runs the sweep rankings")
    print("  anti-correlated, so there is no stable argmax to select. Report as a null.")
    if swept_best != best_cfg_strength:
        print(f"  (This run's argmax was {swept_best}; that it differs from the configured default")
        print("   is itself the point — do not treat either as a winner.)")

    cfg_sweep_df.to_csv(f"{cfg.output_dir}/cfg_strength_sweep.csv", index=False)
    return best_cfg_strength


def final_evaluation(ctx):
    """Every held-out utterance, reported the way a censored proportion has to be: medians, a
    failure count with a Wilson interval, an octave-flag rate, and a paired comparison against
    the zero-shot arm."""
    cfg = ctx.cfg
    reseed(cfg, "final_eval")
    final_eval_df = evaluate_holdout(ctx.model, ctx.tokenizer, ctx.holdout_df, cfg, ctx.device,
                                     cfg_strength=ctx.best_cfg_strength, return_wavs=True)

    final_eval_df.drop(columns=["wav"]).to_csv(f"{cfg.output_dir}/health_metrics_final.csv", index=False)

    # --- per-emotion summary: medians, because the F0 mean is what v2 got burned on ------------
    summary = final_eval_df.groupby("emotion").agg(
        n=("completion_ratio", "size"),
        voiced_frac_median=("voiced_fraction", "median"),
        duration_fit_median=("duration_fit", "median"),
        f0_median=("f0_median", "median"),
        f0_iqr_median=("f0_iqr", "median"),
        rms_median=("rms", "median"),
        octave_flags=("f0_octave_flag", "sum"),
        at_400hz_bound=("f0_at_bound", "sum"),
        f0_median_wide=("f0_median_wide", "median"),
        f0_invalid=("f0_valid", lambda s: int((~s).sum())),
    ).round(3)
    print("Per-emotion evaluation (n = %d):" % len(final_eval_df))
    print(summary.to_string())

    # --- failure count, which is what a ceiling-censored proportion can actually resolve -------
    thr = cfg.completion_fail_threshold
    print(f"\nClips with voiced_fraction < {thr}:")
    for emo in cfg.emotions:
        sub = final_eval_df[final_eval_df["emotion"] == emo]
        k = int((sub["voiced_fraction"] < thr).sum())
        lo, hi = wilson_ci(k, len(sub))
        print(f"  {emo:9s} {k:2d}/{len(sub):2d}   95% CI on failure rate [{lo:.3f}, {hi:.3f}]")
    k_all = int((final_eval_df["voiced_fraction"] < thr).sum())
    lo, hi = wilson_ci(k_all, len(final_eval_df))
    print(f"  {'TOTAL':9s} {k_all:2d}/{len(final_eval_df):2d}   95% CI [{lo:.3f}, {hi:.3f}]")

    ceil = int((final_eval_df["voiced_fraction"] >= 0.9999).sum())
    print(f"\nCeiling censoring check: {ceil}/{len(final_eval_df)} "
          f"({100*ceil/len(final_eval_df):.0f}%) of clips sit at exactly 1.000.")
    print("  (v2's duration_fit sat at the ceiling for 71% of clips, which is why every ablation")
    print("   on it came back null. If voiced_fraction is also heavily censored, report the failure")
    print("   count above as the headline and treat the mean as descriptive only.)")

    # --- fine-tuned vs zero-shot, paired by utterance ------------------------------------------
    if ctx.zeroshot_df is not None:
        merged = final_eval_df.merge(ctx.zeroshot_df, on="utt_id", suffixes=("_ft", "_zs"))
        print(f"\n=== Fine-tuned vs zero-shot pretrained (paired, n={len(merged)}) ===")
        for col, unit in [("voiced_fraction", ""), ("duration_fit", ""),
                           ("f0_median", " Hz"), ("f0_iqr", " Hz"), ("rms", "")]:
            res = paired_bootstrap_diff(merged[f"{col}_ft"], merged[f"{col}_zs"], n_boot=cfg.n_boot_diff)
            print(f"  {col:16s} {describe_diff(res, 'fine-tuned', 'zero-shot', unit)}")
            sd = float(np.nanstd(merged[f"{col}_ft"] - merged[f"{col}_zs"], ddof=1))
            print(f"  {'':16s} minimum detectable effect at this n: "
                  f"{min_detectable_effect(sd, len(merged), paired=True):.4g}{unit}")
        merged.to_csv(f"{cfg.output_dir}/finetuned_vs_zeroshot_paired.csv", index=False)
    return final_eval_df


def paraphrase_generalization(ctx):
    """Does the model generalise to natural-language emotion instructions it never saw in
    training, or is it brittle to the exact `[EMOTION]` prefix? Inference only."""
    cfg = ctx.cfg
    paraphrase_eval_df = evaluate_holdout(ctx.model, ctx.tokenizer, ctx.holdout_df, cfg, ctx.device,
                                          cfg_strength=ctx.best_cfg_strength,
                                          prompt_fn=paraphrase_prompt, return_wavs=True)

    preds = [ctx.ser.predict(w, cfg.sample_rate) for w in paraphrase_eval_df["wav"]]
    paraphrase_accuracy = float(np.mean([p == e for p, e in zip(preds, paraphrase_eval_df["emotion"])]))

    print(f"Bracket-style ([EMOTION] prefix)     SER agreement: {ctx.ser_accuracy:.1%}")
    print(f"Natural-language paraphrase prompt   SER agreement: {paraphrase_accuracy:.1%}")

    generalization_df = pd.DataFrame({
        "condition": ["bracket_token", "natural_language_paraphrase"],
        "ser_accuracy": [ctx.ser_accuracy, paraphrase_accuracy],
        "mean_completion_ratio": [ctx.final_eval_df["completion_ratio"].mean(),
                                  paraphrase_eval_df["completion_ratio"].mean()],
    })
    print(generalization_df)
    generalization_df.to_csv(f"{cfg.output_dir}/conditioning_generalization.csv", index=False)
    return generalization_df


def demo_samples(ctx):
    """One fixed sentence rendered in every emotion, saved as demo_<emotion>.wav."""
    cfg = ctx.cfg
    demo_wavs = {}
    for emotion in cfg.emotions:
        wav = generate_speech(ctx.model, ctx.tokenizer, cfg.demo_text, cfg, ctx.device, emotion=emotion,
                               cfg_strength=ctx.best_cfg_strength,
                               out_path=f"{cfg.output_dir}/demo_{emotion}.wav")
        demo_wavs[emotion] = wav
        print(f"--- {emotion} ---")
        show_audio(wav, cfg.sample_rate)
    return demo_wavs
