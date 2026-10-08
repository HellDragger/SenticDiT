"""Assembled results for the paper, the cumulative cross-run log, and the output manifest.

Every number the paper needs, in one place, with the interval attached and the provenance of each
metric stated. Anything that cannot be given an interval is marked as descriptive so it cannot be
promoted to a claim by accident.

Outputs -> paper sections:
| File                                    | Use it for |
|-----------------------------------------|------------|
| paper_results_table.csv                 | every headline number with its interval, tagged by run and seed |
| run_log.csv                             | accumulates across runs; the spread is the variance estimate for the nulls |
| mcd_per_utterance.csv                   | corrected dtw MCD, exclusion reasons, duration ratios, register gaps |
| pitch_drift_mechanism.csv               | per-utterance pitch shift vs UTMOS change — the mechanism figure |
| utmos_anchor.csv                        | stratified, length-filtered real-MELD anchor |
| mcd_alignment_artefact.csv / .png       | methods figure: plain climbs to ~17 dB while dtw stays flat near 5 |
| health_metrics_final.csv / _zeroshot.csv| both carry UTMOS and the wide-band F0 diagnostic |
| finetuned_vs_zeroshot_paired.csv        | the controlled comparison |
| ser_diagnostics.csv, ser_predictions_real_audio.csv | the demonstrated classifier-bias result |
| wer_per_utterance.csv                   | intelligibility: fine-tuned, zero-shot, real MELD |
| ser_two_classifier_comparison.csv       | the generalised SER-bias table (2 classifiers x 3 audio sources) |
| speaker_consistency.csv                 | identity stability before and after fine-tuning |
| configd_comparison.csv / configd_per_utterance.csv | the single-factor data-quantity result (configd experiment) |
| cfg_strength_sweep.csv                  | report as a null; two runs gave anti-correlated rankings |
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation.wer import corpus_wer
from .stats import bootstrap_ci, paired_bootstrap_diff, wilson_ci


def paper_results_table(ctx) -> pd.DataFrame:
    cfg = ctx.cfg
    final_eval_df, zeroshot_df = ctx.final_eval_df, ctx.zeroshot_df
    kept, mcd_df = ctx.mcd_kept, ctx.mcd_df
    utmos_scores = ctx.utmos is not None
    paper_rows = []

    def add(section, metric, value, ci=None, n=None, note=""):
        paper_rows.append({"section": section, "metric": metric, "value": value,
                            "ci_low": None if ci is None else ci[0],
                            "ci_high": None if ci is None else ci[1],
                            "n": n, "note": note})

    # --- system performance -------------------------------------------------------------------
    m, lo, hi = bootstrap_ci(final_eval_df["voiced_fraction"], n_boot=cfg.bootstrap_n)
    add("system", "voiced_fraction (mean)", round(m, 4), (round(lo, 4), round(hi, 4)),
        len(final_eval_df), "corrected completion definition; check censoring before quoting")
    k = int((final_eval_df["voiced_fraction"] < cfg.completion_fail_threshold).sum())
    lo, hi = wilson_ci(k, len(final_eval_df))
    add("system", f"failure rate (<{cfg.completion_fail_threshold})", round(k / len(final_eval_df), 4),
        (round(lo, 4), round(hi, 4)), len(final_eval_df), "Wilson interval; headline completion number")
    add("system", "f0_median (Hz)", round(float(final_eval_df["f0_median"].median()), 1), None,
        len(final_eval_df), f"65-400 Hz band; {int(final_eval_df['f0_octave_flag'].sum())} octave flags")

    if kept is not None and len(kept):
        col = f"mcd_{cfg.mcd_mode}"
        m, lo, hi = bootstrap_ci(kept[col], n_boot=cfg.bootstrap_n)
        add("system", f"MCD {cfg.mcd_mode} (dB)", round(m, 2), (round(lo, 2), round(hi, 2)),
            int(kept[col].notna().sum()),
            f"aligned, corrupt refs filtered ({int(mcd_df['excluded'].notna().sum())} excluded)")
        for diag in cfg.mcd_diagnostic_modes:
            dcol = f"mcd_{diag}"
            if dcol in kept.columns:
                add("methods", f"MCD {diag} (dB, diagnostic)", round(float(kept[dcol].mean()), 2),
                    None, len(kept),
                    f"corr(dur)={kept['abs_log_dur_ratio'].corr(kept[dcol]):+.3f} — not a fidelity metric")
        add("methods", "corrupt references excluded", int(mcd_df["excluded"].notna().sum()), None,
            len(mcd_df), "truncated MELD extractions, e.g. 52 chars against a 0.13 s recording")

    if utmos_scores:
        m, lo, hi = bootstrap_ci(final_eval_df["utmos"], n_boot=cfg.bootstrap_n)
        add("system", "UTMOS", round(m, 3), (round(lo, 3), round(hi, 3)), len(final_eval_df),
            "reference-free; no speaker assumption")

    # --- controls ------------------------------------------------------------------------------
    if zeroshot_df is not None:
        mg = final_eval_df.merge(zeroshot_df, on="utt_id", suffixes=("_ft", "_zs"))
        for col in ["voiced_fraction", "f0_iqr"] + (["utmos"] if utmos_scores else []):
            r = paired_bootstrap_diff(mg[f"{col}_ft"], mg[f"{col}_zs"], n_boot=cfg.n_boot_diff)
            if r.get("n"):
                add("control", f"fine-tuned - zero-shot: {col}", round(r["diff"], 4),
                    (round(r["ci_low"], 4), round(r["ci_high"], 4)), r["n"],
                    "SIGNIFICANT" if r["significant"] else "null")

    for key, label in [("synthetic_finetuned", "SER on fine-tuned synthetic"),
                        ("synthetic_zeroshot", "SER on zero-shot synthetic"),
                        ("real_meld", "SER on real MELD audio")]:
        if key in ctx.ser_stats and ctx.ser_stats[key]:
            st = ctx.ser_stats[key]
            add("ser", f"{label}: agreement", round(st["agreement"], 3), None, st["n"],
                f"kappa {st['kappa']:+.3f}, collapse {st['collapse_index']:.1%} on '{st['modal_label']}'")

    # --- mechanism rows -----------------------------------------------------------------------
    if utmos_scores and zeroshot_df is not None:
        _mg = final_eval_df.merge(zeroshot_df, on="utt_id", suffixes=("_ft", "_zs"))
        _mg = _mg[np.isfinite(_mg["f0_median_ft"]) & np.isfinite(_mg["f0_median_zs"])
                  & (_mg["f0_median_zs"] > 0)]
        if len(_mg):
            shift = 12.0 * np.log2(_mg["f0_median_ft"] / _mg["f0_median_zs"])
            add("mechanism", "pitch drift (semitones, median)", round(float(shift.median()), 2),
                None, len(_mg), "fine-tuned vs zero-shot, same utterances")
            add("mechanism", "corr(pitch shift, UTMOS change)",
                round(float(shift.corr(_mg["utmos_ft"] - _mg["utmos_zs"])), 3), None, len(_mg),
                "per-utterance; tests the mechanism rather than the 7-point emotion average")

    # --- v5 rows -------------------------------------------------------------------------------
    wer_results = ctx.wer_results
    if wer_results.get("fine_tuned") is not None and len(wer_results["fine_tuned"]):
        for key in ["fine_tuned", "zero_shot", "real_meld"]:
            w = wer_results.get(key)
            if w is not None and len(w):
                add("intelligibility", f"WER {key}", round(corpus_wer(w), 4),
                    None, len(w), "corpus-level, Whisper-normalised")
        if "zero_shot" in wer_results and len(wer_results["zero_shot"]):
            _w = wer_results["fine_tuned"].merge(wer_results["zero_shot"], on="utt_id", suffixes=("_ft", "_zs"))
            r = paired_bootstrap_diff(_w["wer_ft"], _w["wer_zs"], n_boot=cfg.n_boot_diff)
            if r.get("n"):
                add("control", "fine-tuned - zero-shot: WER", round(r["diff"], 4),
                    (round(r["ci_low"], 4), round(r["ci_high"], 4)), r["n"],
                    "SIGNIFICANT" if r["significant"] else "null")

    for key, st in ctx.e2v_stats.items():
        if st:
            add("ser", f"emotion2vec on {key}: agreement", round(st["agreement"], 3), None, st["n"],
                f"kappa {st['kappa']:+.3f}, collapse {st['collapse_index']:.1%} on '{st['modal_label']}'")

    for key, v in ctx.spk_results.items():
        if key == "centroid_diff":
            if v[0] is not None:
                add("speaker", "centroid similarity, fine-tuned - zero-shot", round(v[0], 4),
                    (round(v[1], 4), round(v[2], 4)), None, "negative = less stable identity")
        else:
            add("speaker", f"self-consistency {key}", round(v[0], 4), None, None, f"sd {v[1]:.3f}")

    paper_df = pd.DataFrame(paper_rows)
    paper_df.to_csv(f"{cfg.output_dir}/paper_results_table.csv", index=False)
    paper_df["run_tag"] = cfg.run_tag
    paper_df["eval_seed"] = cfg.eval_seed
    print(paper_df.to_string(index=False))

    _append_run_log(paper_df, cfg)

    print("\n" + "=" * 78)
    print("STILL REQUIRED BEFORE SUBMISSION — not obtainable from this pipeline:")
    print("  1. Human listening study: 7-way forced-choice emotion identification + naturalness")
    print("     MOS, ~15 raters, fine-tuned vs zero-shot vs real MELD anchors. This is the only")
    print("     direct evidence about expressiveness, which is the paper's actual subject.")
    print("  2. Re-run the ablations on the corrected metrics in a second session")
    print("     (skip_ablations=False), bumping run_tag and eval_seed.")
    print("=" * 78)
    return paper_df


def _append_run_log(paper_df, cfg):
    """Replications accumulate into one file. Bump cfg.run_tag and cfg.eval_seed per run and the
    spread across rows becomes the variance estimate the paper needs for its null results."""
    run_log_path = Path(cfg.output_dir) / "run_log.csv"
    if run_log_path.exists():
        prev = pd.read_csv(run_log_path)
        combined = pd.concat([prev[prev["run_tag"] != cfg.run_tag], paper_df], ignore_index=True)
    else:
        combined = paper_df
    combined.to_csv(run_log_path, index=False)

    n_runs = combined["run_tag"].nunique()
    if n_runs > 1:
        print(f"\n=== Stability across {n_runs} runs ===")
        piv = combined.pivot_table(index="metric", columns="run_tag", values="value")
        piv["spread"] = piv.max(axis=1) - piv.min(axis=1)
        piv["mean"] = combined.groupby("metric")["value"].mean()
        print(piv.round(4).to_string())
        print("\nAny effect smaller than its own spread across runs is not a finding.")


def write_manifest(output_dir) -> pd.DataFrame:
    output_files = []
    for root, _, files in os.walk(output_dir):
        for fname in sorted(files):
            fpath = Path(root) / fname
            output_files.append({"file": str(fpath.relative_to(output_dir)),
                                  "size_kb": round(fpath.stat().st_size / 1024, 1)})

    manifest_df = pd.DataFrame(output_files)
    manifest_df.to_csv(f"{output_dir}/manifest.csv", index=False)
    return manifest_df
