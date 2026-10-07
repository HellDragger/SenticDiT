"""Mel-Cepstral Distortion — plain DTW, with corrupt references filtered.

`dtw_sl` is not a fidelity metric: pymcd returns `cof x (DTW-aligned MCD)` with
`cof = longer/shorter` frame counts, so it is fidelity times a duration penalty (it correlated
with duration mismatch at +0.846 / +0.747 across two v3 runs). `plain` zero-pads and saturates
around 12-17 dB. Both are kept only as diagnostics.

14% of MELD references are truncated extractions (one 52-character sentence has a 0.13 s
recording). They are filtered on reference duration and duration ratio, the exclusions are
counted, and the unfiltered value is reported alongside.
"""
import subprocess
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from ..generation import generate_speech
from ..metrics import compute_health_metrics
from ..stats import bootstrap_ci
from ..utils import reseed


def _load_calculate_mcd():
    try:
        from pymcd.mcd import Calculate_MCD
        return Calculate_MCD
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymcd"], check=False)
        try:
            from pymcd.mcd import Calculate_MCD
            return Calculate_MCD
        except ImportError:
            return None


def _register_distance_semitones(path_a, path_b, sr, cfg):
    """Pitch-register gap in semitones. MELD references are cast members; the synthesised voice
    is not. Above ~5 semitones, MCD against that reference is substantially speaker identity."""
    try:
        ya, _ = librosa.load(path_a, sr=sr, mono=True)
        yb, _ = librosa.load(path_b, sr=sr, mono=True)
        ma = compute_health_metrics(ya, sr, cfg)["f0_median"]
        mb = compute_health_metrics(yb, sr, cfg)["f0_median"]
        if np.isfinite(ma) and np.isfinite(mb) and ma > 0 and mb > 0:
            return float(abs(12.0 * np.log2(mb / ma)))
    except Exception:
        pass
    return float("nan")


def mcd_evaluation(ctx):
    """Returns (mcd_df, kept) or (None, None) if pymcd is unavailable."""
    cfg = ctx.cfg
    Calculate_MCD = _load_calculate_mcd()
    if Calculate_MCD is None or len(ctx.holdout_df) == 0:
        print("pymcd unavailable or no held-out references — skipping MCD.")
        return None, None

    reseed(cfg, "mcd")
    modes = [cfg.mcd_mode] + [m for m in cfg.mcd_diagnostic_modes if m != cfg.mcd_mode]
    calcs = {m: Calculate_MCD(MCD_mode=m) for m in modes}
    main_col = f"mcd_{cfg.mcd_mode}"

    gen_path = "/tmp/_mcd_gen.wav"
    mcd_rows = []
    for emotion in cfg.emotions:
        rows = ctx.holdout_df[ctx.holdout_df["emotion"] == emotion]
        if cfg.mcd_refs_per_emotion:
            rows = rows.head(cfg.mcd_refs_per_emotion)
        for idx, ref_row in rows.iterrows():
            text = str(ref_row["utterance"]).strip()
            if not text or not ref_row.get("wav_path"):
                continue
            generate_speech(ctx.model, ctx.tokenizer, text, cfg, ctx.device, emotion=emotion,
                            cfg_strength=ctx.best_cfg_strength, out_path=gen_path)
            rec = {"emotion": emotion, "utt_id": idx, "n_chars": len(text), "excluded": None}
            try:
                ref_dur = librosa.get_duration(path=ref_row["wav_path"])
                gen_dur = librosa.get_duration(path=gen_path)
                ratio = gen_dur / max(ref_dur, 1e-6)
                rec.update({"ref_dur": ref_dur, "gen_dur": gen_dur, "duration_ratio": ratio,
                            "abs_log_dur_ratio": abs(np.log(max(ratio, 1e-6))),
                            "cof": max(ratio, 1 / max(ratio, 1e-6))})
                rec["register_semitones"] = _register_distance_semitones(
                    ref_row["wav_path"], gen_path, cfg.sample_rate, cfg)

                # --- corrupt-reference filter -------------------------------------------------
                if ref_dur < cfg.mcd_min_ref_dur:
                    rec["excluded"] = f"reference {ref_dur:.2f}s < {cfg.mcd_min_ref_dur}s (truncated MELD extraction)"
                elif not (1 / cfg.mcd_max_dur_ratio <= ratio <= cfg.mcd_max_dur_ratio):
                    rec["excluded"] = f"duration ratio {ratio:.2f} outside [{1/cfg.mcd_max_dur_ratio:.2f}, {cfg.mcd_max_dur_ratio:.2f}]"

                for name, c in calcs.items():
                    rec[f"mcd_{name}"] = c.calculate_mcd(ref_row["wav_path"], gen_path)
            except Exception as e:
                rec["error"] = str(e)
            mcd_rows.append(rec)

    mcd_df = pd.DataFrame(mcd_rows)
    mcd_df.to_csv(f"{cfg.output_dir}/mcd_per_utterance.csv", index=False)

    kept = mcd_df[mcd_df["excluded"].isna()]
    dropped = mcd_df[mcd_df["excluded"].notna()]
    print(f"MCD over {len(mcd_df)} held-out utterances "
          f"({len(kept)} kept, {len(dropped)} excluded as corrupt references)\n")
    if len(dropped):
        print("Excluded references (these are broken MELD audio, not generation failures):")
        for reason, grp in dropped.groupby(dropped["excluded"].str.split("(").str[0]):
            print(f"  {len(grp):3d}x {reason.strip()}")
        print("  worst cases:")
        print(dropped.nlargest(3, "duration_ratio")[
            ["emotion", "n_chars", "ref_dur", "gen_dur", "duration_ratio"]].round(2).to_string(index=False))
        print()

    print(f"Per-emotion {cfg.mcd_mode} MCD with bootstrapped 95% CIs (filtered):")
    for emotion in cfg.emotions:
        vals = kept.loc[kept["emotion"] == emotion, main_col].dropna()
        if not len(vals):
            continue
        m, lo, hi = bootstrap_ci(vals, n_boot=cfg.bootstrap_n)
        print(f"  {emotion:9s} {m:6.2f} dB   95% CI [{lo:5.2f}, {hi:5.2f}]   n={len(vals)}")
    m, lo, hi = bootstrap_ci(kept[main_col].dropna(), n_boot=cfg.bootstrap_n)
    print(f"  {'OVERALL':9s} {m:6.2f} dB   95% CI [{lo:5.2f}, {hi:5.2f}]   n={int(kept[main_col].notna().sum())}")

    if cfg.mcd_report_unfiltered_too:
        mu, lu, hu = bootstrap_ci(mcd_df[main_col].dropna(), n_boot=cfg.bootstrap_n)
        print(f"  {'(unfiltered)':9s} {mu:6.2f} dB   95% CI [{lu:5.2f}, {hu:5.2f}]   "
              f"n={int(mcd_df[main_col].notna().sum())}   <- effect of the corrupt references")

    # --- what each variant tracks -----------------------------------------------------------
    print("\n=== What each MCD variant is actually tracking (filtered set) ===")
    for name in modes:
        col = f"mcd_{name}"
        if col in kept.columns:
            d = kept[["abs_log_dur_ratio", "register_semitones", col]].dropna()
            if len(d) > 3:
                print(f"  {col:14s} mean {d[col].mean():6.2f} dB | "
                      f"corr(dur) {d['abs_log_dur_ratio'].corr(d[col]):+.3f} | "
                      f"corr(register) {d['register_semitones'].corr(d[col]):+.3f}")
    print("  Expect: dtw near zero on both. dtw_sl strongly positive on duration (the length")
    print("  penalty is multiplicative). plain weakly correlated because it saturates.")

    # --- validate the recovery identity used on v3's saved data ------------------------------
    if "mcd_dtw_sl" in kept.columns and "cof" in kept.columns:
        rec_est = kept["mcd_dtw_sl"] / kept["cof"]
        d = pd.concat([rec_est.rename("est"), kept[main_col].rename("actual")], axis=1).dropna()
        if len(d) > 3:
            print(f"\n  Recovery check: dtw_sl/cof vs measured dtw -> "
                  f"r={d['est'].corr(d['actual']):+.3f}, "
                  f"mean abs diff {np.abs(d['est'] - d['actual']).mean():.2f} dB")
            print("  (This identity is what let v3's clean MCD be reconstructed without re-running.)")

    print(f"\n  median |duration ratio - 1| (filtered) = "
          f"{(kept['duration_ratio'] - 1).abs().median():.3f}")
    print(f"  clips with register gap > 5 semitones: "
          f"{int((kept['register_semitones'] > 5).sum())}/{len(kept)}")
    print("  A high residual MCD with near-zero correlations means what remains is genuine")
    print("  spectral distance to a different speaker recorded in a different environment.")
    print("  Against MELD that floor is high; state it as a scope limit and lean on UTMOS.")
    return mcd_df, kept


def alignment_artefact(ctx, mcd_df=None):
    """Score a generated clip against time-stretched copies of ITSELF. Same speaker, same words,
    so every dB is alignment artefact. v2's reported per-emotion range (6.40-24.10 dB) reproduces
    from duration mismatch alone."""
    cfg = ctx.cfg
    Calculate_MCD = _load_calculate_mcd()
    if Calculate_MCD is None:
        return None

    demo_src = Path(cfg.output_dir) / "demo_neutral.wav"
    if not demo_src.exists():
        generate_speech(ctx.model, ctx.tokenizer, cfg.demo_text,
                        cfg, ctx.device, emotion="neutral", cfg_strength=ctx.best_cfg_strength,
                        out_path=str(demo_src))

    y_ref, sr_ref = librosa.load(str(demo_src), sr=22050, mono=True)   # pymcd's internal rate
    sf.write("/tmp/_artefact_ref.wav", y_ref, sr_ref)

    modes = ["plain", "dtw", "dtw_sl"]
    artefact_calcs = {m: Calculate_MCD(MCD_mode=m) for m in modes}
    artefact_rows = []
    for stretch in [1.00, 1.05, 1.15, 1.30, 1.60]:
        ys = librosa.effects.time_stretch(y_ref, rate=stretch)
        sf.write("/tmp/_artefact_stretched.wav", ys, sr_ref)
        rec = {"stretch": stretch}
        for m in modes:
            rec[m] = artefact_calcs[m].calculate_mcd(
                "/tmp/_artefact_ref.wav", "/tmp/_artefact_stretched.wav")
        artefact_rows.append(rec)

    artefact_df = pd.DataFrame(artefact_rows).set_index("stretch").round(2)
    artefact_df.to_csv(f"{cfg.output_dir}/mcd_alignment_artefact.csv")
    print("MCD between a clip and a TIME-STRETCHED COPY OF ITSELF")
    print("(identical speaker, identical content — every dB here is alignment artefact)\n")
    print(artefact_df.to_string())

    fig, ax = plt.subplots(figsize=(6, 4))
    for m, style in zip(modes, ["o-", "s--", "^:"]):
        ax.plot(artefact_df.index, artefact_df[m], style, label=m)
    if mcd_df is not None and len(mcd_df):
        lo, hi = mcd_df[f"mcd_{cfg.mcd_mode}"].min(), mcd_df[f"mcd_{cfg.mcd_mode}"].max()
        ax.axhspan(lo, hi, alpha=0.12, color="grey",
                   label=f"observed {cfg.mcd_mode} range")
    ax.set_xlabel("time-stretch factor applied to an identical copy")
    ax.set_ylabel("MCD (dB)")
    ax.set_title("Unaligned MCD is a duration-mismatch meter")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{cfg.output_dir}/mcd_alignment_artefact.png", dpi=150)
    plt.show()
    return artefact_df
