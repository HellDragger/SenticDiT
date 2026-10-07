"""Reference-free quality (UTMOS) and the pitch-drift mechanism test.

MCD compares against a different speaker recorded from broadcast television, so it cannot carry a
fidelity claim. UTMOS scores audio on its own, which makes the fine-tuned-vs-zero-shot comparison
speaker-independent and paired. The real-MELD anchor is stratified across emotions and
length-filtered, and reported with caveats rather than used as a floor.
"""
import librosa
import numpy as np
import pandas as pd
import torch

from ..stats import bootstrap_ci, describe_diff, paired_bootstrap_diff


class UTMOSScorer:
    def __init__(self, device):
        self.device = device
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                                    trust_repo=True).to(device).eval()

    @torch.no_grad()
    def __call__(self, wav, sr):
        w = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
        t = torch.from_numpy(np.asarray(w, dtype=np.float32)).unsqueeze(0).to(self.device)
        return float(self.model(t, 16000).item())


def utmos_evaluation(ctx):
    """Adds a `utmos` column to ctx.final_eval_df (and ctx.zeroshot_df) in place.
    Returns the scorer, or None if UTMOS could not be loaded."""
    cfg = ctx.cfg
    if not cfg.run_utmos:
        return None
    try:
        score_utmos = UTMOSScorer(ctx.device)
        final_eval_df, zeroshot_df = ctx.final_eval_df, ctx.zeroshot_df

        final_eval_df["utmos"] = [score_utmos(w, cfg.sample_rate) for w in final_eval_df["wav"]]
        print("UTMOS, fine-tuned (higher is better, 1-5):")
        print(final_eval_df.groupby("emotion")["utmos"]
              .agg(["mean", "std", "count"]).round(3).to_string())
        m, lo, hi = bootstrap_ci(final_eval_df["utmos"], n_boot=cfg.bootstrap_n)
        print(f"  OVERALL {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

        if zeroshot_df is not None:
            zeroshot_df["utmos"] = [
                score_utmos(librosa.load(p, sr=cfg.sample_rate, mono=True)[0], cfg.sample_rate)
                for p in zeroshot_df["wav_file"]]
            mg = final_eval_df.merge(zeroshot_df, on="utt_id", suffixes=("_ft", "_zs"))
            res = paired_bootstrap_diff(mg["utmos_ft"], mg["utmos_zs"], n_boot=cfg.n_boot_diff)
            print("\n  " + describe_diff(res, "fine-tuned", "zero-shot"))
            print("  Two v3 runs: -0.951 [-1.08, -0.82] and -1.027 [-1.16, -0.89].")

            # re-save WITH the utmos column; the zero-shot stage wrote this CSV before UTMOS existed
            zeroshot_df.drop(columns=["wav"], errors="ignore").to_csv(
                f"{cfg.output_dir}/health_metrics_zeroshot.csv", index=False)

        # --- anchor: stratified and length-filtered ------------------------------------------
        anchor_rows = []
        for emotion in cfg.emotions:
            pool = ctx.holdout_df[(ctx.holdout_df["emotion"] == emotion) & ctx.holdout_df["wav_path"].notna()]
            taken = 0
            for _, r in pool.iterrows():
                if taken >= cfg.utmos_anchor_per_emotion:
                    break
                try:
                    y, _ = librosa.load(r["wav_path"], sr=cfg.sample_rate, mono=True)
                except Exception:
                    continue
                dur = len(y) / cfg.sample_rate
                if dur < cfg.utmos_anchor_min_dur:
                    continue
                anchor_rows.append({"emotion": emotion, "dur": dur,
                                     "utmos": score_utmos(y, cfg.sample_rate)})
                taken += 1
        anchor_df = pd.DataFrame(anchor_rows)
        if len(anchor_df):
            am, alo, ahi = bootstrap_ci(anchor_df["utmos"], n_boot=cfg.bootstrap_n)
            print(f"\n  Real MELD anchor: {am:.3f}  95% CI [{alo:.3f}, {ahi:.3f}]  "
                  f"n={len(anchor_df)}")
            print("   per emotion: " + ", ".join(
                f"{e} {v:.2f}" for e, v in anchor_df.groupby("emotion")["utmos"].mean().items()))
            print("   Caveat, and state it in the paper: UTMOS is trained on clean read speech and")
            print("   penalises background noise. MELD is broadcast television with laugh tracks,")
            print("   music and overlapping speakers. Treat this as a property of the training")
            print("   corpus, not as a quality floor the system should reach.")
            anchor_df.to_csv(f"{cfg.output_dir}/utmos_anchor.csv", index=False)

        final_eval_df.drop(columns=["wav"]).to_csv(
            f"{cfg.output_dir}/health_metrics_final.csv", index=False)
        return score_utmos
    except Exception as e:
        print(f"UTMOS unavailable ({e}) — skipping. Needs internet enabled on the Kaggle session.")
        return None


def pitch_drift_mechanism(ctx):
    """Per utterance: does the size of the pitch shift between the zero-shot and fine-tuned
    rendering predict the size of the UTMOS drop on that same sentence? A correlation across
    seven emotion averages could be coincidence; this cannot."""
    cfg = ctx.cfg
    if ctx.utmos is None or ctx.zeroshot_df is None:
        return None
    final_eval_df = ctx.final_eval_df
    mg = final_eval_df.merge(ctx.zeroshot_df, on="utt_id", suffixes=("_ft", "_zs"))
    mg = mg[np.isfinite(mg["f0_median_ft"]) & np.isfinite(mg["f0_median_zs"])
            & (mg["f0_median_zs"] > 0)].copy()

    # signed pitch shift in semitones, per utterance
    mg["pitch_shift_st"] = 12.0 * np.log2(mg["f0_median_ft"] / mg["f0_median_zs"])
    mg["utmos_delta"] = mg["utmos_ft"] - mg["utmos_zs"]

    print("Per-utterance pitch drift caused by fine-tuning:")
    print(f"  median shift   {mg['pitch_shift_st'].median():+.2f} semitones")
    print(f"  IQR            [{mg['pitch_shift_st'].quantile(.25):+.2f}, "
          f"{mg['pitch_shift_st'].quantile(.75):+.2f}]")
    print(f"  clips shifted up by more than an octave: "
          f"{int((mg['pitch_shift_st'] > 12).sum())}/{len(mg)}")
    print(f"  zero-shot median F0 {mg['f0_median_zs'].median():.1f} Hz  ->  "
          f"fine-tuned {mg['f0_median_ft'].median():.1f} Hz")

    print("\nDoes the per-utterance drift predict the per-utterance quality loss?")
    r_p = mg["pitch_shift_st"].corr(mg["utmos_delta"])
    r_s = mg["pitch_shift_st"].corr(mg["utmos_delta"], method="spearman")
    print(f"  corr(pitch shift, UTMOS change)  Pearson {r_p:+.3f} | Spearman {r_s:+.3f}  n={len(mg)}")
    print(f"  corr(|pitch shift|, UTMOS change) Pearson "
          f"{mg['pitch_shift_st'].abs().corr(mg['utmos_delta']):+.3f}")
    print(f"  corr(f0_median_ft, utmos_ft)      Pearson "
          f"{mg['f0_median_ft'].corr(mg['utmos_ft']):+.3f}   (v3 runs: -0.309)")
    print(f"  corr(f0_iqr_ft,    utmos_ft)      Pearson "
          f"{mg['f0_iqr_ft'].corr(mg['utmos_ft']):+.3f}")

    # quartiles of drift vs quality loss -- the readable version of the same thing
    mg["drift_q"] = pd.qcut(mg["pitch_shift_st"], 4,
                             labels=["Q1 least drift", "Q2", "Q3", "Q4 most drift"])
    print("\nUTMOS change by pitch-drift quartile:")
    print(mg.groupby("drift_q", observed=True)[
        ["pitch_shift_st", "utmos_zs", "utmos_ft", "utmos_delta"]].mean().round(3).to_string())

    print("\nHow much is the 400 Hz primary band clipping?")
    print(f"  clips with f0_median within 5% of the bound: "
          f"{int(final_eval_df['f0_at_bound'].sum())}/{len(final_eval_df)}")
    wide = final_eval_df["f0_median_wide"].dropna()
    narrow = final_eval_df["f0_median"].dropna()
    if len(wide) and len(narrow):
        print(f"  median F0 at 65-400 Hz: {narrow.median():.1f} Hz | "
              f"at 65-{cfg.f0_diagnostic_ceiling_hz:.0f} Hz: {wide.median():.1f} Hz")
        print("  A large gap means the primary band is truncating a real drift, not just")
        print("  suppressing octave errors. Report the wide-band figure as the drift measure.")

    emo_col = next((c for c in ("emotion_ft", "emotion") if c in mg.columns), None)
    out_cols = [c for c in ["utt_id", emo_col, "f0_median_zs", "f0_median_ft",
                             "pitch_shift_st", "utmos_zs", "utmos_ft", "utmos_delta"]
                if c and c in mg.columns]
    mg[out_cols].to_csv(f"{cfg.output_dir}/pitch_drift_mechanism.csv", index=False)
    return mg
