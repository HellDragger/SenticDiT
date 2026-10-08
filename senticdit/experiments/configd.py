"""Experiment "configd": data quantity at fixed compute (senticdit_configd.ipynb).

Two arms trained from the clean pretrained weights, identical except for the training pool:

|                | D                        | C'                         |
|----------------|--------------------------|----------------------------|
| Pool           | MELD train split only    | train+dev+test (pooled)    |
| Rank / targets | 16 / attention + shared AdaLN MLP | same              |
| Sampling       | class-balanced, reweighted within its own pool | same |
| Steps          | config_d_max_steps (2000)| same                       |
| Eval           | same held-out set, SAME noise draws | same            |

Nothing else is on the GPU during training: evaluation models (UTMOS, Whisper, emotion2vec) are
loaded only after an arm is trained and freed before the next one — the v5 run lacked this and
ran out of memory. Also runs the CPU-only MELD register check.

Expected time: ~10 min setup + 2 x (~1.9 h training + ~20 min generation and scoring) ~ 4.7 h.
Checkpoints every 500 steps; to resume, set `configd_resume_root` to the folder containing
`configd_D_train_only/` and `configd_C_pooled_matched/`.
"""
import gc
import subprocess
import sys
import time
from pathlib import Path

import jiwer
import librosa
import numpy as np
import pandas as pd
import torch

from ..context import RunContext
from ..data import add_class_weights, make_loader
from ..evaluation.ser import Emotion2Vec
from ..evaluation.utmos import UTMOSScorer
from ..evaluation.wer import WhisperWER
from ..generation import bracket_prompt, generate_and_score
from ..metrics import compute_health_metrics
from ..model import attach_lora, reset_transformer, summarize_lora_targets
from ..stats import describe_diff, min_detectable_effect, paired_bootstrap_diff
from ..training import train_lora
from ..utils import reseed, section, zip_outputs

ARMS = ("D_train_only", "C_pooled_matched")


def meld_register_check(ctx):
    """Where did the pitch drift come from? MELD's own register.

    Fine-tuning moved this voice from a median of ~143 Hz to ~244 Hz, made it less stable as an
    identity, and moved its UTMOS toward MELD's own. One explanation covers all three: the adapter
    learned MELD's speakers and recording conditions, not only its emotions. That predicts the
    fine-tuned register should sit near the register of MELD's training audio. CPU only.
    """
    cfg = ctx.cfg
    train_df = ctx.train_df
    meld_f0_rows = []
    # Same rows and order as the notebook's groupby(group_keys=False).apply(sample), written out
    # so it does not depend on whether this pandas version drops the grouping column in apply.
    sample = pd.concat([g.sample(n=min(len(g), cfg.meld_register_per_emotion), random_state=cfg.eval_seed)
                        for _, g in train_df[train_df["wav_path"].notna()].groupby("emotion")])
    for _, r in sample.iterrows():
        try:
            y, _ = librosa.load(r["wav_path"], sr=cfg.sample_rate, mono=True)
        except Exception:
            continue
        if len(y) < cfg.sample_rate:          # skip the truncated sub-second extractions
            continue
        m = compute_health_metrics(y, cfg.sample_rate, cfg)
        if np.isfinite(m["f0_median"]):
            meld_f0_rows.append({"emotion": r["emotion"], "f0_median": m["f0_median"],
                                 "octave_flag": m["f0_octave_flag"]})
    meld_f0 = pd.DataFrame(meld_f0_rows)
    meld_f0.to_csv(f"{cfg.output_dir}/meld_training_f0.csv", index=False)

    print(f"MELD training audio, stratified sample (n={len(meld_f0)}, 65-400 Hz band):")
    print(f"  median F0 {meld_f0['f0_median'].median():.1f} Hz | IQR "
          f"[{meld_f0['f0_median'].quantile(.25):.1f}, {meld_f0['f0_median'].quantile(.75):.1f}] Hz")
    print(f"  share above 200 Hz: {(meld_f0['f0_median'] > 200).mean():.1%}")
    print(meld_f0.groupby("emotion")["f0_median"].median().round(1).to_string())
    print("\nCompare: zero-shot ~143 Hz, fine-tuned ~244 Hz (v5 run 1).")
    print("If MELD's median sits near the fine-tuned value, the drift is the adapter absorbing the")
    print("training corpus's speaker distribution -- speaker leakage, not an emotion effect.")
    return meld_f0


# ---------------------------------------------------------------------------------------------
# Evaluation models, loaded only AFTER an arm is trained and freed before the next one
# ---------------------------------------------------------------------------------------------

def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gpu_report(tag):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"  [{tag}] GPU free {free/2**30:.2f} / {total/2**30:.2f} GiB")


def load_eval_models(cfg, device):
    m = {}
    try:
        m["utmos"] = UTMOSScorer(device)
    except Exception as e:
        print(f"  UTMOS unavailable ({e})")
    try:
        m["asr"] = WhisperWER(cfg.asr_model_id)
    except Exception as e:
        print(f"  Whisper unavailable ({e})")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "funasr", "jiwer"], check=False)
        m["e2v"] = Emotion2Vec(cfg.emotion2vec_model_id)
    except Exception as e:
        print(f"  emotion2vec unavailable ({e})")
    return m


def score_arm(df, m, cfg):
    """UTMOS, Whisper WER and emotion2vec prediction for every clip in df["wav"]."""
    out = df.copy()
    sr = cfg.sample_rate
    if "utmos" in m:
        out["utmos"] = [m["utmos"](w, sr) for w in out["wav"]]
    if "asr" in m:
        asr = m["asr"]
        refs, hyps = [], []
        for _, r in out.iterrows():
            hyp = asr.transcribe(r["wav"], sr)
            refs.append(asr.normalize(r["text"])); hyps.append(asr.normalize(hyp) or "<empty>")
        out["asr_ref"], out["asr_hyp"] = refs, hyps
        out["wer"] = [jiwer.wer(a, b) if a else np.nan for a, b in zip(refs, hyps)]
        out["wer_clipped"] = out["wer"].clip(upper=1.0)   # bounded: insertions can't dominate
    if "e2v" in m:
        out["e2v_pred"] = [m["e2v"].predict(w, sr) for w in out["wav"]]
        out["e2v_correct"] = (out["e2v_pred"] == out["emotion"]).astype(float)
    return out


# ---------------------------------------------------------------------------------------------
# Config D
# ---------------------------------------------------------------------------------------------

def run_config_d(ctx):
    cfg = ctx.cfg
    best_cfg_strength = float(cfg.cfg_strength)     # fixed at 2.0; the CFG sweep is a null
    _gpu_report("before Config D")

    train_df = ctx.train_df
    pools = {
        "D_train_only": train_df[train_df["split"] == "train"].reset_index(drop=True),
        "C_pooled_matched": train_df.reset_index(drop=True),
    }
    eff_batch = cfg.batch_size * cfg.grad_accum_steps
    print("Training pools:")
    for name, pool in pools.items():
        print(f"  {name:18s} {len(pool):6d} clips | {cfg.config_d_max_steps} steps x {eff_batch} "
              f"= {cfg.config_d_max_steps * eff_batch / max(len(pool), 1):.2f} epochs")

    reset_transformer(ctx.model, ctx.original_state)

    configd_records = []
    for arm, pool in pools.items():
        t0 = time.time()
        print(f"\n=== Config D experiment — arm {arm} ===")
        pool = add_class_weights(pool)

        reseed(cfg, f"configd_train_{arm}")
        attach_lora(ctx.model, ctx.original_state, cfg.config_d_rank, cfg,
                    include_adaln=cfg.config_d_include_adaln)
        summarize_lora_targets(ctx.model.transformer)
        _gpu_report("before training")

        loader = make_loader(pool, cfg)
        arm_ckpt = Path(cfg.output_dir) / "checkpoints" / f"configd_{arm}"
        arm_ckpt.mkdir(parents=True, exist_ok=True)
        resume_dir = Path(cfg.configd_resume_root) / f"configd_{arm}" if cfg.configd_resume_root else arm_ckpt
        losses_arm = train_lora(ctx.model, ctx.tokenizer, loader, cfg, ctx.device,
                                max_steps=cfg.config_d_max_steps, lr=cfg.learning_rate,
                                ckpt_dir=arm_ckpt, ckpt_every=cfg.config_d_ckpt_every,
                                resume_search_dir=resume_dir,
                                base_transformer_state=ctx.original_state)
        del loader
        _free_gpu()

        reseed(cfg, "configd_eval")                 # SAME seed for both arms -> matched noise draws
        df_arm = generate_and_score(ctx.model, ctx.tokenizer, ctx.holdout_df, cfg, ctx.device,
                                    emotions=cfg.emotions, n_per_emotion=cfg.eval_samples_per_emotion,
                                    cfg_strength=best_cfg_strength, steps=cfg.cfm_steps,
                                    prompt_fn=bracket_prompt, return_wavs=True)
        print(f"  generated {len(df_arm)} clips; loading evaluation models")
        evm = load_eval_models(cfg, ctx.device)
        df_arm = score_arm(df_arm, evm, cfg)
        del evm
        _free_gpu()

        df_arm["arm"] = arm
        df_arm["final_loss"] = float(np.mean(losses_arm[-200:])) if losses_arm else float("nan")
        configd_records.append(df_arm.drop(columns=["wav"]))
        pd.concat(configd_records, ignore_index=True).to_csv(
            f"{cfg.output_dir}/configd_per_utterance.csv", index=False)   # saved after each arm

        reset_transformer(ctx.model, ctx.original_state)
        print(f"  arm {arm} done in {(time.time() - t0) / 3600:.2f} h")

    configd_df = pd.concat(configd_records, ignore_index=True)
    d = configd_df[configd_df["arm"] == "D_train_only"]
    c = configd_df[configd_df["arm"] == "C_pooled_matched"]
    mg = c.merge(d, on="utt_id", suffixes=("_C", "_D"))
    print(f"\n=== Pooled (C') minus train-only (D), paired by utterance, n={len(mg)} ===")
    rows = []
    for col, unit in [("voiced_fraction", ""), ("f0_median", " Hz"), ("f0_iqr", " Hz"),
                      ("utmos", ""), ("wer_clipped", ""), ("e2v_correct", "")]:
        if f"{col}_C" not in mg.columns:
            continue
        res = paired_bootstrap_diff(mg[f"{col}_C"], mg[f"{col}_D"], n_boot=cfg.n_boot_diff)
        if not res.get("n"):
            continue
        sd = float(np.nanstd(mg[f"{col}_C"] - mg[f"{col}_D"], ddof=1))
        mde = min_detectable_effect(sd, res["n"], paired=True)
        print(f"  {col:16s} {describe_diff(res, 'pooled', 'train-only', unit)}")
        print(f"  {'':16s} MDE {mde:.4g}{unit}")
        rows.append({"metric": col, **res, "mde": mde})

    rare = mg[mg["emotion_C"].isin(["fear", "disgust"])]
    for col in ["utmos", "e2v_correct"]:
        if f"{col}_C" in rare.columns and len(rare):
            res = paired_bootstrap_diff(rare[f"{col}_C"], rare[f"{col}_D"], n_boot=cfg.n_boot_diff)
            print(f"  rare classes (fear+disgust) {col}: {describe_diff(res, 'pooled', 'train-only')}")
            rows.append({"metric": f"{col}_fear_disgust", **res})

    if "wer_C" in mg.columns:
        for arm_key in ["C", "D"]:
            print(f"  corpus WER {arm_key}: "
                  f"{jiwer.wer(mg[f'asr_ref_{arm_key}'].tolist(), mg[f'asr_hyp_{arm_key}'].tolist()):.3f}")
    print(f"  final training loss (mean of last 200 micro-batches): "
          f"C' {c['final_loss'].iloc[0]:.4f} | D {d['final_loss'].iloc[0]:.4f}")

    configd_cmp = pd.DataFrame(rows)
    configd_cmp.to_csv(f"{cfg.output_dir}/configd_comparison.csv", index=False)
    print("\nIf every interval spans zero: at this compute budget, adding dev+test to the training")
    print("pool did not measurably change the output. Report as a null with the MDE.")
    return configd_cmp


def run(cfg):
    ctx = RunContext(cfg)
    ctx.prepare_data()
    ctx.load_model()
    section("16.6b · Where did the pitch drift come from? MELD's own register")
    meld_register_check(ctx)
    section("16.7 · Config D — does more distinct data help, at fixed compute?")
    run_config_d(ctx)
    if cfg.zip_results:
        zip_outputs(Path(cfg.output_dir).parent / "configd_results", cfg.output_dir,
                    ["meld_training_f0.csv", "configd_per_utterance.csv", "configd_comparison.csv"]
                    + [f"checkpoints/configd_{arm}" for arm in ARMS])
    return ctx
