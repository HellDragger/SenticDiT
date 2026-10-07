"""Config D — does more distinct data help, at fixed compute?

Two arms trained from the clean pretrained weights, identical except for the training pool:

|                | D                        | C'                         |
|----------------|--------------------------|----------------------------|
| Pool           | MELD train split only    | train+dev+test (pooled)    |
| Rank / targets | config_d_rank / attn QKV | same                       |
| Sampling       | class-balanced, reweighted within its own pool | same |
| Steps          | config_d_max_steps       | same                       |
| Eval           | same held-out set, SAME noise draws | same            |

C' is retrained rather than reused because the main run used a different step budget and target
set. Each arm checkpoints every `config_d_ckpt_every` steps so an interrupted session can resume.
After this stage the model no longer carries the main-run adapter (`lora_final/` is unaffected).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import add_class_weights, make_loader
from .generation import evaluate_holdout
from .model import attach_lora, reset_transformer, summarize_lora_targets
from .stats import describe_diff, min_detectable_effect, paired_bootstrap_diff
from .training import train_lora
from .utils import reseed


def run_config_d(ctx):
    cfg = ctx.cfg
    if not cfg.run_config_d:
        print("cfg.run_config_d=False — skipping the data-quantity experiment.")
        return None

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
        print("     " + ", ".join(f"{e} {n}" for e, n in pool["emotion"].value_counts().items()))

    # unload the main-run adapter; restore clean pretrained weights
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

        arm_ckpt = Path(cfg.output_dir) / "checkpoints" / f"configd_{arm}"
        arm_ckpt.mkdir(parents=True, exist_ok=True)
        losses_arm = train_lora(ctx.model, ctx.tokenizer, make_loader(pool, cfg), cfg, ctx.device,
                                max_steps=cfg.config_d_max_steps, lr=cfg.learning_rate,
                                ckpt_dir=arm_ckpt, ckpt_every=cfg.config_d_ckpt_every,
                                resume_search_dir=arm_ckpt,
                                base_transformer_state=ctx.original_state)

        reseed(cfg, "configd_eval")                    # SAME seed for both arms -> matched noise draws
        df_arm = evaluate_holdout(ctx.model, ctx.tokenizer, ctx.holdout_df, cfg, ctx.device,
                                  cfg_strength=ctx.best_cfg_strength, return_wavs=True)
        if ctx.utmos is not None:
            df_arm["utmos"] = [ctx.utmos(w, cfg.sample_rate) for w in df_arm["wav"]]
        if ctx.asr is not None:
            w = ctx.asr.wer_rows(df_arm, lambda r: (r["wav"], cfg.sample_rate), f"WER {arm}")
            if len(w):
                df_arm = df_arm.merge(w[["utt_id", "wer", "cer"]], on="utt_id", how="left")
        try:
            df_arm["ser_pred"] = [ctx.ser.predict(w, cfg.sample_rate) for w in df_arm["wav"]]
        except Exception:
            pass
        df_arm["arm"] = arm
        df_arm["final_loss"] = float(np.mean(losses_arm[-200:])) if losses_arm else float("nan")
        configd_records.append(df_arm.drop(columns=["wav"]))

        reset_transformer(ctx.model, ctx.original_state)
        print(f"  arm {arm} done in {(time.time() - t0) / 3600:.2f} h")

    configd_df = pd.concat(configd_records, ignore_index=True)
    configd_df.to_csv(f"{cfg.output_dir}/configd_per_utterance.csv", index=False)

    d = configd_df[configd_df["arm"] == "D_train_only"]
    c = configd_df[configd_df["arm"] == "C_pooled_matched"]
    mg = c.merge(d, on="utt_id", suffixes=("_C", "_D"))
    print(f"\n=== Pooled (C') minus train-only (D), paired by utterance, n={len(mg)} ===")
    rows = []
    metrics = [("voiced_fraction", ""), ("f0_median", " Hz"), ("f0_iqr", " Hz")]
    metrics += [("utmos", "")] if "utmos_C" in mg.columns else []
    metrics += [("wer", "")] if "wer_C" in mg.columns else []
    for col, unit in metrics:
        res = paired_bootstrap_diff(mg[f"{col}_C"], mg[f"{col}_D"], n_boot=cfg.n_boot_diff)
        if not res.get("n"):
            continue
        sd = float(np.nanstd(mg[f"{col}_C"] - mg[f"{col}_D"], ddof=1))
        mde = min_detectable_effect(sd, res["n"], paired=True)
        print(f"  {col:16s} {describe_diff(res, 'pooled', 'train-only', unit)}")
        print(f"  {'':16s} MDE {mde:.4g}{unit}")
        rows.append({"metric": col, **res, "mde": mde})

    # the original claim was specifically about the rarest classes
    rare = mg[mg["emotion_C"].isin(["fear", "disgust"])]
    if "utmos_C" in rare.columns and len(rare):
        res = paired_bootstrap_diff(rare["utmos_C"], rare["utmos_D"], n_boot=cfg.n_boot_diff)
        print(f"\n  rare classes only (fear+disgust) UTMOS: {describe_diff(res, 'pooled', 'train-only')}")
        rows.append({"metric": "utmos_fear_disgust", **res})
    if "ser_pred_C" in mg.columns:
        acc_c = float((mg["ser_pred_C"] == mg["emotion_C"]).mean())
        acc_d = float((mg["ser_pred_D"] == mg["emotion_D"]).mean())
        print(f"  wav2vec2 SER agreement: pooled {acc_c:.3f} vs train-only {acc_d:.3f} "
              f"(classifier is biased on synthetic speech; descriptive only)")

    configd_cmp = pd.DataFrame(rows)
    configd_cmp.to_csv(f"{cfg.output_dir}/configd_comparison.csv", index=False)
    print("\nAn interval spanning zero on every metric means: at this compute budget, adding")
    print("dev+test to the training pool did not measurably change the output. Report it as a null")
    print("with the MDE, not as support for or against the original pooling claim.")
    return configd_cmp
