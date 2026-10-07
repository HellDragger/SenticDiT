"""Short-budget ablations: sampling strategy, LoRA rank, LoRA target modules.

Every arm trains `cfg.ablation_max_steps` from the clean pretrained weights and is scored on the
same held-out set. Each is short enough to re-run within one session, so none of them use the
checkpoint/resume system. With `cfg.skip_ablations=True` (and the overrides set) they are no-ops.
"""
import pandas as pd

from .data import make_loader
from .generation import evaluate_holdout
from .model import attach_lora, reset_transformer, summarize_lora_targets
from .stats import bootstrap_ci
from .training import train_lora


def _train_and_score(ctx, loader):
    cfg = ctx.cfg
    train_lora(ctx.model, ctx.tokenizer, loader, cfg, ctx.device,
               max_steps=cfg.ablation_max_steps, lr=cfg.learning_rate)
    return evaluate_holdout(ctx.model, ctx.tokenizer, ctx.holdout_df, cfg, ctx.device)


def sampling_ablation(ctx):
    """Does class-balanced sampling fix the minority-class (fear / disgust) failures?"""
    cfg = ctx.cfg
    if cfg.skip_ablations:
        print("cfg.skip_ablations=True — skipping the sampling ablation "
              "(see ablation_sampling_comparison.csv from your earlier run for these results).")
        return None

    arms = []
    for arm, balanced in [("naive", False), ("balanced", True)]:
        label = "A: naive (unweighted)" if arm == "naive" else "B: class-balanced"
        print(f"=== Ablation Arm {label} sampling ===")
        attach_lora(ctx.model, ctx.original_state, cfg.lora_r, cfg)
        df = _train_and_score(ctx, make_loader(ctx.train_df, cfg, balanced=balanced,
                                               dataset=ctx.train_dataset))
        df["arm"] = arm
        arms.append(df)
        reset_transformer(ctx.model, ctx.original_state)
        print(f"Arm {label[0]} done, transformer reset to clean pretrained state.")

    ablation_df = pd.concat(arms, ignore_index=True)
    ablation_summary = ablation_df.groupby(["arm", "emotion"])[["completion_ratio", "f0_mean", "f0_std", "rms"]].mean().round(3)
    print(ablation_summary)

    print("\n--- Bootstrap 95% CI on completion_ratio, pooled across emotions ---")
    for arm in ["naive", "balanced"]:
        vals = ablation_df[ablation_df["arm"] == arm]["completion_ratio"].values
        mean, lo, hi = bootstrap_ci(vals, n_boot=cfg.bootstrap_n)
        print(f"{arm:10s}: mean={mean:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  n={len(vals)}")

    print("\n--- Same, restricted to the two rarest MELD classes (fear, disgust) ---")
    for arm in ["naive", "balanced"]:
        vals = ablation_df[(ablation_df["arm"] == arm) & (ablation_df["emotion"].isin(["fear", "disgust"]))]["completion_ratio"].values
        mean, lo, hi = bootstrap_ci(vals, n_boot=cfg.bootstrap_n)
        print(f"{arm:10s}: mean={mean:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  n={len(vals)}")

    ablation_df.to_csv(f"{cfg.output_dir}/ablation_sampling_comparison.csv", index=False)
    return ablation_df


def rank_ablation(ctx) -> int:
    """Parameter-efficiency trade-off: completion_ratio / f0_std vs trainable-parameter count."""
    cfg = ctx.cfg
    if cfg.best_lora_rank_override is not None:
        print(f"cfg.best_lora_rank_override set — using r={cfg.best_lora_rank_override} "
              f"without re-running the rank ablation.")
        return cfg.best_lora_rank_override
    if cfg.skip_ablations:
        print(f"cfg.skip_ablations=True and no override given — falling back to cfg.lora_r={cfg.lora_r}. "
              f"Set cfg.best_lora_rank_override to the rank you already found instead, if you have one.")
        return cfg.lora_r

    rank_results = []
    for r in cfg.lora_rank_sweep:
        print(f"=== Rank ablation: r={r} ===")
        attach_lora(ctx.model, ctx.original_state, r, cfg)
        n_trainable = sum(p.numel() for p in ctx.model.transformer.parameters() if p.requires_grad)
        df_r = _train_and_score(ctx, make_loader(ctx.train_df, cfg, dataset=ctx.train_dataset))
        df_r["lora_r"] = r
        df_r["trainable_params"] = n_trainable
        rank_results.append(df_r)
        reset_transformer(ctx.model, ctx.original_state)

    rank_ablation_df = pd.concat(rank_results, ignore_index=True)
    rank_summary = rank_ablation_df.groupby("lora_r")[["completion_ratio", "f0_std", "rms", "trainable_params"]].mean().round(4)
    print(rank_summary)
    rank_ablation_df.to_csv(f"{cfg.output_dir}/ablation_lora_rank.csv", index=False)

    rank_candidates = rank_summary[rank_summary["f0_std"] < cfg.f0_std_sanity_bound]
    pool = rank_candidates if len(rank_candidates) else rank_summary
    # Break near-ties on completion_ratio by f0_std (lower = more stable pitch) instead of
    # arbitrary index order — plain idxmax() previously picked r=16 over r=64 despite them
    # tying on completion_ratio and r=64 being clearly better on f0_std.
    best_completion = pool["completion_ratio"].max()
    tied = pool[pool["completion_ratio"] >= best_completion - cfg.rank_tie_tolerance]
    best_rank = int(tied["f0_std"].idxmin())
    print(f"\nSelected LoRA rank for the main run: r={best_rank} "
          f"(candidates within {cfg.rank_tie_tolerance} completion_ratio of the best were "
          f"tie-broken by lowest f0_std)")
    return best_rank


def target_ablation(ctx, best_rank: int) -> bool:
    """Attention-only LoRA vs attention + the AdaLN conditioning MLP, compared on f0_std.

    Don't point resume_search_dir at an attention-only checkpoint when testing the AdaLN variant:
    the adapter shapes differ. Main-run checkpoints are namespaced by config to prevent this.
    """
    cfg = ctx.cfg
    if cfg.best_include_adaln_override is not None:
        print(f"cfg.best_include_adaln_override set — using include_adaln={cfg.best_include_adaln_override} "
              f"without re-running this ablation.")
        return cfg.best_include_adaln_override
    if cfg.skip_ablations:
        print("cfg.skip_ablations=True and no override given — defaulting to include_adaln=True. "
              "Set cfg.best_include_adaln_override explicitly if you already know which one won.")
        return True

    target_results = []
    for include_adaln in [False, True]:
        label = "attn_adaln" if include_adaln else "attn_only"
        print(f"=== Target-module ablation: {label} ===")
        attach_lora(ctx.model, ctx.original_state, best_rank, cfg, include_adaln=include_adaln)
        target_info = summarize_lora_targets(ctx.model.transformer)
        if include_adaln and target_info["n_adaln"] == 0:
            print("  WARNING: 0 AdaLN modules matched even though include_adaln=True — "
                  "this checkpoint's module names may differ from what's assumed. "
                  "Do not trust the 'attn_adaln' results below if you see this.")

        df_t = _train_and_score(ctx, make_loader(ctx.train_df, cfg, dataset=ctx.train_dataset))
        df_t["config"] = label
        target_results.append(df_t)
        reset_transformer(ctx.model, ctx.original_state)

    target_ablation_df = pd.concat(target_results, ignore_index=True)
    target_summary = target_ablation_df.groupby("config")[["completion_ratio", "f0_mean", "f0_std", "rms"]].mean().round(3)
    print(target_summary)
    target_ablation_df.to_csv(f"{cfg.output_dir}/ablation_lora_targets.csv", index=False)

    print("\n--- Bootstrap 95% CI on f0_std (lower = more stable pitch), pooled across emotions ---")
    for label in ["attn_only", "attn_adaln"]:
        vals = target_ablation_df[target_ablation_df["config"] == label]["f0_std"].dropna().values
        mean, lo, hi = bootstrap_ci(vals, n_boot=cfg.bootstrap_n)
        print(f"{label:12s}: mean={mean:.1f}  95% CI=[{lo:.1f}, {hi:.1f}]  n={len(vals)}")

    if "attn_adaln" in target_summary.index and "attn_only" in target_summary.index:
        best = bool(target_summary.loc["attn_adaln", "f0_std"] < target_summary.loc["attn_only", "f0_std"])
    else:
        best = True
    print(f"\nSelected include_adaln={best} for the main run (chosen by lower mean f0_std).")
    return best
