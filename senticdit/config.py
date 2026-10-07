"""Run configuration for SenticDiT v5.

Every knob the Kaggle notebook can set lives here. The notebook constructs a `Config`, overrides
what it needs, and hands it to `senticdit.pipeline.SenticDiTRun`. Defaults reproduce the v5
notebook exactly.
"""
from dataclasses import dataclass, field
from typing import List, Optional


EMOTION_ALIAS = {
    "neutral": "neutral", "surprise": "surprise", "surprised": "surprise",
    "fear": "fear", "fearful": "fear", "sadness": "sadness", "sad": "sadness",
    "joy": "joy", "happy": "joy", "happiness": "joy", "disgust": "disgust",
    "anger": "anger", "angry": "anger",
}


@dataclass
class Config:
    # --- MELD paths -------------------------------------------------------------
    kaggle_input_dir: str = "/kaggle/input/datasets/zaber666/meld-dataset/MELD-RAW/MELD.Raw"
    audio_dir:        str = "/kaggle/input/datasets/aryansharma26/meld-audio/MELD_audio"
    output_dir:       str = "/kaggle/working/audiodit_fixed"

    # --- Model -------------------------------------------------------------------
    model_id: str = "meituan-longcat/LongCat-AudioDiT-1B"
    longcat_repo_url: str = "https://github.com/meituan-longcat/LongCat-AudioDiT.git"
    longcat_repo_dir: str = "/kaggle/working/LongCat-AudioDiT"   # provides the `audiodit` package
    sample_rate: int = 24000    # overwritten from real model config after load
    latent_hop:  int = 2048     # overwritten from real model config after load
    max_audio_sec: float = 6.0 # per-clip cap (batches pad only to *their own* real max)

    # --- Calibrated text->duration model ---------------------------------------------
    # seconds = dur_slope * num_chars + dur_intercept. Refit on the training data by
    # `data.fit_duration_model`; these defaults are the fit from earlier runs, used only when
    # generating without MELD available (e.g. demo_inference.py).
    dur_slope: float = 0.0534
    dur_intercept: float = 1.014

    # --- LoRA ----------------------------------------------------------------------
    lora_r: int = 32                # fallback default; the rank ablation below may override this
    lora_alpha_mult: int = 2        # lora_alpha is always set to lora_r * lora_alpha_mult
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v"])
    lora_target_modules_with_adaln: List[str] = field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "mlp.1"])
    # "mlp.1" targets the Linear inside AudioDiTAdaLNMLP — either every block's own
    # `adaln_mlp.mlp.1` ("local" AdaLN mode) or the single shared `adaln_global_mlp.mlp.1`
    # ("global" mode). Verified there's only one `self.mlp = nn.Sequential(...)` anywhere in
    # the DiT source (the FFN uses `self.ff`, not `self.mlp`), so this suffix is unambiguous
    # for this specific architecture. The target ablation prints how many modules it actually
    # matched in the loaded checkpoint — trust that printout, not this comment.
    lora_rank_sweep: List[int] = field(default_factory=lambda: [16, 32, 64])
    rank_tie_tolerance: float = 0.01   # ranks within this much completion_ratio of the best are
                                        # treated as tied; the tie is broken by lowest f0_std instead
                                        # of arbitrary index order

    # --- Training --------------------------------------------------------------------
    batch_size: int = 4
    grad_accum_steps: int = 4
    # 20, not 25-50: every loss curve seen so far drops fast in the first few hundred steps then
    # goes flat/noisy for thousands more. More real, distinct data is a better bet for using extra
    # epochs productively than raw repetition.
    num_epochs: int = 20
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    checkpoint_steps: int = 200     # full resumable state saved this often during the main run
    num_workers: int = 2
    seed: int = 42
    debug_subset: Optional[int] = None   # e.g. 200 for a fast smoke test of the WHOLE pipeline

    # --- Multi-session resume --------------------------------------------------------------
    # Point this at a *previous session's* checkpoint dir (added as a Kaggle "Input"), e.g.
    # "/kaggle/input/senticdit-checkpoint-v1/checkpoints/r16_attn_only", to continue the main run
    # where it left off. Leave as None to start fresh.
    resume_search_dir: Optional[str] = None

    # --- Ablation (sampling strategy + LoRA rank + target modules) ----------------------------
    ablation_max_steps: int = 500
    eval_samples_per_emotion: Optional[int] = None   # None = use the whole held-out set
    bootstrap_n: int = 3000
    eval_holdout_per_emotion: int = 30   # carved out of the COMBINED train+dev+test pool, never
                                          # used in training — all evaluation runs on this

    # --- Re-run efficiency ------------------------------------------------------------------
    # Set skip_ablations=True and the overrides to the values you already found, and the
    # ablation stages become near-instant no-ops.
    skip_ablations: bool = False
    best_lora_rank_override: Optional[int] = None
    best_include_adaln_override: Optional[bool] = None

    # --- Generation / CFG tuning -----------------------------------------------------------
    cfm_steps: int = 16
    cfg_strength: float = 2.0             # fixed by fiat; the sweep is reported as a null
    cfg_sweep_values: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 4.0])
    cfg_sweep_per_emotion: int = 10
    f0_std_sanity_bound: float = 150.0    # natural speech rarely exceeds this within one sentence
    baseline_text: str = "I cannot believe this is actually working now."
    demo_text: str = "I never thought this would happen to me."

    # --- F0 extraction bounds ------------------------------------------------------------------
    # Constrained to the adult speaking range, median preferred over mean, cross-checked against
    # WORLD's harvest (resistant to octave doubling).
    f0_floor_hz: float = 65.0
    f0_ceiling_hz: float = 400.0
    f0_octave_check: bool = True
    # Fine-tuning moved median F0 from 143 Hz to ~249 Hz; a wider diagnostic pass measures how
    # much the 400 Hz primary ceiling is clipping.
    f0_diagnostic_ceiling_hz: float = 600.0

    # --- MCD -------------------------------------------------------------------------------------
    # "plain" zero-pads and measures duration mismatch; "dtw_sl" multiplies aligned MCD by the
    # length ratio (fidelity x duration penalty). Only plain DTW is a fidelity metric.
    mcd_mode: str = "dtw"
    mcd_diagnostic_modes: tuple = ("plain", "dtw_sl")   # computed for comparison, never reported
                                                        # as fidelity
    mcd_refs_per_emotion: Optional[int] = None      # None = every held-out reference

    # --- Reference quality filter ---------------------------------------------------------------
    # MELD contains truncated audio extractions (e.g. a 52-character sentence with a 0.13 s
    # reference). These are corrupt references, not generation failures.
    mcd_min_ref_dur: float = 1.0
    mcd_max_dur_ratio: float = 2.0        # symmetric; also excludes ratios below 0.5
    mcd_report_unfiltered_too: bool = True

    # --- UTMOS anchor ----------------------------------------------------------------------------
    # Stratified across emotions and length-filtered; reported with caveats, not used as a floor.
    utmos_anchor_per_emotion: int = 10
    utmos_anchor_min_dur: float = 2.0

    # --- Reproducibility -------------------------------------------------------------------------
    # Each evaluation block reseeds from eval_seed, so arms are independent of what ran before
    # them. Replicate by bumping eval_seed and run_tag; results append to run_log.csv.
    eval_seed: int = 1234
    run_tag: str = "v4_run1"
    force_regenerate_zeroshot: bool = True

    # --- Completion metric ------------------------------------------------------------------------
    completion_fail_threshold: float = 0.90

    # --- Controls ----------------------------------------------------------------------------------
    run_zero_shot_baseline: bool = True    # untouched pretrained model on the same held-out set
    run_ser_real_audio_check: bool = True  # run the SER wrapper over real MELD recordings
    run_utmos: bool = True                 # reference-free, speaker-independent MOS
    n_boot_diff: int = 10000               # resamples for CIs on differences

    # --- v5: additional evaluation models ---------------------------------------------------------
    run_wer: bool = True
    asr_model_id: str = "openai/whisper-large-v3-turbo"
    run_emotion2vec: bool = True
    emotion2vec_model_id: str = "emotion2vec/emotion2vec_plus_large"
    run_speaker_consistency: bool = True
    speaker_model_id: str = "microsoft/wavlm-base-plus-sv"
    verify_weight_loading: bool = True

    # --- v5: Config D — isolating data quantity ------------------------------------------------
    # Both arms share rank, targets, class-balanced sampling, step budget, LR and evaluation noise
    # draws; ONLY the training pool differs (MELD train split vs pooled train+dev+test).
    run_config_d: bool = True
    config_d_max_steps: int = 2000         # ~3.3 epochs of train-only, ~2.4 of pooled
    config_d_rank: int = 16
    config_d_include_adaln: bool = False
    config_d_ckpt_every: int = 500

    # --- Evaluation ------------------------------------------------------------------------
    ser_model_id: str = "r-f/wav2vec-english-speech-emotion-recognition"

    emotions: List[str] = field(default_factory=lambda: [
        "neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"
    ])

    @property
    def emotion2id(self):
        return {e: i for i, e in enumerate(self.emotions)}
