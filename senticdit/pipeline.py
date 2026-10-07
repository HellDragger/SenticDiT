"""End-to-end SenticDiT v5 run.

`SenticDiTRun` holds the state shared between stages (model, data splits, evaluation frames) and
exposes each stage as a method, in the order the paper's results depend on. `run_all()` runs them
all; individual stages can be called one by one from a notebook when debugging.

Expected runtime on 2x T4 (rates from the v3 logs): setup + loading guard ~8 min, zero-shot
~11 min, main run ~0 if the resume checkpoint is complete (~16 h from scratch), CFG sweep ~16 min,
final eval + SER + paraphrase ~31 min, MCD ~25 min, UTMOS/mechanism/WER/emotion2vec/speaker
~22 min, Config D ~4.3 h. With Config D: ~6.5-7.5 h, which fits one 12 h Kaggle session.
Re-running the ablations (skip_ablations=False) adds ~5 h — use a second session.
"""
import warnings
import logging
from pathlib import Path

import torch

from . import ablations, report
from .config import Config
from .config_d import run_config_d
from .data import EmotionalSpeechDataset, add_class_weights, fit_duration_model, load_meld, \
    make_loader, split_train_holdout
from .evaluation import health, mcd, ser, speaker, utmos, wer
from .generation import generate_speech
from .model import attach_lora, load_pretrained, summarize_lora_targets, verify_weight_loading
from .training import plot_loss_curve, train_lora
from .utils import get_device, section, set_global_seed, show_audio


class SenticDiTRun:
    def __init__(self, cfg: Config):
        warnings.filterwarnings("ignore")
        logging.basicConfig(level=logging.WARNING)

        self.cfg = cfg
        self.device = get_device()
        set_global_seed(cfg.seed)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
        print("GPU count:", torch.cuda.device_count())
        print("Output dir:", cfg.output_dir)

        # data
        self.train_df = self.holdout_df = self.train_dataset = None
        # model
        self.model = self.tokenizer = self.original_state = None
        self.baseline_wav = None
        # selections
        self.best_rank = self.best_include_adaln = None
        self.best_cfg_strength = float(cfg.cfg_strength)
        self.losses = []
        # evaluation state
        self.zeroshot_df = self.final_eval_df = None
        self.ser, self.ser_stats, self.ser_accuracy = None, {}, None
        self.utmos = None
        self.asr, self.wer_results = None, {}
        self.e2v_stats, self.spk_results = {}, {}
        self.mcd_df = self.mcd_kept = None
        self.configd_cmp = None
        self.paper_df = None

    # ------------------------------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------------------------------

    def prepare_data(self):
        section("2 · MELD dataset — pooled train+dev+test with a fresh held-out carve-out")
        processed = load_meld(self.cfg)
        train_df, self.holdout_df = split_train_holdout(processed, self.cfg)

        section("2.2 · Class-balanced sample weights + calibrated text->duration model")
        self.train_df = add_class_weights(train_df)
        print("Class counts:\n", self.train_df["emotion"].value_counts())
        print("\nSample weight range:", self.train_df["sample_weight"].min(), "-",
              self.train_df["sample_weight"].max())
        fit_duration_model(self.train_df, self.cfg)

    def load_model(self):
        section("3 · Load the real pretrained AudioDiT-1B")
        self.model, self.tokenizer, self.original_state = load_pretrained(self.cfg, self.device)
        if self.cfg.verify_weight_loading:
            section("3.1 · Weight-loading guard")
            verify_weight_loading(self.model, self.cfg)
        self.train_dataset = EmotionalSpeechDataset(self.train_df, self.cfg)
        print(f"{len(self.train_dataset)} training clips available")

    def baseline_sample(self):
        section("4 · Sanity check — baseline generation before any fine-tuning")
        self.baseline_wav = generate_speech(
            self.model, self.tokenizer, self.cfg.baseline_text, self.cfg, self.device,
            out_path=f"{self.cfg.output_dir}/baseline_no_finetune.wav")
        print("Baseline (pretrained, no LoRA) sample:")
        show_audio(self.baseline_wav, self.cfg.sample_rate)

    # ------------------------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------------------------

    def zero_shot(self):
        section("7.1 · Zero-shot pretrained baseline")
        self.zeroshot_df = health.zero_shot_baseline(self)

    def run_ablations(self):
        section("8 · Ablation: naive vs class-balanced sampling")
        ablations.sampling_ablation(self)
        section("8.1 · Ablation: LoRA rank")
        self.best_rank = ablations.rank_ablation(self)
        section("8.2 · Ablation: LoRA target modules (attention-only vs + AdaLN)")
        self.best_include_adaln = ablations.target_ablation(self, self.best_rank)

    def train_main(self):
        """Class-balanced sampling + selected rank + selected targets, resumable. Checkpoints
        are namespaced by rank + target config so a resume can never continue an incompatible run."""
        section("9 · Main fine-tuning run")
        cfg = self.cfg
        attach_lora(self.model, self.original_state, self.best_rank, cfg,
                    include_adaln=self.best_include_adaln)
        self.model.transformer.print_trainable_parameters()
        summarize_lora_targets(self.model.transformer)
        print(f"Main run config -> rank={self.best_rank}, include_adaln={self.best_include_adaln}, "
              f"sampling=class_balanced (WeightedRandomSampler), epochs={cfg.num_epochs}")

        train_loader = make_loader(self.train_df, cfg, dataset=self.train_dataset)
        steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
        main_max_steps = steps_per_epoch * cfg.num_epochs
        print(f"Main run: {steps_per_epoch} steps/epoch x {cfg.num_epochs} epochs = {main_max_steps} steps")
        if cfg.resume_search_dir:
            print(f"Resume search dir configured: {cfg.resume_search_dir}")

        config_tag = f"r{self.best_rank}_{'attn_adaln' if self.best_include_adaln else 'attn_only'}"
        ckpt_dir = Path(cfg.output_dir) / "checkpoints" / config_tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint dir for this configuration: {ckpt_dir}")

        self.losses = train_lora(self.model, self.tokenizer, train_loader, cfg, self.device,
                                 max_steps=main_max_steps, lr=cfg.learning_rate, ckpt_dir=ckpt_dir,
                                 ckpt_every=cfg.checkpoint_steps,
                                 resume_search_dir=cfg.resume_search_dir,
                                 base_transformer_state=self.original_state)

        final_path = Path(cfg.output_dir) / "lora_final"
        self.model.transformer.save_pretrained(str(final_path))
        print("Saved final LoRA adapter ->", final_path)

        section("10 · Loss curve")
        plot_loss_curve(self.losses, f"{cfg.output_dir}/loss_curve.png")

    # ------------------------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------------------------

    def cfg_sweep(self):
        section("11 · CFG-strength sweep")
        self.best_cfg_strength = health.cfg_sweep(self)

    def final_eval(self):
        section("12 · Full per-emotion evaluation")
        self.final_eval_df = health.final_evaluation(self)

    def ser_eval(self):
        section("13 · Automatic emotion recognition (+ real-audio control)")
        self.ser, self.ser_stats, self.ser_accuracy = ser.ser_evaluation(self)

    def paraphrase(self):
        section("14 · Conditioning-generalisation test")
        health.paraphrase_generalization(self)

    def demos(self):
        section("15 · Final per-emotion samples")
        health.demo_samples(self)

    def mcd_eval(self):
        section("16 · Mel-Cepstral Distortion — dtw, filtered")
        self.mcd_df, self.mcd_kept = mcd.mcd_evaluation(self)
        section("16.1 · Quantifying the alignment artefact on identical audio")
        mcd.alignment_artefact(self, self.mcd_df)

    def utmos_eval(self):
        section("16.2 · UTMOS — reference-free quality")
        self.utmos = utmos.utmos_evaluation(self)
        section("16.3 · Does the pitch drift explain the quality loss?")
        utmos.pitch_drift_mechanism(self)

    def wer_eval(self):
        section("16.4 · Intelligibility — Whisper WER")
        self.asr, self.wer_results = wer.wer_evaluation(self)

    def emotion2vec_eval(self):
        section("16.5 · A second SER model — emotion2vec")
        self.e2v_stats = ser.emotion2vec_evaluation(self)

    def speaker_eval(self):
        section("16.6 · Speaker self-consistency")
        self.spk_results = speaker.speaker_consistency(self)

    def config_d(self):
        section("16.7 · Config D — does more distinct data help, at fixed compute?")
        self.configd_cmp = run_config_d(self)

    def write_report(self):
        section("17 · Assembled results for the paper")
        self.paper_df = report.paper_results_table(self)
        section("18 · Output manifest")
        manifest = report.write_manifest(self.cfg.output_dir)
        print(manifest.to_string(index=False))
        return self.paper_df

    # ------------------------------------------------------------------------------------------

    def run_all(self):
        self.prepare_data()
        self.load_model()
        self.baseline_sample()
        self.zero_shot()
        self.run_ablations()
        self.train_main()
        self.cfg_sweep()
        self.final_eval()
        self.ser_eval()
        self.paraphrase()
        self.demos()
        self.mcd_eval()
        self.utmos_eval()
        self.wer_eval()
        self.emotion2vec_eval()
        self.speaker_eval()
        self.config_d()
        return self.write_report()


def run(cfg: Config):
    """Convenience wrapper: build a run, execute every stage, return it."""
    r = SenticDiTRun(cfg)
    r.run_all()
    return r
