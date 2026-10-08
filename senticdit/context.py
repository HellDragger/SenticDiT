"""State shared by every experiment: config, device, MELD splits, and the loaded model.

Each experiment runs in its own process (see `senticdit.launcher`), exactly like a separate Kaggle
session of the original notebooks, so GPU memory never carries over from one to the next.
"""
import logging
import warnings
from pathlib import Path

import torch

from .config import Config
from .data import EmotionalSpeechDataset, add_class_weights, fit_duration_model, load_meld, \
    split_train_holdout
from .model import load_pretrained, verify_weight_loading
from .utils import get_device, section, set_global_seed


class RunContext:
    def __init__(self, cfg: Config):
        warnings.filterwarnings("ignore")
        logging.basicConfig(level=logging.WARNING)

        self.cfg = cfg
        self.device = get_device()
        print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
        print("GPU count:", torch.cuda.device_count())
        set_global_seed(cfg.seed)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        print("Output dir:", cfg.output_dir)

        self.train_df = self.holdout_df = self.train_dataset = None
        self.model = self.tokenizer = self.original_state = None

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

    def load_model(self, snapshot=True):
        """Load AudioDiT-1B, run the weight-loading guard, and (with MELD loaded) build the
        training dataset. snapshot=False skips caching the clean transformer state, which only
        experiments that train and reset LoRA adapters need."""
        section("3 · Load the real pretrained AudioDiT-1B")
        self.model, self.tokenizer, self.original_state = load_pretrained(self.cfg, self.device,
                                                                          snapshot=snapshot)
        if self.cfg.verify_weight_loading:
            section("3.1 · Weight-loading guard")
            verify_weight_loading(self.model, self.cfg)
        if self.train_df is not None:
            self.train_dataset = EmotionalSpeechDataset(self.train_df, self.cfg)
            print(f"{len(self.train_dataset)} training clips available")

    def free_model(self):
        """Release the TTS model so every scorer has the GPU to itself."""
        import gc
        self.model = None
        self.original_state = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
