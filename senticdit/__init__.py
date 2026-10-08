"""SenticDiT: emotion-conditioned LoRA fine-tuning of LongCat-AudioDiT-1B on MELD, and the
experiments that evaluate it.

Kept import-light on purpose: `setup_env.prepare_environment()` has to run (and upgrade
transformers) before torch-dependent modules are imported, and `launcher` runs each experiment in
its own process.
"""
from .config import Config

__all__ = ["Config"]
