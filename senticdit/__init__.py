"""SenticDiT: emotion-conditioned LoRA fine-tuning of LongCat-AudioDiT-1B on MELD.

Kept import-light on purpose: `setup_env.prepare_environment()` has to run (and upgrade
transformers) before torch-dependent modules are imported. Import `senticdit.pipeline` after it.
"""
from .config import Config

__all__ = ["Config"]
