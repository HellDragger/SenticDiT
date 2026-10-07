import random
import zlib

import numpy as np
import torch


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def reseed(cfg, tag: str, seed: int = None):
    """Reseed before each evaluation block.

    With one global seed, the noise a block draws depends on how much RNG every preceding stage
    consumed. That is why v3 run B -- which skipped the ablations -- reproduced run A's zero-shot
    arm exactly while its fine-tuned arm drifted. Reseeding per block decouples each arm from
    what ran before it. To replicate deliberately, change cfg.eval_seed and rerun.
    """
    s = (cfg.eval_seed if seed is None else seed) + zlib.crc32(tag.encode()) % 10000
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return s


def show_audio(wav, sr):
    """Inline player in a notebook; silently does nothing from a plain script."""
    try:
        from IPython import get_ipython
        from IPython.display import Audio, display
        if get_ipython() is not None:
            display(Audio(wav, rate=sr))
    except ImportError:
        pass


def section(title: str):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
