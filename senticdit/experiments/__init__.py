"""The four Kaggle experiments behind the paper, one module each. Every module exposes
`run(cfg)`; `scripts/run_experiment.py` runs one per process.

| name        | original notebook            | what it produces |
|-------------|------------------------------|------------------|
| v5          | senticdit_v5.ipynb           | main run, ablations, full held-out evaluation, paper table |
| configd     | senticdit_configd.ipynb      | data quantity at fixed compute + MELD register check |
| addon       | senticdit_addon.ipynb        | CREMA-D cloned-voice MCD + second-opinion metrics on MELD |
| crossclone  | senticdit_crossclone.ipynb   | 2x2 cross-speaker cloning: speaker vs channel in MCD |
"""
import importlib

EXPERIMENTS = ("v5", "configd", "addon", "crossclone")


def get(name):
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {name!r}; choose from {EXPERIMENTS}")
    return importlib.import_module(f"{__name__}.{name}")
