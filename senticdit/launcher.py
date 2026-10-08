"""Run experiments from a notebook, each in its own Python process.

A fresh process per experiment reproduces the original setup — each experiment was its own Kaggle
session — so no GPU memory, RNG state or loaded model carries over between them. Output streams
into the notebook cell and is also saved to <output_dir>/log_<experiment>.txt.

Imports only the standard library and the (torch-free) Config, so it is safe to use before the
heavy dependencies are installed.
"""
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .config import Config

REPO_DIR = Path(__file__).resolve().parents[1]


def run_experiment(name, settings, repo_dir=REPO_DIR):
    cfg = Config(**settings)                     # fail fast on a misspelt setting
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / f"config_{name}.json"
    cfg_path.write_text(json.dumps(asdict(cfg), indent=2))

    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    if name != "v5":                             # set by the configd / addon / crossclone notebooks
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [sys.executable, "-u", str(Path(repo_dir) / "scripts" / "run_experiment.py"), name,
           "--config", str(cfg_path)]
    print(f"{'#' * 78}\n# experiment: {name}\n# config: {cfg_path}\n{'#' * 78}", flush=True)
    with open(out / f"log_{name}.txt", "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, env=env)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"experiment {name!r} failed with exit code {rc}; see the log above")
