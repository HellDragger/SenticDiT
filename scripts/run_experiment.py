"""Run one SenticDiT experiment (v5, configd, addon, crossclone) in this process.

    python scripts/run_experiment.py v5 --config config_v5.json
    python scripts/run_experiment.py crossclone --set cremad_dir=data/AudioWAV --set output_dir=outputs

--config is a JSON object of `senticdit.config.Config` fields; --set overrides one field (the value
is parsed as JSON when possible, e.g. --set skip_ablations=true, otherwise taken as a string).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from senticdit.config import Config  # noqa: E402
from senticdit.experiments import EXPERIMENTS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=EXPERIMENTS)
    p.add_argument("--config", help="JSON file of Config fields")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one Config field")
    p.add_argument("--install", action="store_true",
                   help="pip-install dependencies first (the Kaggle notebook does this itself)")
    return p.parse_args()


def load_settings(args):
    settings = json.loads(Path(args.config).read_text()) if args.config else {}
    for kv in args.set:
        key, value = kv.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        settings[key] = value
    return settings


def main():
    args = parse_args()
    cfg = Config(**load_settings(args))

    from senticdit.setup_env import fetch_longcat, prepare_environment
    if args.install:
        prepare_environment(longcat_dir=cfg.longcat_repo_dir, longcat_url=cfg.longcat_repo_url)
    else:
        fetch_longcat(cfg.longcat_repo_dir, cfg.longcat_repo_url)

    from senticdit.experiments import get
    get(args.experiment).run(cfg)


if __name__ == "__main__":
    main()
