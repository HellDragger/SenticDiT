"""Run the full SenticDiT v5 pipeline outside a notebook.

    python scripts/run_pipeline.py --skip-ablations --rank 16 --no-adaln \
        --resume /kaggle/input/<prev-output>/audiodit_fixed/checkpoints/r16_attn_only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from senticdit.config import Config  # noqa: E402


def parse_args():
    d = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meld-dir", default=d.kaggle_input_dir)
    p.add_argument("--audio-dir", default=d.audio_dir)
    p.add_argument("--output-dir", default=d.output_dir)
    p.add_argument("--longcat-dir", default=d.longcat_repo_dir)
    p.add_argument("--resume", default=None, help="checkpoint dir of a previous main run")
    p.add_argument("--skip-ablations", action="store_true")
    p.add_argument("--rank", type=int, default=None, help="best_lora_rank_override")
    p.add_argument("--adaln", dest="adaln", action="store_true", default=None)
    p.add_argument("--no-adaln", dest="adaln", action="store_false")
    p.add_argument("--epochs", type=int, default=d.num_epochs)
    p.add_argument("--debug-subset", type=int, default=None)
    p.add_argument("--no-config-d", action="store_true")
    p.add_argument("--run-tag", default=d.run_tag)
    p.add_argument("--eval-seed", type=int, default=d.eval_seed)
    p.add_argument("--no-install", action="store_true", help="skip pip installs (deps already present)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        kaggle_input_dir=args.meld_dir, audio_dir=args.audio_dir, output_dir=args.output_dir,
        longcat_repo_dir=args.longcat_dir, resume_search_dir=args.resume,
        skip_ablations=args.skip_ablations, best_lora_rank_override=args.rank,
        best_include_adaln_override=args.adaln, num_epochs=args.epochs,
        debug_subset=args.debug_subset, run_config_d=not args.no_config_d,
        run_tag=args.run_tag, eval_seed=args.eval_seed,
    )

    from senticdit.setup_env import prepare_environment
    prepare_environment(install=not args.no_install, longcat_dir=cfg.longcat_repo_dir,
                        longcat_url=cfg.longcat_repo_url)

    from senticdit.pipeline import run
    run(cfg)


if __name__ == "__main__":
    main()
