"""Generate emotional speech with LongCat-AudioDiT-1B, optionally with a trained LoRA adapter.

    python scripts/demo_inference.py --text "I cannot believe this is happening right now!" \
        --emotion surprise --adapter /kaggle/working/audiodit_fixed/lora_final --output surprise.wav

Duration comes from the calibrated text->duration model (Config.dur_slope / dur_intercept).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from senticdit.config import Config  # noqa: E402


def parse_args():
    emotions = Config().emotions
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", required=True, help="Text to synthesize.")
    p.add_argument("--emotion", default="neutral", choices=emotions)
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter directory (e.g. <output_dir>/lora_final). Omit for the base model.")
    p.add_argument("--output", default="demo_output.wav")
    p.add_argument("--cfg-strength", type=float, default=Config().cfg_strength)
    p.add_argument("--steps", type=int, default=Config().cfm_steps)
    p.add_argument("--longcat-dir", default=Config().longcat_repo_dir)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(longcat_repo_dir=args.longcat_dir)

    from senticdit.setup_env import fetch_longcat
    fetch_longcat(cfg.longcat_repo_dir, cfg.longcat_repo_url)

    from peft import PeftModel
    from senticdit.generation import generate_speech
    from senticdit.model import load_pretrained
    from senticdit.utils import get_device

    device = get_device()
    model, tokenizer, _ = load_pretrained(cfg, device, snapshot=False)
    if args.adapter:
        if not Path(args.adapter).exists():
            raise FileNotFoundError(f"Adapter not found: {args.adapter}")
        model.transformer = PeftModel.from_pretrained(model.transformer, args.adapter).to(device)
        print(f"Loaded LoRA adapter from {args.adapter}")
    else:
        print("No adapter given — generating with the pretrained base model.")

    generate_speech(model, tokenizer, args.text, cfg, device, emotion=args.emotion,
                    out_path=args.output, steps=args.steps, cfg_strength=args.cfg_strength)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
