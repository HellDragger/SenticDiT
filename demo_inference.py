import os
import torch
import argparse
from pathlib import Path
from accelerate import Accelerator

from src.config import CFG, EMOTION_TOK
from src.models import TextEncoder, AudioDiTWrapper, apply_lora, build_scheduler
from src.inference import generate_emotional_speech

def parse_args():
    parser = argparse.ArgumentParser(description="Generate emotional speech using fine-tuned AudioDiT.")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize.")
    parser.add_argument(
        "--emotion", type=str, default="neutral", 
        choices=list(EMOTION_TOK.keys()), 
        help="Target emotion for the speech."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, 
        help="Path to the trained LoRA checkpoint directory (e.g., ./audiodit_meld_output/lora_weights). If None, uses base model."
    )
    parser.add_argument("--output", type=str, default="demo_output.wav", help="Output WAV file path.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup
    accelerator = Accelerator(mixed_precision="fp16" if CFG.use_fp16 else "no")
    print(f"Device: {accelerator.device}")
    
    # 2. Load Models
    print("Loading text encoder and AudioDiT backbone...")
    text_encoder = TextEncoder().to(accelerator.device)
    audiodit     = apply_lora(AudioDiTWrapper())
    scheduler    = build_scheduler()
    
    # 3. Load Weights (if provided)
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading LoRA weights from {args.checkpoint}...")
        unwrapped = accelerator.unwrap_model(audiodit)
        if hasattr(unwrapped.unet, "load_adapter"):
            unwrapped.unet.load_adapter(args.checkpoint, adapter_name="default")
        else:
            lora_sd = torch.load(Path(args.checkpoint) / "lora_state_dict.pt", map_location="cpu")
            unwrapped.unet.load_state_dict(lora_sd, strict=False)
        print("✅ Weights loaded.")
    else:
        print("⚠ No checkpoint provided or found. Running with un-tuned LoRA base weights.")

    audiodit = audiodit.to(accelerator.device)
    
    # 4. Generate
    print(f"\nGenerating Audio...")
    print(f"Text:    '{args.text}'")
    print(f"Emotion: [{args.emotion.upper()}]")
    
    generate_emotional_speech(
        text=args.text,
        emotion_label=args.emotion,
        audiodit_model=accelerator.unwrap_model(audiodit),
        text_enc=text_encoder,
        sched=scheduler,
        save_path=args.output
    )
    print("✅ Done.")

if __name__ == "__main__":
    main()