import torch
import os
import pandas as pd
from pathlib import Path
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

from src.config import CFG, DEVICE
from src.models import TextEncoder, AudioDiTWrapper, apply_lora, build_scheduler
from src.data_loader import MELDDataset, collate_fn
from src.trainer import train
from src.inference import generate_emotional_speech

def main():
    # 1. Setup Accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if CFG.use_fp16 else "no",
        gradient_accumulation_steps=CFG.grad_accum_steps,
    )
    print(f"Accelerator Device: {accelerator.device}")

    # 2. Build Models
    print("Loading models...")
    text_encoder = TextEncoder().to(accelerator.device)
    audiodit     = apply_lora(AudioDiTWrapper())
    scheduler    = build_scheduler()

    # 3. Setup Mock Data (Replace with actual CSV/MELD Logic)
    # We create a dummy dataframe just to show the pipeline executes
    Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)
    dummy_df = pd.DataFrame({
        "wav_path": [], # Populate with actual paths
        "utterance": ["Test"],
        "emotion": ["joy"],
        "emotion_id": [CFG.emotions.index("joy")]
    })
    
    train_dataset = MELDDataset(dummy_df)
    train_loader  = DataLoader(
        train_dataset, batch_size=CFG.batch_size, collate_fn=collate_fn
    )

    # 4. Optimizers
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, audiodit.parameters()),
        lr=CFG.learning_rate, weight_decay=1e-3,
    )
    
    total_steps  = max(1, len(train_loader) * CFG.num_epochs // CFG.grad_accum_steps)
    warmup_steps = int(total_steps * CFG.warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    audiodit, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        audiodit, optimizer, train_loader, lr_scheduler
    )

    # 5. Train
    if len(dummy_df) > 0:
        print("Starting training...")
        train(
            audiodit, text_encoder, scheduler, train_loader, 
            optimizer, lr_scheduler, accelerator
        )
    else:
        print("Skipping training loop (add data to dummy_df).")

    # 6. Test Inference
    print("Testing Inference Generation...")
    test_wav = generate_emotional_speech(
        text="I am so happy!",
        emotion_label="joy",
        audiodit_model=accelerator.unwrap_model(audiodit),
        text_enc=text_encoder,
        sched=scheduler,
        save_path=f"{CFG.output_dir}/sample_joy.wav"
    )
    print("Pipeline Complete.")

if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    main()