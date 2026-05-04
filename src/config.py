import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────────────────────
    kaggle_input_dir : str = "/kaggle/input/datasets/zaber666/meld-dataset/MELD-RAW/MELD.Raw"
    output_dir       : str = "./audiodit_meld_output"
    audio_cache_dir  : str = "./audio_cache"  
    
    # ── Model ─────────────────────────────────────────────────────────────────
    model_id         : str  = "meituan-longcat/LongCat-AudioDiT-1B"
    text_encoder_id  : str  = "google/flan-t5-small"
    sample_rate      : int  = 24_000
    hop_length       : int  = 256
    n_fft            : int  = 1024
    n_mels           : int  = 80
    max_audio_sec    : float = 10.0

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_r           : int   = 16
    lora_alpha       : int   = 32
    lora_dropout     : float = 0.05
    lora_target_modules : List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    )

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size           : int   = 4
    grad_accum_steps     : int   = 4
    num_epochs           : int   = 10
    learning_rate        : float = 5e-4
    warmup_ratio         : float = 0.03
    max_grad_norm        : float = 1.0
    num_workers          : int   = 2
    seed                 : int   = 42
    use_fp16             : bool  = True
    grad_checkpointing   : bool  = True
    checkpoint_steps     : int   = 100
    resume_from_checkpoint : str = "" 

    # ── Diffusion ─────────────────────────────────────────────────────────────
    num_train_timesteps : int   = 1000
    num_inference_steps : int   = 50
    guidance_scale      : float = 3.5

    # ── Emotions ──────────────────────────────────────────────────────────────
    emotions : List[str] = field(default_factory=lambda: [
        "neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"
    ])

# Global Instantiation
CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION2ID  = {e: i for i, e in enumerate(CFG.emotions)}
ID2EMOTION  = {i: e for e, i in EMOTION2ID.items()}
EMOTION_TOK = {e: f"[{e.upper()}]" for e in CFG.emotions}