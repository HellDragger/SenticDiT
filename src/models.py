import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, AutoModel
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model
from typing import List
from src.config import CFG, EMOTION_TOK

class TextEncoder(nn.Module):
    def __init__(self, model_id: str = CFG.text_encoder_id, max_length: int = 128):
        super().__init__()
        self.tokenizer  = T5Tokenizer.from_pretrained(model_id)
        self.encoder    = T5EncoderModel.from_pretrained(model_id)
        self.max_length = max_length
        num_added = self.tokenizer.add_tokens(list(EMOTION_TOK.values()))
        if num_added:
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, prompts: List[str], device) -> torch.Tensor:
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_length,
        ).to(device)
        return self.encoder(**enc).last_hidden_state

def build_scheduler(num_train_timesteps: int = CFG.num_train_timesteps) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps = num_train_timesteps,
        beta_schedule       = "scaled_linear",
        clip_sample         = False,
        prediction_type     = "epsilon",
    )

class AudioDiTWrapper(nn.Module):
    def __init__(self, model_id: str = CFG.model_id, use_fp16: bool = CFG.use_fp16):
        super().__init__()
        dtype  = torch.float16 if use_fp16 else torch.float32
        
        try:
            self.unet = AutoModel.from_pretrained(
                model_id, torch_dtype=dtype,
                trust_remote_code=True, low_cpu_mem_usage=True,
            )
        except Exception as e:
            print(f"Failed to load real model. Using diffusers fallback. Error: {e}")
            from diffusers import UNet2DConditionModel
            self.unet = UNet2DConditionModel(
                sample_size=64, in_channels=1, out_channels=1, layers_per_block=2,
                block_out_channels=(64, 128, 256, 256), cross_attention_dim=512,
                down_block_types=("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",),
                up_block_types=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
            )
            
        if CFG.grad_checkpointing and hasattr(self.unet, "enable_gradient_checkpointing"):
            self.unet.enable_gradient_checkpointing()

    def forward(self, noisy_mel, timesteps, encoder_hidden_states):
        out = self.unet(
            sample=noisy_mel, timestep=timesteps, encoder_hidden_states=encoder_hidden_states,
        )
        if hasattr(out, "sample"): return out.sample
        if isinstance(out, (tuple, list)): return out[0]
        return out

def apply_lora(model: AudioDiTWrapper) -> AudioDiTWrapper:
    # Basic target resolution (simplified from your auto-discover logic)
    lora_cfg = LoraConfig(
        r=CFG.lora_r, lora_alpha=CFG.lora_alpha, lora_dropout=CFG.lora_dropout,
        target_modules=CFG.lora_target_modules, bias="none",
    )
    model.unet = get_peft_model(model.unet, lora_cfg)
    return model