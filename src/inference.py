import torch
import numpy as np
import librosa
import soundfile as sf
from diffusers import DDIMScheduler
from typing import Optional
from src.config import CFG
from src.data_loader import build_emotion_prompt

def mel_to_wav_griffinlim(
    log_mel: np.ndarray, sr: int = CFG.sample_rate, n_fft: int = CFG.n_fft,
    hop: int = CFG.hop_length, n_iter: int = 64,
) -> np.ndarray:
    log_mel = np.clip(log_mel, -1.0, 1.0)
    mel_db  = log_mel * 80.0 - 80.0
    mel_pow = librosa.db_to_power(mel_db)
    mel_pow = np.clip(mel_pow, 1e-10, None) 
    stft    = librosa.feature.inverse.mel_to_stft(mel_pow, sr=sr, n_fft=n_fft, power=2.0)
    wav     = librosa.griffinlim(stft, n_iter=n_iter, hop_length=hop)
    wav     = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    peak = np.abs(wav).max()
    if peak > 1e-6: wav = wav / peak * 0.95
    return wav.astype(np.float32)

@torch.no_grad()
def generate_emotional_speech(
    text: str, emotion_label: str, audiodit_model, text_enc, sched,
    num_steps: int = CFG.num_inference_steps,
    guidance_scale: float = CFG.guidance_scale,
    save_path: Optional[str] = None,
) -> np.ndarray:
    audiodit_model.eval()
    device = next(audiodit_model.parameters()).device

    prompt      = build_emotion_prompt(text, emotion_label)
    null_prompt = build_emotion_prompt("", "neutral")
    cond_emb    = text_enc([prompt], device)
    uncond_emb  = text_enc([null_prompt], device)

    T         = int(CFG.max_audio_sec * CFG.sample_rate / CFG.hop_length)
    mel_shape = (1, 1, CFG.n_mels, T)
    latent    = torch.randn(mel_shape, device=device, dtype=cond_emb.dtype)

    inference_sched = DDIMScheduler.from_config(sched.config)
    inference_sched.set_timesteps(num_steps)

    for t in inference_sched.timesteps:
        t_batch   = t.unsqueeze(0).to(device)
        latent_in = inference_sched.scale_model_input(latent, t)
        noise_cond   = audiodit_model(latent_in, t_batch, cond_emb)
        noise_uncond = audiodit_model(latent_in, t_batch, uncond_emb)
        noise_pred   = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latent = inference_sched.step(noise_pred, t, latent).prev_sample

    generated_mel = latent.squeeze().cpu().float().numpy()
    generated_mel = np.clip(generated_mel, -3.0, 3.0) 
    wav = mel_to_wav_griffinlim(generated_mel)

    if save_path:
        sf.write(save_path, wav, CFG.sample_rate)
        print(f"Saved -> {save_path}")

    return wav