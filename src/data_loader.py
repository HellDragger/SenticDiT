import torch
import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset
from typing import Dict, List
from src.config import CFG, EMOTION_TOK

def build_emotion_prompt(text: str, emotion: str) -> str:
    token = EMOTION_TOK.get(emotion.lower(), "[NEUTRAL]")
    return f"{token} {text.strip()}"

def wav_to_mel(
    wav_path: str,
    sr: int = CFG.sample_rate,
    n_mels: int = CFG.n_mels,
    n_fft: int = CFG.n_fft,
    hop: int = CFG.hop_length,
    max_sec: float = CFG.max_audio_sec,
) -> np.ndarray:
    y, file_sr = librosa.load(wav_path, sr=sr, mono=True)
    max_samples = int(max_sec * sr)
    if len(y) > max_samples:
        y = y[:max_samples]
    else:
        y = np.pad(y, (0, max_samples - len(y)))
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel.astype(np.float32)

class MELDDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row    = self.df.iloc[idx]
        mel    = wav_to_mel(row["wav_path"])
        mel_t  = torch.from_numpy(mel).unsqueeze(0)
        prompt = build_emotion_prompt(
            row.get("utterance", ""),
            row.get("emotion", "neutral")
        )
        return {
            "mel"       : mel_t,
            "prompt"    : prompt,
            "emotion_id": int(row["emotion_id"]),
            "wav_path"  : row["wav_path"],
        }

def collate_fn(batch: List[Dict]) -> Dict:
    mels   = torch.stack([b["mel"] for b in batch])
    eids   = torch.tensor([b["emotion_id"] for b in batch])
    prompts   = [b["prompt"] for b in batch]
    wav_paths = [b["wav_path"] for b in batch]
    return {"mel": mels, "prompt": prompts, "emotion_id": eids, "wav_path": wav_paths}