"""MELD loading, the pooled train / held-out split, sampling weights, duration model, and the
variable-length training dataset.

Why pool train+dev+test with a fresh held-out carve-out instead of merging everything: if `dev`
became training data, every held-out number would measure memorisation. Pooling still gives
MELD's rarest classes (fear/disgust) meaningfully more examples, and the stratified held-out set
(`cfg.eval_holdout_per_emotion` per emotion, never trained on) keeps evaluation valid. This
deviates from MELD's standard protocol; the paper says so explicitly.
"""
from functools import partial
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import EMOTION_ALIAS

SPLITS = ["train", "dev", "test"]


# ---------------------------------------------------------------------------------------------
# MELD CSVs and audio
# ---------------------------------------------------------------------------------------------

def find_meld_files(root: str) -> Dict[str, Path]:
    root = Path(root)
    mapping = {}
    csv_patterns = {"train": "train_sent_emo", "dev": "dev_sent_emo", "test": "test_sent_emo"}
    for split, pat in csv_patterns.items():
        hits = list(root.rglob(f"{pat}.csv"))
        if hits:
            mapping[f"{split}_csv"] = hits[0]
    return mapping


def load_meld_csv(csv_path: Path, emotion2id: Dict[str, int]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        if "utterance_id" in c: rename[c] = "utterance_id"
        elif "emotion" in c:    rename[c] = "emotion"
        elif "dialogue_id" in c: rename[c] = "dialogue_id"
        elif "utterance" in c:  rename[c] = "utterance"
    df = df.rename(columns=rename)
    df["emotion"] = df["emotion"].str.lower().map(EMOTION_ALIAS).fillna("neutral")
    df["emotion_id"] = df["emotion"].map(emotion2id)
    if "dialogue_id" in df.columns and "utterance_id" in df.columns:
        df["video_fname"] = df.apply(lambda r: f"dia{int(r.dialogue_id)}_utt{int(r.utterance_id)}.mp4", axis=1)
    return df


def import_wav_paths(df: pd.DataFrame, audio_dir: Path, split_name: str) -> pd.DataFrame:
    split_dir = Path(audio_dir) / split_name
    paths = []
    for _, row in df.iterrows():
        wav = split_dir / row["video_fname"].replace(".mp4", ".wav")
        paths.append(str(wav) if wav.exists() else None)
    df = df.copy()
    df["wav_path"] = paths
    df["split"] = split_name   # lets Config D rebuild a train-split-only pool
    return df


def load_meld(cfg) -> Dict[str, pd.DataFrame]:
    """Every MELD split that has a CSV, restricted to rows with a matching WAV."""
    meld_files = find_meld_files(cfg.kaggle_input_dir)
    for k, v in meld_files.items():
        print(f"  {k:12s} -> {v}")

    processed = {}
    for split in SPLITS:
        key = f"{split}_csv"
        if key not in meld_files:
            continue
        df = load_meld_csv(meld_files[key], cfg.emotion2id)
        print(f"{split:5s}: {len(df)} rows")
        d = import_wav_paths(df, cfg.audio_dir, split)
        before = len(d)
        d = d[d["wav_path"].notna()].reset_index(drop=True)
        processed[split] = d
        print(f"[{split}] {len(d)}/{before} rows have a matching WAV file")
    return processed


def split_train_holdout(processed: Dict[str, pd.DataFrame], cfg):
    combined_df = pd.concat([processed[s] for s in SPLITS if s in processed], ignore_index=True)
    print(f"\nCombined train+dev+test pool: {len(combined_df)} utterances with matched audio")
    print(combined_df["emotion"].value_counts())

    holdout_parts, train_parts = [], []
    for emotion, group in combined_df.groupby("emotion"):
        group = group.sample(frac=1.0, random_state=cfg.seed)
        # never hold out more than 20% of a class, so this can't gut an already-tiny class's training data
        n_hold = min(cfg.eval_holdout_per_emotion, max(1, len(group) // 5))
        holdout_parts.append(group.iloc[:n_hold])
        train_parts.append(group.iloc[n_hold:])

    holdout_df = pd.concat(holdout_parts, ignore_index=True).reset_index(drop=True)
    train_df = pd.concat(train_parts, ignore_index=True).reset_index(drop=True)

    if cfg.debug_subset:
        train_df = train_df.sample(n=min(cfg.debug_subset, len(train_df)), random_state=cfg.seed).reset_index(drop=True)

    print(f"\nTraining set (combined pool minus held-out eval): {len(train_df)} utterances")
    print(train_df["emotion"].value_counts())
    print(f"\nHeld-out evaluation set (never trained on): {len(holdout_df)} utterances")
    print(holdout_df["emotion"].value_counts())
    return train_df, holdout_df


# ---------------------------------------------------------------------------------------------
# Class-balanced weights + calibrated text->duration model
# ---------------------------------------------------------------------------------------------

def add_class_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Inverse-frequency weight per row for WeightedRandomSampler, computed within `df`'s own pool."""
    df = df.copy()
    counts = df["emotion"].value_counts()
    df["sample_weight"] = df["emotion"].map(lambda e: 1.0 / counts[e])
    return df


def fit_duration_model(train_df: pd.DataFrame, cfg):
    """Least-squares fit of seconds ~ num_chars on real MELD pairs; writes the result into cfg."""
    char_lens, durations = [], []
    for _, row in train_df.iterrows():
        try:
            info = sf.info(row["wav_path"])
            durations.append(info.frames / info.samplerate)
            char_lens.append(len(str(row["utterance"])))
        except Exception:
            continue

    char_lens = np.array(char_lens, dtype=float)
    durations = np.array(durations, dtype=float)
    A = np.vstack([char_lens, np.ones_like(char_lens)]).T
    slope, intercept = np.linalg.lstsq(A, durations, rcond=None)[0]
    cfg.dur_slope, cfg.dur_intercept = float(slope), float(intercept)
    print(f"\nCalibrated duration model: seconds = {cfg.dur_slope:.4f} * num_chars + {cfg.dur_intercept:.3f}")
    print(f"(fit on {len(char_lens)} real MELD utterance/duration pairs)")


def calibrated_duration_frames(text: str, cfg, min_sec: float = 0.5, max_sec: float = 8.0) -> int:
    sec = cfg.dur_slope * len(text) + cfg.dur_intercept
    sec = float(np.clip(sec, min_sec, max_sec))
    return max(1, int(sec * cfg.sample_rate / cfg.latent_hop))


# ---------------------------------------------------------------------------------------------
# Dataset & DataLoader — variable length, properly masked
# ---------------------------------------------------------------------------------------------
# Each item keeps its real (capped) length, and collate_fn pads only to that *batch's* own max,
# while real_frame_lens lets the training loss mask out the padded region.

class EmotionalSpeechDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav, _ = librosa.load(row["wav_path"], sr=self.cfg.sample_rate, mono=True)
        max_cap = int(self.cfg.max_audio_sec * self.cfg.sample_rate)
        if len(wav) > max_cap:
            wav = wav[:max_cap]
        emotion = str(row.get("emotion", "neutral"))
        text = str(row.get("utterance", "")).strip()
        prompt = f"[{emotion.upper()}] {text}"
        return {"wav": torch.from_numpy(wav).float(), "prompt": prompt}


def collate_fn(batch, cfg):
    lens = [b["wav"].shape[0] for b in batch]
    max_len_raw = max(lens)
    max_len = ((max_len_raw + cfg.latent_hop - 1) // cfg.latent_hop) * cfg.latent_hop
    max_len = max(max_len, cfg.latent_hop)

    wavs = torch.zeros(len(batch), 1, max_len)
    real_frame_lens = []
    for i, b in enumerate(batch):
        L = b["wav"].shape[0]
        wavs[i, 0, :L] = b["wav"]
        real_frame_lens.append(max(1, L // cfg.latent_hop))
    prompts = [b["prompt"] for b in batch]
    return {"wav": wavs, "prompt": prompts,
            "real_frame_lens": torch.tensor(real_frame_lens, dtype=torch.long)}


def make_loader(df: pd.DataFrame, cfg, balanced: bool = True, dataset: Dataset = None) -> DataLoader:
    """Class-balanced (WeightedRandomSampler over `df`'s own class counts) or naive shuffled loader."""
    dataset = dataset if dataset is not None else EmotionalSpeechDataset(df, cfg)
    collate = partial(collate_fn, cfg=cfg)
    if not balanced:
        return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=cfg.num_workers, drop_last=True)
    weights = df["sample_weight"].values if "sample_weight" in df.columns \
        else add_class_weights(df)["sample_weight"].values
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(df), replacement=True)
    return DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler,
                      collate_fn=collate, num_workers=cfg.num_workers, drop_last=True)
