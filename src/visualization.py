import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use("Agg") # Non-interactive backend suitable for scripts/servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List
from pathlib import Path

from src.config import CFG
from src.data_loader import wav_to_mel
from src.inference import generate_emotional_speech, safe_write_wav

# ── Exploratory Data Analysis ─────────────────────────────────────────────────

def plot_emotion_distribution(dfs_dict: Dict[str, pd.DataFrame], save_path: str) -> None:
    fig, axes = plt.subplots(1, len(dfs_dict), figsize=(6 * len(dfs_dict), 4))
    if len(dfs_dict) == 1:
        axes = [axes]
        
    palette = plt.cm.Set2(np.linspace(0, 1, len(CFG.emotions)))
    
    for ax, (split, df) in zip(axes, dfs_dict.items()):
        counts = df["emotion"].value_counts().reindex(CFG.emotions, fill_value=0)
        bars = ax.bar(counts.index, counts.values, color=palette, edgecolor="white", linewidth=0.8)
        ax.set_title(f"MELD — {split.capitalize()} split", fontsize=13, fontweight="bold")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=40)
        
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(val), ha="center", va="bottom", fontsize=8)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")

def get_audio_durations(df: pd.DataFrame) -> List[float]:
    durations = []
    for wav_path in df["wav_path"]:
        try:
            info = sf.info(wav_path)
            durations.append(info.frames / info.samplerate)
        except Exception:
            pass
    return durations

def plot_duration_histograms(dfs_dict: Dict[str, pd.DataFrame], save_path: str) -> None:
    fig, axes = plt.subplots(1, len(dfs_dict), figsize=(6 * len(dfs_dict), 4), sharey=False)
    if len(dfs_dict) == 1: 
        axes = [axes]
        
    for ax, (split, df) in zip(axes, dfs_dict.items()):
        durs = get_audio_durations(df)
        ax.hist(durs, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.axvline(np.median(durs), color="#DD8452", linestyle="--", label=f"Median {np.median(durs):.1f}s")
        ax.axvline(CFG.max_audio_sec, color="red", linestyle=":", label=f"Clip limit {CFG.max_audio_sec}s")
        ax.set_title(f"{split.capitalize()} — Duration", fontsize=12, fontweight="bold")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")


# ── Training & Evaluation Plots ───────────────────────────────────────────────

def plot_loss_curve(losses: List[float], save_path: str, smooth_window: int = 20) -> None:
    if not losses:
        print("No step_losses available to plot.")
        return
        
    arr    = np.array(losses)
    kernel = np.ones(smooth_window) / smooth_window
    smooth = np.convolve(arr, kernel, mode="valid")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(arr, alpha=0.35, color="#4C72B0", linewidth=0.8, label="Step loss")
    ax.plot(range(smooth_window - 1, len(arr)), smooth,
            color="#DD8452", linewidth=2.0, label=f"Smoothed ({smooth_window}-step MA)")
            
    ax.set_title("AudioDiT LoRA Fine-tuning — Loss Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")

def plot_mel_comparison(
    original_wav_path: str, 
    text: str, 
    emotion: str, 
    save_path: str,
    audiodit_model,
    text_enc,
    sched
) -> None:
    """Plots a side-by-side comparison of the ground truth mel vs generated mel."""
    orig_mel = wav_to_mel(original_wav_path)
    
    gen_wav = generate_emotional_speech(
        text=text, 
        emotion_label=emotion,
        audiodit_model=audiodit_model,
        text_enc=text_enc,
        sched=sched
    )
    
    tmp_path = "/tmp/_comparison_gen.wav"
    safe_write_wav(tmp_path, gen_wav)
    gen_mel = wav_to_mel(tmp_path)

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05)
    
    for col, (mel, title) in enumerate([
        (orig_mel, "Ground Truth"),
        (gen_mel,  f"Generated [{emotion.upper()}]"),
    ]):
        ax = fig.add_subplot(gs[col])
        img = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Bins" if col == 0 else "")
        if col == 1: 
            ax.set_yticks([])
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="dB")
        
    suptitle = f'"{text[:60]}…"' if len(text) > 60 else f'"{text}"'
    fig.suptitle(suptitle, fontsize=11, style="italic", y=1.02)
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")