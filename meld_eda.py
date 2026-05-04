"""
MELD Dataset EDA for Research Paper (Headless CPU Execution)
Generates high-resolution, paper-ready plots for emotion-conditioned TTS analysis.
"""

import os
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np

# Visualization
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Kaggle headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Audio processing
import librosa
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
KAGGLE_INPUT_DIR = "/kaggle/input/datasets/zaber666/meld-dataset/MELD-RAW/MELD.Raw"
AUDIO_DIR = Path("/kaggle/input/datasets/aryansharma26/meld-audio/MELD_audio")
OUTPUT_DIR = Path("/kaggle/working/paper_eda_plots")

EMOTIONS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
EMOTION_ALIAS = {
    "neutral": "neutral", "surprise": "surprise", "surprised": "surprise",
    "fear": "fear", "fearful": "fear", "sadness": "sadness", "sad": "sadness",
    "joy": "joy", "happy": "joy", "happiness": "joy", "disgust": "disgust",
    "anger": "anger", "angry": "anger",
}

# Paper-ready styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
PALETTE = sns.color_palette("Set2", len(EMOTIONS))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_and_merge_data():
    print("Loading text and aligning with audio paths...")
    dfs = []
    root = Path(KAGGLE_INPUT_DIR)
    
    for split in ["train", "dev", "test"]:
        csv_hits = list(root.rglob(f"{split}_sent_emo.csv"))
        if not csv_hits:
            continue
            
        df = pd.read_csv(csv_hits[0])
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Normalize columns
        rename_map = {c: "utterance_id" for c in df.columns if "utterance_id" in c}
        rename_map.update({c: "dialogue_id" for c in df.columns if "dialogue_id" in c})
        df = df.rename(columns=rename_map)
        
        df["emotion"] = df["emotion"].str.lower().map(EMOTION_ALIAS).fillna("neutral")
        df["video_fname"] = df.apply(lambda r: f"dia{int(r.dialogue_id)}_utt{int(r.utterance_id)}.mp4", axis=1)
        df["split"] = split
        
        # Attach WAV paths
        split_dir = AUDIO_DIR / split
        df["wav_path"] = df["video_fname"].apply(
            lambda f: str(split_dir / f.replace(".mp4", ".wav")) if (split_dir / f.replace(".mp4", ".wav")).exists() else None
        )
        
        dfs.append(df)
        
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.dropna(subset=["wav_path", "utterance"])
    print(f"Total valid samples matched with audio: {len(final_df)}")
    return final_df

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION (Multiprocessed for CPU)
# ═══════════════════════════════════════════════════════════════════════════════
def extract_audio_features(wav_path):
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        duration = len(y) / sr
        # Calculate RMS energy as a proxy for vocal intensity/loudness
        rms = librosa.feature.rms(y=y)[0]
        mean_energy = np.mean(rms)
        return duration, mean_energy
    except Exception:
        return None, None

def parallel_feature_extraction(df):
    print("Extracting acoustic features (Duration, Energy) on CPU...")
    # Using ProcessPool to speed up CPU-bound librosa tasks
    paths = df["wav_path"].tolist()
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(extract_audio_features, paths), total=len(paths)))
        
    df["duration_sec"] = [r[0] for r in results]
    df["mean_energy"] = [r[1] for r in results]
    df["word_count"] = df["utterance"].apply(lambda x: len(str(x).split()))
    
    return df.dropna(subset=["duration_sec"])

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_emotion_distribution(df):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x="emotion", order=EMOTIONS, palette=PALETTE, edgecolor="black")
    plt.title("Utterance Distribution across Emotion Classes")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Utterances")
    
    # Add counts above bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, xytext=(0, 5), 
                    textcoords='offset points')
                    
    plt.savefig(OUTPUT_DIR / "fig1_emotion_distribution.png")
    plt.close()

def plot_duration_by_emotion(df):
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x="emotion", y="duration_sec", order=EMOTIONS, palette=PALETTE, inner="quartile")
    plt.title("Audio Duration Distribution per Emotion")
    plt.xlabel("Emotion")
    plt.ylabel("Duration (seconds)")
    plt.ylim(0, df["duration_sec"].quantile(0.99)) # Cut off extreme outliers for a cleaner plot
    plt.savefig(OUTPUT_DIR / "fig2_audio_duration_violin.png")
    plt.close()

def plot_text_length_by_emotion(df):
    plt.figure(figsize=(12, 6))
    sns.boxenplot(data=df, x="emotion", y="word_count", order=EMOTIONS, palette=PALETTE)
    plt.title("Text Utterance Length (Word Count) per Emotion")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Words")
    plt.ylim(0, df["word_count"].quantile(0.99))
    plt.savefig(OUTPUT_DIR / "fig3_text_length_boxen.png")
    plt.close()

def plot_acoustic_energy(df):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df, x="mean_energy", hue="emotion", common_norm=False, fill=True, alpha=0.3, palette=PALETTE)
    plt.title("Acoustic Energy (RMS) Density by Emotion")
    plt.xlabel("Mean RMS Energy")
    plt.ylabel("Density")
    plt.xlim(0, df["mean_energy"].quantile(0.98))
    plt.savefig(OUTPUT_DIR / "fig4_acoustic_energy_kde.png")
    plt.close()

def plot_wordclouds(df):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, emotion in enumerate(EMOTIONS):
        text = " ".join(df[df["emotion"] == emotion]["utterance"].astype(str))
        wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis", max_words=100).generate(text)
        
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(emotion.capitalize(), fontsize=18, fontweight="bold")
        axes[i].axis("off")
        
    axes[-1].axis("off") # Hide the 8th empty subplot
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_emotion_wordclouds.png")
    plt.close()

def plot_scatter_duration_vs_words(df):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="word_count", y="duration_sec", hue="emotion", 
                    palette=PALETTE, alpha=0.6, s=20)
    
    # Add a regression line for the overall trend
    sns.regplot(data=df, x="word_count", y="duration_sec", scatter=False, color="black", line_kws={"linestyle": "--"})
    
    plt.title("Speaking Rate: Word Count vs. Audio Duration")
    plt.xlabel("Word Count")
    plt.ylabel("Duration (seconds)")
    plt.xlim(0, df["word_count"].quantile(0.99))
    plt.ylim(0, df["duration_sec"].quantile(0.99))
    plt.savefig(OUTPUT_DIR / "fig6_speaking_rate_scatter.png")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Starting EDA Pipeline...")
    df = load_and_merge_data()
    
    if not df.empty:
        df = parallel_feature_extraction(df)
        
        print(f"Generating plots in {OUTPUT_DIR} ...")
        plot_emotion_distribution(df)
        plot_duration_by_emotion(df)
        plot_text_length_by_emotion(df)
        plot_acoustic_energy(df)
        plot_scatter_duration_vs_words(df)
        plot_wordclouds(df)
        
        # Save the enriched dataframe for future fast-loading
        df.drop(columns=["wav_path"]).to_csv(OUTPUT_DIR / "meld_enriched_eda.csv", index=False)
        
        print("✅ EDA Complete. All figures saved.")
    else:
        print("❌ Dataframes are empty. Check Kaggle input paths.")