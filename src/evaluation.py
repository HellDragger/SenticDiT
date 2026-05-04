import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from pymcd.mcd import Calculate_MCD
    MCD_AVAILABLE = True
except ImportError:
    MCD_AVAILABLE = False
    print("⚠  pymcd not found – MCD evaluation will be skipped.")

from src.config import CFG
from src.inference import generate_emotional_speech, safe_write_wav

def compute_mcd_score(ref_wav_path: str, gen_wav: np.ndarray, sr: int = CFG.sample_rate) -> float:
    """Computes the Mel Cepstral Distortion between a reference WAV and a generated waveform."""
    if not MCD_AVAILABLE:
        return float("nan")
        
    tmp_gen = "/tmp/_gen_eval.wav"
    safe_write_wav(tmp_gen, gen_wav, sr)
    
    try:
        mcd_calc = Calculate_MCD(MCD_mode="plain")
        return mcd_calc.calculate_mcd(ref_wav_path, tmp_gen)
    except Exception as e:
        print(f"  MCD error: {e}")
        return float("nan")

def run_mcd_evaluation_per_emotion(
    df: pd.DataFrame, 
    audiodit_model, 
    text_enc, 
    sched
) -> pd.DataFrame:
    """
    Evaluates MCD on exactly one random sample per emotion class present in the DataFrame.
    Saves the results to a CSV in the output directory.
    """
    print(f"Starting MCD evaluation per emotion...")
    # Group by emotion and sample exactly 1 row per group
    subset = df.groupby("emotion").sample(n=1, random_state=CFG.seed)
    results = []
    
    for _, row in subset.iterrows():
        emotion_label = row.get("emotion", "neutral")
        utterance = row.get("utterance", "")
        
        # Generate the audio using the live model components
        gen_wav = generate_emotional_speech(
            text=utterance,
            emotion_label=emotion_label,
            audiodit_model=audiodit_model,
            text_enc=text_enc,
            sched=sched
        )
        
        mcd = compute_mcd_score(row["wav_path"], gen_wav)
        
        results.append({
            "utterance": utterance,
            "emotion": emotion_label, 
            "mcd": mcd
        })
        print(f"  MCD [{emotion_label:9s}]: {mcd:.2f} dB")
        
    df_out = pd.DataFrame(results)
    mean_mcd = df_out['mcd'].mean()
    
    print(f"\nMean MCD across all tested emotions: {mean_mcd:.2f} dB")
    
    # Save results
    out_path = Path(CFG.output_dir) / "mcd_results_per_emotion.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved MCD results -> {out_path}")
    
    return df_out