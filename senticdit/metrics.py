"""Generation-health metrics.

F0 band: fmax is 400 Hz rather than C6 (1046.5 Hz), and every clip is cross-checked against
WORLD's `harvest`. A pyin/WORLD median ratio near 2.0 or 0.5 is an octave error and is flagged
per clip rather than averaged in silently. Medians replace means.

Completion: v2's `trim(wav)/total_dur` is kept as `duration_fit` — it scores the duration model,
not utterance completion. `completion_ratio` now means the fraction of the trimmed clip above
the silence floor, so internal pauses count.
"""
import librosa
import numpy as np

try:
    import pyworld as pw
    _PYWORLD_OK = True
except Exception:
    _PYWORLD_OK = False
    print("pyworld unavailable — F0 octave cross-check disabled (pyin values unverified).")


def compute_health_metrics(wav: np.ndarray, sr: int, cfg) -> dict:
    total_dur = len(wav) / sr
    y_trim, _ = librosa.effects.trim(wav, top_db=30)
    speech_dur = len(y_trim) / sr

    # -- two quantities v2 conflated into one ------------------------------------------------
    # duration_fit: trimmed length / requested length. total_dur comes from the calibrated
    # duration model, so this scores the duration predictor.
    duration_fit = speech_dur / total_dur if total_dur > 0 else 0.0
    # voiced_fraction: fraction of the trimmed clip actually above the silence floor.
    if len(y_trim):
        intervals = librosa.effects.split(y_trim, top_db=30)
        voiced_fraction = float(sum(e - s for s, e in intervals) / len(y_trim))
    else:
        voiced_fraction = 0.0

    rms = float(np.sqrt(np.mean(y_trim ** 2))) if len(y_trim) else 0.0

    f0_mean = f0_median = f0_std = f0_iqr = float("nan")
    voiced_ratio = 0.0
    world_median = pyin_world_ratio = float("nan")
    octave_flag = False
    try:
        f0, voiced, _ = librosa.pyin(y_trim, fmin=cfg.f0_floor_hz, fmax=cfg.f0_ceiling_hz, sr=sr)
        f0v = f0[np.isfinite(f0)]
        voiced_ratio = float(np.mean(voiced)) if voiced is not None and len(voiced) else 0.0
        if len(f0v):
            f0_mean = float(np.mean(f0v))
            f0_median = float(np.median(f0v))
            f0_std = float(np.std(f0v))
            f0_iqr = float(np.subtract(*np.percentile(f0v, [75, 25])))
            if cfg.f0_octave_check and _PYWORLD_OK:
                x = np.asarray(y_trim, dtype=np.float64)
                wf0, wt = pw.harvest(x, sr, f0_floor=cfg.f0_floor_hz, f0_ceil=cfg.f0_ceiling_hz)
                wf0 = pw.stonemask(x, wf0, wt, sr)
                wv = wf0[wf0 > 0]
                if len(wv):
                    world_median = float(np.median(wv))
                    pyin_world_ratio = f0_median / world_median
                    octave_flag = bool(abs(pyin_world_ratio - 2.0) < 0.25
                                       or abs(pyin_world_ratio - 0.5) < 0.12)
    except Exception:
        pass

    # -- diagnostic pass at a wider ceiling ----------------------------------------------------
    # Fine-tuning moved this voice up by roughly 9.5 semitones, so the 400 Hz bound is also
    # clipping a real effect. Measuring at 600 Hz says how much.
    f0_median_wide = float("nan")
    f0_at_bound = False
    try:
        fw, _, _ = librosa.pyin(y_trim, fmin=cfg.f0_floor_hz,
                                 fmax=cfg.f0_diagnostic_ceiling_hz, sr=sr)
        fwv = fw[np.isfinite(fw)]
        if len(fwv):
            f0_median_wide = float(np.median(fwv))
            # a median within 5% of the primary ceiling means the narrow band is truncating
            f0_at_bound = bool(np.isfinite(f0_median) and f0_median > 0.95 * cfg.f0_ceiling_hz)
    except Exception:
        pass

    return {
        "total_dur": total_dur, "speech_dur": speech_dur,
        "completion_ratio": voiced_fraction,   # corrected definition
        "voiced_fraction": voiced_fraction,
        "duration_fit": duration_fit,          # v2's completion_ratio, under its real name
        "rms": rms,
        "f0_mean": f0_mean, "f0_median": f0_median, "f0_std": f0_std, "f0_iqr": f0_iqr,
        "voiced_ratio": voiced_ratio,
        "f0_world_median": world_median, "f0_pyin_world_ratio": pyin_world_ratio,
        "f0_octave_flag": octave_flag,
        "f0_median_wide": f0_median_wide,
        "f0_at_bound": f0_at_bound,
        "f0_valid": bool(np.isfinite(f0_median)),
    }
