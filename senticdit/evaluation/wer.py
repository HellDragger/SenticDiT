"""Intelligibility — Whisper WER / CER.

UTMOS says the fine-tuned voice sounds worse; it does not say whether it is still understood.
Whisper transcribes every clip; WER and CER are computed against the prompt text after Whisper's
English normaliser. Real MELD recordings are an anchor for how hard the *text* is, not a target.
"""
import re

import jiwer
import librosa
import numpy as np
import pandas as pd
import torch

from ..stats import describe_diff, paired_bootstrap_diff


class WhisperWER:
    def __init__(self, model_id):
        from transformers import pipeline
        self.asr = pipeline("automatic-speech-recognition", model=model_id,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device=0 if torch.cuda.is_available() else -1)

    def normalize(self, t):
        t = str(t).replace("\x92", "'").replace("’", "'").replace("Â", "")
        try:
            t = self.asr.tokenizer.normalize(t)          # Whisper's EnglishTextNormalizer
        except Exception:
            t = re.sub(r"[^a-z0-9' ]+", " ", t.lower())
        return re.sub(r"\s+", " ", t).strip()

    def transcribe(self, wav, sr):
        w = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=16000) \
            if sr != 16000 else np.asarray(wav, dtype=np.float32)
        out = self.asr({"raw": w, "sampling_rate": 16000},
                       generate_kwargs={"language": "english", "task": "transcribe"})
        return out["text"]

    def wer_rows(self, df, wav_getter, label):
        """Per-utterance WER/CER for every row of df; wav_getter(row) -> (wav, sr)."""
        rows = []
        for _, r in df.iterrows():
            try:
                hyp = self.transcribe(*wav_getter(r))
            except Exception:
                continue
            ref_n, hyp_n = self.normalize(r["text"]), self.normalize(hyp)
            if not ref_n:
                continue
            rows.append({"utt_id": r["utt_id"], "emotion": r["emotion"], "ref": ref_n, "hyp": hyp_n,
                         "wer": jiwer.wer(ref_n, hyp_n if hyp_n else "<empty>"),
                         "cer": jiwer.cer(ref_n, hyp_n if hyp_n else "<empty>")})
        out = pd.DataFrame(rows)
        if len(out):
            print(f"  {label:22s} corpus WER {corpus_wer(out):.3f} | mean CER {out['cer'].mean():.3f} | n={len(out)}")
        return out


def corpus_wer(rows: pd.DataFrame) -> float:
    return jiwer.wer(rows["ref"].tolist(), [h if h else "<empty>" for h in rows["hyp"]])


def wer_evaluation(ctx):
    """Returns (asr, wer_results); asr is None if Whisper could not be loaded."""
    cfg = ctx.cfg
    wer_results = {}
    if not cfg.run_wer:
        return None, wer_results
    try:
        asr = WhisperWER(cfg.asr_model_id)
        final_eval_df, zeroshot_df = ctx.final_eval_df, ctx.zeroshot_df

        print("Whisper WER (lower is better):")
        wer_ft = asr.wer_rows(final_eval_df, lambda r: (r["wav"], cfg.sample_rate), "fine-tuned")
        wer_results["fine_tuned"] = wer_ft
        if zeroshot_df is not None:
            wer_results["zero_shot"] = asr.wer_rows(
                zeroshot_df,
                lambda r: (librosa.load(r["wav_file"], sr=cfg.sample_rate, mono=True)[0], cfg.sample_rate),
                "zero-shot")

        real = ctx.holdout_df[ctx.holdout_df["wav_path"].notna()].copy()
        real["utt_id"], real["text"] = real.index, real["utterance"]
        real = real[real["wav_path"].map(lambda p: librosa.get_duration(path=p) >= 1.0)]
        wer_results["real_meld"] = asr.wer_rows(
            real, lambda r: (librosa.load(r["wav_path"], sr=cfg.sample_rate, mono=True)[0],
                             cfg.sample_rate), "real MELD (>=1 s)")

        if "zero_shot" in wer_results and len(wer_ft) and len(wer_results["zero_shot"]):
            mg = wer_ft.merge(wer_results["zero_shot"], on="utt_id", suffixes=("_ft", "_zs"))
            for col in ["wer", "cer"]:
                res = paired_bootstrap_diff(mg[f"{col}_ft"], mg[f"{col}_zs"], n_boot=cfg.n_boot_diff)
                print(f"\n  {col.upper()}: " + describe_diff(res, "fine-tuned", "zero-shot"))
            print("\n  Per-emotion WER (fine-tuned / zero-shot):")
            print(mg.groupby("emotion_ft")[["wer_ft", "wer_zs"]].mean().round(3).to_string())
            # does the pitch drift also cost intelligibility?
            if "f0_median" in final_eval_df.columns:
                j = mg.merge(final_eval_df[["utt_id", "f0_median"]], on="utt_id")
                print(f"\n  corr(f0_median_ft, WER_ft) = {j['f0_median'].corr(j['wer_ft']):+.3f}")

        pd.concat([d.assign(source=k) for k, d in wer_results.items() if len(d)],
                  ignore_index=True).to_csv(f"{cfg.output_dir}/wer_per_utterance.csv", index=False)
        return asr, wer_results
    except Exception as e:
        print(f"WER skipped ({e}). Needs internet on the Kaggle session.")
        return None, wer_results
