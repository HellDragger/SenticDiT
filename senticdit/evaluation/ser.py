"""Automatic speech-emotion recognition: the wav2vec2 classifier, its collapse diagnostics, the
real-audio control, and emotion2vec as a second classifier.

The wav2vec2 checkpoint's real head is `classifier.dense -> tanh -> classifier.out_proj`, which
matches neither `Wav2Vec2ForCTC` nor HF's `Wav2Vec2ForSequenceClassification`; both silently
discard the trained head and substitute a random one. We define the exact head ourselves and load
the real weights into it — the missing/unexpected key lists printed on load must both be empty.
"""
import subprocess
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from ..stats import wilson_ci

SER_LABEL_NORM = {"happy": "joy", "angry": "anger", "sad": "sadness",
                   "fearful": "fear", "surprised": "surprise", "calm": "neutral"}

E2V_MAP = {"angry": "anger", "disgusted": "disgust", "fearful": "fear", "happy": "joy",
           "neutral": "neutral", "sad": "sadness", "surprised": "surprise"}


class _Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _Wav2Vec2ClassificationHead(config)
        # Deliberately skip self.init_weights()/post_init(): in this transformers version it
        # calls tie_weights(), which hits missing `all_tied_weights_keys` machinery. Every
        # weight is overwritten by load_state_dict anyway.

    def forward(self, input_values, attention_mask=None):
        hidden_states = self.wav2vec2(input_values, attention_mask=attention_mask)[0]
        pooled = torch.mean(hidden_states, dim=1)   # simple mean-pool over time
        return self.classifier(pooled)


class SERClassifier:
    """wav2vec2 SER checkpoint with its real head; `predict` returns a MELD emotion label."""

    def __init__(self, model_id, device):
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, Wav2Vec2FeatureExtractor

        self.device = device
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

        # .from_pretrained() on a bare custom PreTrainedModel subclass hits an internal
        # incompatibility in this transformers version. Build from config, then load the real
        # weights manually; load_state_dict's return value gives the same key verification.
        config = AutoConfig.from_pretrained(model_id)
        model = Wav2Vec2ForSpeechClassification(config)
        weights_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        load_result = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:   ", load_result.missing_keys)
        print("Unexpected keys:", load_result.unexpected_keys)
        print("^ both should be empty. If not, this class still doesn't match — do not trust the "
              "numbers below in that case.")

        self.model = model.to(device).eval()
        print("SER label set:", self.model.config.id2label)

    @torch.no_grad()
    def predict(self, wav, sr):
        wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
        inputs = self.extractor(wav16, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
        logits = self.model(inputs.input_values)
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(torch.argmax(probs, dim=-1).item())
        label = self.model.config.id2label[pred_id].lower()
        return SER_LABEL_NORM.get(label, label)


def ser_diagnostics(y_true, y_pred, labels, title=""):
    """Agreement is not enough on a collapsed classifier. Report what collapse looks like."""
    y_true, y_pred = list(y_true), list(y_pred)
    n = len(y_true)
    if n == 0:
        print(f"{title}: no samples"); return {}
    agreement = sum(t == p for t, p in zip(y_true, y_pred)) / n
    per_class = []
    for lab in labels:
        idx = [i for i, t in enumerate(y_true) if t == lab]
        if idx:
            per_class.append(sum(y_pred[i] == lab for i in idx) / len(idx))
    balanced = float(np.mean(per_class)) if per_class else float("nan")
    counts = {lab: y_pred.count(lab) for lab in labels}
    modal = max(counts, key=counts.get)
    collapse = counts[modal] / n
    probs = np.array([c / n for c in counts.values() if c > 0])
    entropy = float(-np.sum(probs * np.log2(probs)))
    p_e = sum((y_true.count(l) / n) * (y_pred.count(l) / n) for l in labels)
    kappa = (agreement - p_e) / (1 - p_e) if p_e < 1 else float("nan")
    lo, hi = wilson_ci(int(agreement * n), n)

    print(f"--- {title} (n={n}) ---")
    print(f"  agreement        {agreement:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (chance {1/len(labels):.3f})")
    print(f"  balanced acc     {balanced:.3f}")
    print(f"  Cohen's kappa    {kappa:+.3f}   (<=0 means no better than the marginals)")
    print(f"  modal prediction '{modal}' takes {collapse:.1%} of all predictions")
    print(f"  prediction entropy {entropy:.2f} / {np.log2(len(labels)):.2f} bits")
    never = [l for l, c in counts.items() if c == 0]
    if never:
        print(f"  never predicted: {never}")
    return {"agreement": agreement, "balanced_accuracy": balanced, "kappa": kappa,
            "modal_label": modal, "collapse_index": collapse, "entropy_bits": entropy, "n": n}


def ser_evaluation(ctx):
    """SER on the fine-tuned output, then the decisive controls: the same classifier on REAL MELD
    recordings and on the zero-shot output.

    Accuracy clearly above chance on real audio -> the wrapper is sound and the anger-collapse is
    genuine domain mismatch on synthetic voices. Collapse on real audio too -> the wrapper is
    wrong and no SER-based claim should be reported.

    Returns (classifier, ser_stats, bracket_accuracy).
    """
    cfg = ctx.cfg
    ser = SERClassifier(cfg.ser_model_id, ctx.device)

    # Sanity check: the untouched pretrained baseline should read as roughly neutral/calm.
    if ctx.baseline_wav is not None:
        sanity_pred = ser.predict(ctx.baseline_wav, cfg.sample_rate)
        print(f"Sanity check — baseline (no LoRA, should read ~neutral) classified as: {sanity_pred}")
    else:
        print("baseline_wav not in memory this session — skipping sanity check.")

    ser_rows = []
    for _, row in ctx.final_eval_df.iterrows():
        pred = ser.predict(row["wav"], cfg.sample_rate)
        ser_rows.append({"true_emotion": row["emotion"], "predicted_emotion": pred, "correct": pred == row["emotion"]})

    ser_df = pd.DataFrame(ser_rows)
    ser_accuracy = ser_df["correct"].mean()
    print(f"Overall SER-classifier agreement: {ser_accuracy:.1%}\n")
    print("Per-emotion agreement:")
    print(ser_df.groupby("true_emotion")["correct"].mean().round(3))

    confusion = pd.crosstab(ser_df["true_emotion"], ser_df["predicted_emotion"])
    print("\nConfusion matrix:")
    print(confusion)

    ser_df.to_csv(f"{cfg.output_dir}/ser_predictions.csv", index=False)
    confusion.to_csv(f"{cfg.output_dir}/ser_confusion_matrix.csv")

    # --- Is the classifier broken, or is the voice out of domain? -------------------------------
    ser_stats = {}
    ser_stats["synthetic_finetuned"] = ser_diagnostics(
        ser_df["true_emotion"], ser_df["predicted_emotion"], cfg.emotions,
        title="Fine-tuned synthetic audio")

    if cfg.run_ser_real_audio_check:
        real_rows = []
        for _, r in ctx.holdout_df.iterrows():
            wp = r.get("wav_path")
            if not wp:
                continue
            try:
                y, _sr = librosa.load(wp, sr=cfg.sample_rate, mono=True)
            except Exception:
                continue
            if len(y) < int(0.3 * cfg.sample_rate):
                continue
            real_rows.append({"true_emotion": r["emotion"],
                               "predicted_emotion": ser.predict(y, cfg.sample_rate)})
        ser_real_df = pd.DataFrame(real_rows)
        if len(ser_real_df):
            print()
            ser_stats["real_meld"] = ser_diagnostics(
                ser_real_df["true_emotion"], ser_real_df["predicted_emotion"], cfg.emotions,
                title="REAL MELD recordings (same classifier)")
            print("\nConfusion matrix, real audio:")
            print(pd.crosstab(ser_real_df["true_emotion"], ser_real_df["predicted_emotion"]))
            ser_real_df.to_csv(f"{cfg.output_dir}/ser_predictions_real_audio.csv", index=False)

            r_acc = ser_stats["real_meld"]["agreement"]
            r_col = ser_stats["real_meld"]["collapse_index"]
            s_col = ser_stats["synthetic_finetuned"]["collapse_index"]
            print("\n=== VERDICT ===")
            if r_acc > 1.5 / len(cfg.emotions) and r_col < 0.35:
                print("  The classifier works on real speech and collapses only on synthetic speech.")
                print("  -> The anger-collapse is a domain-mismatch property of the CLASSIFIER.")
                print("     The paper's SER-bias finding stands, and is now demonstrated rather than")
                print("     attributed. Report both matrices side by side; that figure is the result.")
            else:
                print(f"  The classifier ALSO fails on real MELD audio "
                      f"(agreement {r_acc:.3f}, collapse {r_col:.1%} vs {s_col:.1%} on synthetic).")
                print("  -> The wrapper is the likely culprit, not the synthetic voice. Suspect the")
                print("     mean-pool in Wav2Vec2ForSpeechClassification.forward. Do NOT report any")
                print("     SER-based claim until this is resolved against the reference implementation.")

    if ctx.zeroshot_df is not None:
        zs_rows = []
        for _, r in ctx.zeroshot_df.iterrows():
            y, _sr = librosa.load(r["wav_file"], sr=cfg.sample_rate, mono=True)
            zs_rows.append({"true_emotion": r["emotion"],
                             "predicted_emotion": ser.predict(y, cfg.sample_rate)})
        ser_zs_df = pd.DataFrame(zs_rows)
        print()
        ser_stats["synthetic_zeroshot"] = ser_diagnostics(
            ser_zs_df["true_emotion"], ser_zs_df["predicted_emotion"], cfg.emotions,
            title="Zero-shot pretrained synthetic audio")
        ser_zs_df.to_csv(f"{cfg.output_dir}/ser_predictions_zeroshot.csv", index=False)
        print("\n  Fine-tuning moved SER agreement from "
              f"{ser_stats['synthetic_zeroshot']['agreement']:.3f} (zero-shot) to "
              f"{ser_stats['synthetic_finetuned']['agreement']:.3f} (fine-tuned).")

    pd.DataFrame(ser_stats).T.to_csv(f"{cfg.output_dir}/ser_diagnostics.csv")
    return ser, ser_stats, ser_accuracy


# ---------------------------------------------------------------------------------------------
# emotion2vec — a second SER model
# ---------------------------------------------------------------------------------------------

def emotion2vec_evaluation(ctx):
    """With one classifier the paper can only say *that checkpoint* is biased on synthetic
    speech. If emotion2vec (different architecture, different data) shows the same pattern, the
    finding is about SER-based evaluation in general. Predictions are restricted to the seven
    MELD classes by taking the argmax over those seven scores."""
    cfg = ctx.cfg
    e2v_stats = {}
    if not cfg.run_emotion2vec:
        return e2v_stats
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "funasr"], check=False)
        from funasr import AutoModel as FunASRModel
        e2v = FunASRModel(model=cfg.emotion2vec_model_id, hub="hf", disable_update=True)
        _tmp = "/tmp/_e2v.wav"

        def e2v_predict(wav, sr):
            w = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=16000) \
                if sr != 16000 else np.asarray(wav, dtype=np.float32)
            sf.write(_tmp, w, 16000)
            res = e2v.generate(_tmp, granularity="utterance", extract_embedding=False)[0]
            best, best_s = None, -1.0
            for lab, sc in zip(res["labels"], res["scores"]):
                eng = lab.split("/")[-1].strip().lower()
                if eng in E2V_MAP and sc > best_s:
                    best, best_s = E2V_MAP[eng], float(sc)
            return best

        def e2v_run(items, title):
            t, p = [], []
            for emo, wav, sr in items:
                try:
                    pred = e2v_predict(wav, sr)
                except Exception:
                    continue
                if pred is not None:
                    t.append(emo); p.append(pred)
            print()
            return (ser_diagnostics(t, p, cfg.emotions, title=title),
                    pd.DataFrame({"true_emotion": t, "predicted_emotion": p}))

        sources = {
            "fine_tuned": [(r["emotion"], r["wav"], cfg.sample_rate) for _, r in ctx.final_eval_df.iterrows()],
            "real_meld": [(r["emotion"], librosa.load(r["wav_path"], sr=16000, mono=True)[0], 16000)
                          for _, r in ctx.holdout_df.iterrows()
                          if r.get("wav_path") and librosa.get_duration(path=r["wav_path"]) >= 0.3],
        }
        if ctx.zeroshot_df is not None:
            sources["zero_shot"] = [(r["emotion"], librosa.load(r["wav_file"], sr=cfg.sample_rate, mono=True)[0],
                                     cfg.sample_rate) for _, r in ctx.zeroshot_df.iterrows()]

        e2v_preds = []
        for key, items in sources.items():
            st, dfp = e2v_run(items, f"emotion2vec — {key}")
            e2v_stats[key] = st
            e2v_preds.append(dfp.assign(source=key))
        pd.concat(e2v_preds, ignore_index=True).to_csv(f"{cfg.output_dir}/ser_emotion2vec_predictions.csv", index=False)

        print("\n=== Two classifiers, three audio sources ===")
        rows = []
        for key in ["real_meld", "zero_shot", "fine_tuned"]:
            w2v_key = {"real_meld": "real_meld", "zero_shot": "synthetic_zeroshot",
                       "fine_tuned": "synthetic_finetuned"}[key]
            for name, stats in [("wav2vec2", ctx.ser_stats.get(w2v_key)), ("emotion2vec", e2v_stats.get(key))]:
                if stats:
                    rows.append({"classifier": name, "audio": key, "agreement": round(stats["agreement"], 3),
                                 "kappa": round(stats["kappa"], 3), "collapse": round(stats["collapse_index"], 3),
                                 "modal": stats["modal_label"]})
        cmp_df = pd.DataFrame(rows)
        print(cmp_df.to_string(index=False))
        cmp_df.to_csv(f"{cfg.output_dir}/ser_two_classifier_comparison.csv", index=False)

        r, s_ = e2v_stats.get("real_meld"), e2v_stats.get("fine_tuned")
        if r and s_:
            print("\nVERDICT:")
            if r["kappa"] > s_["kappa"] + 0.05 or s_["collapse_index"] > r["collapse_index"] + 0.10:
                print("  emotion2vec also degrades on synthetic speech relative to real speech.")
                print("  -> The SER-bias finding generalises beyond one checkpoint. State it as such.")
            else:
                print("  emotion2vec does NOT show the same degradation on synthetic speech.")
                print("  -> The bias is specific to the wav2vec2 checkpoint. Narrow the claim.")
        e2v = None   # release the model before emptying the CUDA cache
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"emotion2vec skipped ({e}).")
    return e2v_stats
