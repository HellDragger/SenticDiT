"""Experiment "addon": a second corpus and second-opinion metrics (senticdit_addon.ipynb).

Part A — CREMA-D with voice-cloned references. On MELD every MCD reference is a different
speaker, so MCD cannot carry a fidelity claim. Here each target sentence is synthesised in the
actor's own voice (prompted by the same actor saying a different sentence) and scored against
(1) that actor's real recording — a valid, same-speaker MCD — and (2) a different actor's
recording of the same sentence and emotion — the MELD situation; plus the model's default voice
against the actor, and real-actor-vs-real-actor as the pure speaker distance.

Part B — second opinions on MELD. The 210 held-out utterances are regenerated (zero-shot and
fine-tuned, same seeds as v5) and scored with DNSMOS and NISQA as second MOS predictors, a CTC
wav2vec 2.0 recogniser (which cannot hallucinate long insertions) as a second ASR model, and a
second speaker-embedding model. DNSMOS's background score tests whether the adapter absorbed
MELD's recording conditions.

Inputs: the two MELD datasets, CREMA-D (`ejlok1/cremad`), and the fine-tuned adapter
(`aryansharma26/senticdit-r-16`, see `Config.adapter_dir`). Internet on. ~1.6 h on 2x T4.
"""
import gc
import re
import time
from pathlib import Path

import jiwer
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from ..context import RunContext
from ..cremad import CREMA_EMO, find_cremad, generate_cloned, install_extras, load16, select_targets
from ..evaluation.ser import Emotion2Vec, SERClassifier, ser_diagnostics
from ..evaluation.speaker import (embed_normalized, load_ecapa, load_wav2vec2_xvector,
                                  wavlm_embeddings)
from ..evaluation.utmos import UTMOSScorer
from ..evaluation.wer import WhisperWER
from ..generation import bracket_prompt, generate_and_score, generate_speech
from ..model import summarize_lora_targets
from ..stats import bootstrap_ci, paired_bootstrap_diff
from ..utils import reseed, section, zip_outputs


def _norm(t):
    t = str(t).replace("\x92", "'").replace("’", "'").replace("Â", "").lower()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------------------

def generate_part_a(ctx, targets, wav_dir, cfg_strength):
    """Cloned voice and default voice for every CREMA-D target."""
    cfg = ctx.cfg
    manifest = []      # every clip to be scored: (part, source, key, emotion, text, path)
    t0 = time.time()
    reseed(cfg, "addon_cremad_cloned")
    for _, r in targets.iterrows():
        p = wav_dir / f"cremad_cloned_{r.key}.wav"
        sf.write(p, generate_cloned(ctx.model, ctx.tokenizer, cfg, ctx.device, r.text, r.prompt_path,
                                    r.prompt_text, cfg_strength), cfg.sample_rate)
        manifest.append(("A", "cloned", r.key, r.emotion, r.text, str(p)))
    reseed(cfg, "addon_cremad_default")
    for _, r in targets.iterrows():
        p = wav_dir / f"cremad_default_{r.key}.wav"
        generate_speech(ctx.model, ctx.tokenizer, r.text, cfg, ctx.device, emotion=None,
                        cfg_strength=cfg_strength, out_path=str(p))
        manifest.append(("A", "default_voice", r.key, r.emotion, r.text, str(p)))
    for _, r in targets.iterrows():
        manifest.append(("A", "real_actor", r.key, r.emotion, r.text, r.ref_same))
        manifest.append(("A", "real_other_actor", r.key, r.emotion, r.text, r.ref_other))
    print(f"Part A generation: {2 * len(targets)} clips in {(time.time() - t0) / 60:.1f} min")
    return manifest


def generate_part_b(ctx, out_dir, wav_dir, cfg_strength):
    """Regenerate the held-out MELD clips, zero-shot then fine-tuned, with v5's seeds."""
    from peft import PeftModel

    cfg = ctx.cfg
    manifest = []
    t0 = time.time()
    reseed(cfg, "zeroshot")                                   # same tag and seed as the v5 run
    zs_df = generate_and_score(ctx.model, ctx.tokenizer, ctx.holdout_df, cfg, ctx.device,
                               emotions=cfg.emotions, n_per_emotion=None, cfg_strength=cfg_strength,
                               steps=cfg.cfm_steps, prompt_fn=bracket_prompt, return_wavs=True)
    for _, r in zs_df.iterrows():
        p = wav_dir / f"meld_zs_{r.utt_id}.wav"; sf.write(p, r.wav, cfg.sample_rate)
        manifest.append(("B", "zero_shot", str(r.utt_id), r.emotion, r.text, str(p)))

    ctx.model.transformer = PeftModel.from_pretrained(ctx.model.transformer, cfg.adapter_dir)
    ctx.model.transformer.to(ctx.device).eval()
    summarize_lora_targets(ctx.model.transformer)
    reseed(cfg, "final_eval")
    ft_df = generate_and_score(ctx.model, ctx.tokenizer, ctx.holdout_df, cfg, ctx.device,
                               emotions=cfg.emotions, n_per_emotion=None, cfg_strength=cfg_strength,
                               steps=cfg.cfm_steps, prompt_fn=bracket_prompt, return_wavs=True)
    for _, r in ft_df.iterrows():
        p = wav_dir / f"meld_ft_{r.utt_id}.wav"; sf.write(p, r.wav, cfg.sample_rate)
        manifest.append(("B", "fine_tuned", str(r.utt_id), r.emotion, r.text, str(p)))

    n_anchor = 0
    for emo in cfg.emotions:                              # stratified real-MELD anchor, >= 2 s
        taken = 0
        for idx, r in ctx.holdout_df[ctx.holdout_df.emotion == emo].iterrows():
            if taken >= cfg.meld_anchor_per_emotion:
                break
            if r.get("wav_path") and librosa.get_duration(path=r["wav_path"]) >= 2.0:
                manifest.append(("B", "real_meld", str(idx), emo, str(r["utterance"]), r["wav_path"]))
                taken += 1; n_anchor += 1
    pd.concat([zs_df.drop(columns=["wav"]).assign(source="zero_shot"),
               ft_df.drop(columns=["wav"]).assign(source="fine_tuned")]).to_csv(
        out_dir / "meld_regenerated_health.csv", index=False)
    print(f"Part B generation: {len(zs_df) + len(ft_df)} clips + {n_anchor} real anchors "
          f"in {(time.time() - t0) / 60:.1f} min")

    if cfg.v5_dir:
        try:
            v5 = pd.read_csv(Path(cfg.v5_dir) / "health_metrics_final.csv")
            j = ft_df.merge(v5, on="utt_id", suffixes=("_new", "_v5"))
            print(f"Determinism check vs v5 (fine-tuned): corr(f0_median) = "
                  f"{j['f0_median_new'].corr(j['f0_median_v5']):.4f}, "
                  f"corr(total_dur) = {j['total_dur_new'].corr(j['total_dur_v5']):.4f}")
        except Exception as e:
            print(f"determinism check skipped ({e})")
    return manifest


# ---------------------------------------------------------------------------------------------
# Scoring — one evaluator at a time
# ---------------------------------------------------------------------------------------------

def score_manifest(manifest, cfg, device, out_dir):
    """Each evaluator is loaded, run over every clip, and freed before the next one, so memory
    cannot accumulate the way it did in v5. A failure in one evaluator is reported and skipped."""
    scores = manifest.copy()
    emb = {}

    def _run(name, fn):
        t0 = time.time()
        try:
            fn()
            print(f"  {name:22s} done in {(time.time() - t0) / 60:.1f} min")
        except Exception as e:
            print(f"  {name:22s} FAILED ({type(e).__name__}: {e})")
        gc.collect(); torch.cuda.empty_cache()

    def s_utmos():
        m = UTMOSScorer(device)
        scores["utmos"] = [m.score16(load16(p)) for p in scores.path]

    def s_dnsmos():
        from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score as dnsmos
        vals = [dnsmos(torch.from_numpy(load16(p)), 16000, False).numpy() for p in scores.path]
        v = np.stack(vals)                                   # [p808, sig, bak, ovrl]
        scores["dnsmos_p808"], scores["dnsmos_sig"], scores["dnsmos_bak"], scores["dnsmos_ovrl"] = v.T

    def s_nisqa():
        from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment as nisqa
        v = np.stack([nisqa(torch.from_numpy(load16(p)), 16000).numpy() for p in scores.path])
        for i, k in enumerate(["nisqa_mos", "nisqa_noisiness", "nisqa_discontinuity", "nisqa_coloration", "nisqa_loudness"]):
            scores[k] = v[:, i]

    def s_whisper():
        asr = WhisperWER(cfg.asr_model_id)
        hyp = [asr.transcribe(load16(p), 16000) for p in scores.path]
        scores["hyp_whisper"] = [_norm(h) or "<empty>" for h in hyp]

    def s_w2v_ctc():
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        mid = "facebook/wav2vec2-large-960h-lv60-self"
        proc = Wav2Vec2Processor.from_pretrained(mid); m = Wav2Vec2ForCTC.from_pretrained(mid).to(device).eval()
        out = []
        with torch.no_grad():
            for p in scores.path:
                iv = proc(load16(p), sampling_rate=16000, return_tensors="pt").input_values.to(device)
                out.append(_norm(proc.batch_decode(m(iv).logits.argmax(-1))[0]) or "<empty>")
        scores["hyp_w2vctc"] = out

    def s_wavlm():
        emb["wavlm"] = wavlm_embeddings([load16(p) for p in scores.path], cfg.speaker_model_id, device)

    def s_spk2():
        """Second speaker model: SpeechBrain ECAPA-TDNN; falls back to a wav2vec2 x-vector model
        if SpeechBrain cannot import against this torch/torchaudio build."""
        try:
            f = load_ecapa(device)
            emb["spk2_name"] = "ECAPA-TDNN (SpeechBrain)"
        except Exception as e:
            print(f"    SpeechBrain unavailable ({type(e).__name__}); using wav2vec2 x-vector instead")
            f = load_wav2vec2_xvector(device)
            emb["spk2_name"] = "wav2vec2 x-vector (anton-l/wav2vec2-base-superb-sv)"
        emb["spk2"] = embed_normalized(f, [load16(p) for p in scores.path])

    def s_ser():
        ser = SERClassifier(cfg.ser_model_id, device)
        preds = []
        for p in scores.path:
            y, _ = librosa.load(p, sr=cfg.sample_rate, mono=True)
            preds.append(ser.predict(y, cfg.sample_rate))
        scores["ser_w2v"] = preds

    def s_e2v():
        e2v = Emotion2Vec(cfg.emotion2vec_model_id)
        scores["ser_e2v"] = [e2v.predict(load16(p), 16000) for p in scores.path]

    print(f"Scoring {len(scores)} clips:")
    for name, fn in [("UTMOS", s_utmos), ("DNSMOS", s_dnsmos), ("NISQA", s_nisqa),
                     ("Whisper", s_whisper), ("wav2vec2-CTC ASR", s_w2v_ctc),
                     ("WavLM speaker", s_wavlm), ("second speaker model", s_spk2),
                     ("wav2vec2 SER", s_ser), ("emotion2vec SER", s_e2v)]:
        _run(name, fn)

    for col in ["hyp_whisper", "hyp_w2vctc"]:
        if col in scores:
            tag = col.split("_")[1]
            scores[f"wer_{tag}"] = [jiwer.wer(_norm(t), h) if _norm(t) else np.nan for t, h in zip(scores.text, scores[col])]
            scores[f"wer_{tag}_clip"] = scores[f"wer_{tag}"].clip(upper=1.0)
    scores.to_csv(out_dir / "scores_per_clip.csv", index=False)
    for k in ["wavlm", "spk2"]:
        if k in emb:
            np.save(out_dir / f"emb_{k}.npy", emb[k])
    return scores, emb


# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------

def part_b_results(scores, emb, cfg, out_dir):
    """Fine-tuned minus zero-shot, paired by utterance, for every second-opinion metric, with the
    real-MELD anchor. Does each reproduce the UTMOS drop and the intelligibility gain?"""
    B = scores[scores.part == "B"].copy()
    zsB = B[B.source == "zero_shot"].set_index("key"); ftB = B[B.source == "fine_tuned"].set_index("key")
    anc = B[B.source == "real_meld"]
    rowsB = []
    metrics = [c for c in ["utmos", "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "dnsmos_p808", "nisqa_mos",
                           "nisqa_noisiness", "nisqa_discontinuity", "nisqa_coloration", "nisqa_loudness",
                           "wer_whisper_clip", "wer_w2vctc_clip"] if c in B]
    print(f"{'metric':22s} {'zero-shot':>9s} {'fine-tuned':>10s} {'real MELD':>9s}   paired difference")
    for col in metrics:
        k = ftB.index.intersection(zsB.index)
        r = paired_bootstrap_diff(ftB.loc[k, col], zsB.loc[k, col], n_boot=cfg.n_boot_diff)
        rowsB.append({"metric": col, "zero_shot": zsB[col].mean(), "fine_tuned": ftB[col].mean(),
                      "real_meld": anc[col].mean(), **r})
        print(f"{col:22s} {zsB[col].mean():9.3f} {ftB[col].mean():10.3f} {anc[col].mean():9.3f}   "
              f"{r['diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] {'*' if r['significant'] else ''}")
    for col in ["hyp_whisper", "hyp_w2vctc"]:
        if col in B:
            cw = lambda X: jiwer.wer([_norm(t) for t in X.text], X[col].tolist())  # noqa: E731
            print(f"corpus WER ({col[4:]}): zero-shot {cw(zsB):.3f} | fine-tuned {cw(ftB):.3f} | real {cw(anc):.3f}")

    mos_cols = [c for c in ["utmos", "dnsmos_ovrl", "nisqa_mos"] if c in B]
    if len(mos_cols) > 1:
        print("\nPer-clip agreement between MOS predictors (Spearman, all MELD-part clips):")
        print(B[mos_cols].corr(method="spearman").round(3).to_string())

    for k in ["wavlm", "spk2"]:
        if k in emb:
            E = emb[k]; idx = {(s, key): i for i, (s, key) in enumerate(zip(scores.source, scores.key))}
            common = [key for key in ftB.index if ("zero_shot", key) in idx]
            Eft = np.stack([E[idx[("fine_tuned", key)]] for key in common])
            Ezs = np.stack([E[idx[("zero_shot", key)]] for key in common])
            cft = Eft.mean(0) / np.linalg.norm(Eft.mean(0)); czs = Ezs.mean(0) / np.linalg.norm(Ezs.mean(0))
            r = paired_bootstrap_diff(Eft @ cft, Ezs @ czs, n_boot=cfg.n_boot_diff)
            name = emb.get("spk2_name", "WavLM") if k == "spk2" else "WavLM-base-plus-sv"
            print(f"\nSpeaker stability, {name}: similarity to own-system centroid, fine-tuned - zero-shot "
                  f"{r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
            rowsB.append({"metric": f"centroid_similarity_{k}", **r})
    pd.DataFrame(rowsB).to_csv(out_dir / "partB_paired.csv", index=False)
    print("\nReading DNSMOS: if fine-tuning lowers BAK (background) toward the real-MELD value while SIG")
    print("changes less, the adapter absorbed MELD's recording conditions, not only its speakers.")


def part_a_results(scores, emb, targets, cfg, out_dir):
    """Valid MCD next to invalid MCD: the same cloned utterance is scored against the same actor's
    recording and a different actor's recording of the same sentence and emotion. Holding the
    generated clip fixed isolates the effect of reference speaker on MCD."""
    from pymcd.mcd import Calculate_MCD
    mcd_modes = {m: Calculate_MCD(MCD_mode=m) for m in ["dtw", "plain", "dtw_sl"]}
    A = scores[scores.part == "A"]
    path_of = {(s, k): p for s, k, p in zip(A.source, A.key, A.path)}
    pairs = [("cloned", "real_actor", "cloned vs same actor (valid)"),
             ("cloned", "real_other_actor", "cloned vs different actor"),
             ("default_voice", "real_actor", "default voice vs actor (MELD-like)"),
             ("real_other_actor", "real_actor", "real vs real, different actors")]
    mrows = []
    for _, r in targets.iterrows():
        for gen_src, ref_src, label in pairs:
            g, ref = path_of.get((gen_src, r.key)), path_of.get((ref_src, r.key))
            if not g or not ref:
                continue
            gd, rd = librosa.get_duration(path=g), librosa.get_duration(path=ref)
            rec = {"key": r.key, "emotion": r.emotion, "pair": label, "dur_ratio": gd / rd,
                   "abs_log_dur_ratio": abs(np.log(gd / rd))}
            for m, c in mcd_modes.items():
                try:
                    rec[f"mcd_{m}"] = c.calculate_mcd(ref, g)
                except Exception:
                    rec[f"mcd_{m}"] = np.nan
            mrows.append(rec)
    mcdA = pd.DataFrame(mrows); mcdA.to_csv(out_dir / "partA_mcd.csv", index=False)

    print("MCD on CREMA-D by comparison (mean, bootstrapped 95% CI):")
    for _, _, label in pairs:
        d = mcdA[mcdA.pair == label]
        line = f"  {label:38s} n={len(d):3d}"
        for m in ["dtw", "plain", "dtw_sl"]:
            mu, lo, hi = bootstrap_ci(d[f"mcd_{m}"], n_boot=cfg.bootstrap_n)
            line += f" | {m} {mu:5.2f} [{lo:5.2f},{hi:5.2f}]"
        line += f" | corr(dur, dtw) {d['abs_log_dur_ratio'].corr(d['mcd_dtw']):+.2f}"
        print(line)

    same = mcdA[mcdA.pair == pairs[0][2]].set_index("key"); oth = mcdA[mcdA.pair == pairs[1][2]].set_index("key")
    k = same.index.intersection(oth.index)
    r = paired_bootstrap_diff(oth.loc[k, "mcd_dtw"], same.loc[k, "mcd_dtw"], n_boot=cfg.n_boot_diff)
    print(f"\nSame cloned clip, different-actor minus same-actor reference (dtw): "
          f"{r['diff']:+.2f} dB [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}], n={r['n']}")
    print("  ^ the pure effect of reference speaker on MCD, with the generated audio held fixed.")

    # speaker similarity: does cloning actually reproduce the actor?
    if "wavlm" in emb:
        idx = {(s, key): i for i, (s, key) in enumerate(zip(scores.source, scores.key))}
        print("\nSpeaker similarity to the actor's real target recording (WavLM; second model in brackets):")
        for src_name in ["cloned", "default_voice", "real_other_actor"]:
            sims, sims2 = [], []
            for key in targets.key:
                if (src_name, key) in idx and ("real_actor", key) in idx:
                    i, j = idx[(src_name, key)], idx[("real_actor", key)]
                    sims.append(float(emb["wavlm"][i] @ emb["wavlm"][j]))
                    if "spk2" in emb:
                        sims2.append(float(emb["spk2"][i] @ emb["spk2"][j]))
            print(f"  {src_name:18s} {np.mean(sims):.3f}" + (f" [{np.mean(sims2):.3f}]" if sims2 else "") + f"  n={len(sims)}")

    # quality, intelligibility and SER on clean acted speech
    print("\nQuality / intelligibility / SER by source (CREMA-D):")
    cols = [c for c in ["utmos", "dnsmos_ovrl", "dnsmos_bak", "nisqa_mos", "wer_whisper_clip", "wer_w2vctc_clip"] if c in A]
    print(A.groupby("source")[cols].mean().round(3).to_string())
    for sc in ["ser_w2v", "ser_e2v"]:
        if sc in A:
            print(f"\n{sc}:")
            for src_name in ["real_actor", "cloned", "default_voice"]:
                d = A[A.source == src_name]
                ser_diagnostics(d.emotion.tolist(), d[sc].fillna("none").tolist(),
                                labels=list(CREMA_EMO.values()), title=f"{sc} on {src_name}")

    # the alignment-artefact control on clean acted speech
    y, _ = librosa.load(targets.iloc[0].ref_same, sr=22050, mono=True)
    sf.write("/tmp/_a.wav", y, 22050)
    print("\nTime-stretch self-comparison on a real CREMA-D recording:")
    for st_ in [1.05, 1.15, 1.30]:
        sf.write("/tmp/_b.wav", librosa.effects.time_stretch(y, rate=st_), 22050)
        print(f"  stretch {st_:.2f}: " + " | ".join(f"{m} {c.calculate_mcd('/tmp/_a.wav', '/tmp/_b.wav'):.2f}"
                                                    for m, c in mcd_modes.items()))


# ---------------------------------------------------------------------------------------------

def run(cfg):
    ctx = RunContext(cfg)
    ctx.prepare_data()
    ctx.load_model(snapshot=False)

    out_dir = Path(cfg.output_dir) / "addon"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wavs"; wav_dir.mkdir(exist_ok=True)
    cfg_strength = float(cfg.cfg_strength)
    # no [audio] extra for torchmetrics: it pulls pesq, which may not compile
    install_extras("torchmetrics>=1.7", "onnxruntime", "requests", "funasr", "speechbrain")
    print("settings ok | adapter exists:", Path(cfg.adapter_dir).exists())

    section("Part A · CREMA-D — target selection")
    targets = pd.DataFrame()
    if cfg.addon_run_part_a:
        targets = select_targets(find_cremad(cfg.cremad_dir), cfg)
    targets.to_csv(out_dir / "cremad_targets.csv", index=False)

    manifest = []
    if cfg.addon_run_part_a and len(targets):
        section("Part A · generation — cloned voice and default voice")
        manifest += generate_part_a(ctx, targets, wav_dir, cfg_strength)
    if cfg.addon_run_part_b:
        section("Part B · MELD — regenerate zero-shot and fine-tuned clips (same seeds as v5)")
        manifest += generate_part_b(ctx, out_dir, wav_dir, cfg_strength)

    manifest = pd.DataFrame(manifest, columns=["part", "source", "key", "emotion", "text", "path"])
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(manifest.groupby(["part", "source"]).size().to_string())
    ctx.free_model()       # generation done: every scorer gets the GPU to itself

    section("Scoring — one evaluator at a time")
    scores, emb = score_manifest(manifest, cfg, ctx.device, out_dir)
    if cfg.addon_run_part_b:
        section("Part B results — do the second opinions agree?")
        part_b_results(scores, emb, cfg, out_dir)
    if cfg.addon_run_part_a and len(targets):
        section("Part A results — valid MCD next to invalid MCD")
        part_a_results(scores, emb, targets, cfg, out_dir)

    if cfg.zip_results:
        zip_outputs(Path(cfg.output_dir).parent / "addon_results", Path(cfg.output_dir), ["addon"])
    return scores
