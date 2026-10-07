"""Speaker self-consistency (WavLM x-vectors).

Similarity to MELD references would measure resemblance to a random *Friends* actor. Instead:
1. Is the fine-tuned voice still one voice? Mean pairwise cosine across all fine-tuned clips vs
   the same for zero-shot.
2. Did fine-tuning change who is speaking? Cosine between the zero-shot and fine-tuned rendering
   of the *same* sentence.
Real MELD clips (many speakers) give the low end of the scale.
"""
import librosa
import numpy as np
import pandas as pd
import torch

from ..stats import describe_diff, paired_bootstrap_diff

REAL_MELD_CAP = 120


def speaker_consistency(ctx):
    cfg = ctx.cfg
    spk_results = {}
    if not cfg.run_speaker_consistency:
        return spk_results
    try:
        from transformers import AutoFeatureExtractor, WavLMForXVector
        spk_fe = AutoFeatureExtractor.from_pretrained(cfg.speaker_model_id)
        spk_model = WavLMForXVector.from_pretrained(cfg.speaker_model_id).to(ctx.device).eval()

        @torch.no_grad()
        def spk_embed(wav, sr):
            w = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=16000) \
                if sr != 16000 else np.asarray(wav, dtype=np.float32)
            inp = spk_fe(w, sampling_rate=16000, return_tensors="pt").to(ctx.device)
            e = spk_model(**inp).embeddings
            return torch.nn.functional.normalize(e, dim=-1).squeeze(0).cpu().numpy()

        def mean_pairwise(E):
            S = E @ E.T
            iu = np.triu_indices(len(E), k=1)
            return float(S[iu].mean()), float(S[iu].std())

        final_eval_df, zeroshot_df = ctx.final_eval_df, ctx.zeroshot_df
        E_ft = np.stack([spk_embed(w, cfg.sample_rate) for w in final_eval_df["wav"]])
        spk_results["fine_tuned"] = mean_pairwise(E_ft)
        if zeroshot_df is not None:
            E_zs = np.stack([spk_embed(librosa.load(p, sr=cfg.sample_rate, mono=True)[0], cfg.sample_rate)
                             for p in zeroshot_df["wav_file"]])
            spk_results["zero_shot"] = mean_pairwise(E_zs)
        real_paths = [p for p in ctx.holdout_df["wav_path"].dropna()
                      if librosa.get_duration(path=p) >= 1.0][:REAL_MELD_CAP]
        E_real = np.stack([spk_embed(librosa.load(p, sr=16000, mono=True)[0], 16000) for p in real_paths])
        spk_results["real_meld_multispeaker"] = mean_pairwise(E_real)

        print("Within-set speaker consistency (mean pairwise cosine; higher = more like one voice):")
        for k, (m, s_) in spk_results.items():
            print(f"  {k:24s} {m:.3f}  (sd {s_:.3f})")

        if zeroshot_df is not None:
            id_ft = final_eval_df["utt_id"].tolist()
            id_zs = zeroshot_df["utt_id"].tolist()
            common = [u for u in id_ft if u in set(id_zs)]
            same = np.array([float(E_ft[id_ft.index(u)] @ E_zs[id_zs.index(u)]) for u in common])
            print(f"\n  Same sentence, zero-shot vs fine-tuned: mean cosine {same.mean():.3f} "
                  f"(sd {same.std():.3f}, n={len(same)})")
            spk_results["same_sentence_ft_vs_zs"] = (float(same.mean()), float(same.std()))
            # within-system per-clip consistency: similarity of each clip to its own system centroid
            c_ft = E_ft.mean(0); c_ft /= np.linalg.norm(c_ft)
            c_zs = E_zs.mean(0); c_zs /= np.linalg.norm(c_zs)
            res = paired_bootstrap_diff(
                [float(E_ft[id_ft.index(u)] @ c_ft) for u in common],
                [float(E_zs[id_zs.index(u)] @ c_zs) for u in common], n_boot=cfg.n_boot_diff)
            print("  Similarity to own-system centroid: " + describe_diff(res, "fine-tuned", "zero-shot"))
            print("  (negative = the fine-tuned voice is less stable as an identity)")
            spk_results["centroid_diff"] = (res.get("diff"), res.get("ci_low"), res.get("ci_high"))

        pd.DataFrame([{"set": k, "value": v[0], "spread_or_ci": v[1:]} for k, v in spk_results.items()]) \
            .to_csv(f"{cfg.output_dir}/speaker_consistency.csv", index=False)
        spk_model = None   # release the model before emptying the CUDA cache
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Speaker-consistency check skipped ({e}).")
    return spk_results
