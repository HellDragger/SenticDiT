# SenticDiT

Code for *Revisiting Emotional TTS Fine-Tuning on MELD: What Actually Matters*.

SenticDiT fine-tunes [`meituan-longcat/LongCat-AudioDiT-1B`](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B)
— a conditional-flow-matching diffusion transformer over a continuous VAE latent — on MELD with
LoRA, conditioning on emotion by prefixing the text prompt (`[JOY] Hello!`). It then evaluates the
result with a deliberately skeptical protocol: a zero-shot control, paired bootstrap tests,
corrected MCD, UTMOS, Whisper WER, two speech-emotion classifiers, speaker consistency, and a
single-factor data-quantity experiment (Config D).

## Pipeline

1. **Data** — MELD train+dev+test are pooled and a stratified held-out set (30 per emotion) is
   carved out and never trained on. Class-balanced sampling via `WeightedRandomSampler`.
2. **Model** — the real pretrained AudioDiT-1B (UMT5 text encoder, DiT backbone, VAE decoder; no
   mel-spectrogram or vocoder stage), with a weight-loading guard that stops the run if the loaded
   tensors don't match the checkpoint.
3. **Training** — masked rectified-flow loss on VAE latents, LoRA on attention Q/K/V (optionally
   the AdaLN MLP), fully resumable checkpoints for multi-session Kaggle runs.
4. **Ablations** — sampling strategy, LoRA rank (16/32/64), LoRA target modules.
5. **Evaluation** — generation health (voiced fraction, F0 with an octave-error cross-check),
   CFG sweep, SER (wav2vec2 + emotion2vec, with a real-audio control), DTW MCD with corrupt
   references filtered, UTMOS and the pitch-drift mechanism, Whisper WER, WavLM speaker
   consistency, Config D, and an assembled `paper_results_table.csv`.

## Repository structure

```text
SenticDiT/
├── senticdit/                 # the package — all pipeline code
│   ├── config.py              # Config dataclass: every knob, with rationale
│   ├── setup_env.py           # Kaggle bootstrap: pip installs + clone LongCat-AudioDiT
│   ├── data.py                # MELD loading, held-out split, weights, duration model, Dataset
│   ├── model.py               # model loading, weight-loading guard, LoRA helpers
│   ├── training.py            # flow-matching loss, checkpoints/resume, training loop
│   ├── generation.py          # generate_speech, generate_and_score, prompt formats
│   ├── metrics.py             # generation-health metrics (completion, F0)
│   ├── stats.py               # bootstrap / paired bootstrap / Wilson / MDE
│   ├── ablations.py           # sampling, rank, target-module ablations
│   ├── config_d.py            # data quantity at fixed compute
│   ├── evaluation/
│   │   ├── health.py          # zero-shot, CFG sweep, final eval, paraphrase, demos
│   │   ├── ser.py             # wav2vec2 SER + diagnostics, emotion2vec
│   │   ├── mcd.py             # DTW MCD + alignment-artefact control
│   │   ├── utmos.py           # UTMOS, real-MELD anchor, pitch-drift mechanism
│   │   ├── wer.py             # Whisper WER / CER
│   │   └── speaker.py         # WavLM speaker self-consistency
│   ├── report.py              # paper results table, cross-run log, manifest
│   └── pipeline.py            # SenticDiTRun: runs every stage in order
├── senticdit_kaggle.ipynb     # Kaggle runner: clone repo -> config -> run
├── scripts/
│   ├── run_pipeline.py        # full pipeline from the command line
│   ├── demo_inference.py      # synthesize one sentence (optionally with a LoRA adapter)
│   └── meld_eda.py            # dataset EDA figures
├── plots/                     # EDA figures and figures from the earlier v1 pipeline
├── Report.pdf
├── Papers.zip                 # referenced papers
└── requirements.txt
```

## Data

Add these as Kaggle inputs (or download them into `data/` for local runs and point the config at them):

- MELD Dataset — https://www.kaggle.com/datasets/zaber666/meld-dataset
- MELD Audio (pre-extracted WAVs) — https://www.kaggle.com/datasets/aryansharma26/meld-audio
- Checkpoints — https://www.kaggle.com/datasets/aryansharma26/checkpoints

## Running on Kaggle

1. Upload `senticdit_kaggle.ipynb` to Kaggle (GPU T4 ×2, Internet on) and add the datasets above.
2. Edit the configuration cell. To continue a previous main run, add that run's output as an input
   and set `resume_search_dir` to its `checkpoints/<config_tag>` directory.
3. Run all. The notebook clones this repo (`REPO_REF`, default `main`), installs dependencies,
   and calls `SenticDiTRun(CFG).run_all()`. Everything lands in `CFG.output_dir`.

To debug one stage, call the `SenticDiTRun` methods individually (`prepare_data`, `load_model`,
`zero_shot`, `run_ablations`, `train_main`, `final_eval`, …) instead of `run_all()`.

With the ablations skipped and a complete resume checkpoint, a full run takes ~6.5–7.5 h on 2×T4
(Config D is ~4.3 h of that). Re-running the ablations adds ~5 h — use a second session.

## Running locally

```bash
pip install -r requirements.txt
```

```bash
python scripts/run_pipeline.py --meld-dir data/MELD.Raw --audio-dir data/MELD_audio \
    --output-dir outputs --longcat-dir third_party/LongCat-AudioDiT --skip-ablations --rank 16 --no-adaln
```

Generate one sentence with a trained adapter:

```bash
python scripts/demo_inference.py --text "I cannot believe this is happening right now!" \
    --emotion surprise --adapter outputs/lora_final --output surprise.wav
```

Available emotions: neutral, surprise, fear, sadness, joy, disgust, anger.

## Replication

Each evaluation block reseeds from `eval_seed`, so arms don't depend on how much RNG earlier
stages consumed. To replicate, bump `run_tag` and `eval_seed` and rerun; `run_log.csv`
accumulates runs and reports the spread of every metric across them.
