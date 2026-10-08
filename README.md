# SenticDiT

Code for *When Automatic Metrics Mislead: Auditing the Evaluation of LoRA Fine-Tuning for Emotional Speech Synthesis on MELD*.

SenticDiT fine-tunes [`meituan-longcat/LongCat-AudioDiT-1B`](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B)
— a conditional-flow-matching diffusion transformer over a continuous VAE latent — on MELD with
LoRA, conditioning on emotion by prefixing the text prompt (`[JOY] Hello!`). It then evaluates the
result with a deliberately skeptical protocol: a zero-shot control, paired bootstrap tests,
corrected MCD, several MOS / ASR / speaker / emotion models, a single-factor data-quantity
experiment, and same-speaker MCD on a second corpus (CREMA-D) via voice cloning.

## Links

- **Paper (arXiv preprint):** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) <!-- TODO: replace with the arXiv ID -->
- **MELD dataset:** [MELD](https://www.kaggle.com/datasets/zaber666/meld-dataset) <!-- TODO: replace with the MELD link used -->
- **CREMA-D dataset:** [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad) <!-- TODO: replace with the CREMA-D link used -->

## Experiments

Everything in the paper comes from four experiments. Each was originally its own Kaggle notebook;
each is now a module in `senticdit/experiments/` and runs in its own process.

| experiment | original notebook | what it does | time on 2× T4 |
|---|---|---|---|
| `v5` | `senticdit_v5.ipynb` | MELD data split, model load + weight-loading guard, zero-shot control, ablations (sampling / LoRA rank / AdaLN targets), main LoRA run (resumable), CFG sweep, full held-out evaluation, wav2vec2 SER + real-audio control, paraphrase prompts, DTW MCD + alignment artefact, UTMOS + pitch-drift mechanism, Whisper WER, emotion2vec, WavLM speaker consistency, paper results table | ~2 h with the ablations skipped and the main run resumed |
| `configd` | `senticdit_configd.ipynb` | Config D: MELD train split vs pooled train+dev+test at a fixed 2000-step budget, same evaluation noise draws; plus the MELD F0 register check | ~4.7 h |
| `addon` | `senticdit_addon.ipynb` | Part A: CREMA-D voices cloned from each actor, scored by MCD against the same actor (valid) and a different actor (the MELD situation). Part B: the held-out MELD clips regenerated and scored with second-opinion models (DNSMOS, NISQA, wav2vec2-CTC ASR, ECAPA, both SER models) | ~1.6 h |
| `crossclone` | `senticdit_crossclone.ipynb` | 2×2 cross-speaker cloning on CREMA-D (voice A/B × reference A/B) to separate speaker from recording channel in MCD | ~35 min |

Config D ran last inside the original v5 notebook and ran out of GPU memory there (UTMOS, Whisper
and the SER model were still resident), which is why it is a separate experiment.

## Running on Kaggle

1. Upload [`senticdit_kaggle.ipynb`](senticdit_kaggle.ipynb) to Kaggle. Accelerator **GPU T4 ×2**,
   Internet **on**.
2. Add the inputs:
   - [`zaber666/meld-dataset`](https://www.kaggle.com/datasets/zaber666/meld-dataset) — MELD CSVs
   - [`aryansharma26/meld-audio`](https://www.kaggle.com/datasets/aryansharma26/meld-audio) — pre-extracted MELD WAVs
   - [`aryansharma26/senticdit-r-16`](https://www.kaggle.com/datasets/aryansharma26/senticdit-r-16) — the fine-tuned adapter scored by `addon`
   - [`ejlok1/cremad`](https://www.kaggle.com/datasets/ejlok1/cremad) — CREMA-D, for `addon` and `crossclone`
   - the output of the earlier `aryansharma26/senticdit` notebook — the finished main-run checkpoint that `v5` resumes instead of retraining for ~16 h
3. Check the configuration cell. The values there are the ones the original runs used. `RUN`
   switches experiments on and off, `COMMON` applies to all of them, and each experiment has its
   own dict of overrides. Any field of [`senticdit/config.py`](senticdit/config.py) can be set
   (every field is documented there).
4. Run all. The notebook clones this repo (`REPO_REF`, default `main`), installs dependencies,
   then runs each enabled experiment in turn as
   `python scripts/run_experiment.py <name> --config <output_dir>/config_<name>.json`.

All four together take ~9 h, which fits in one 12 h session. To split them across sessions, set
the others to `False` in `RUN`. Each experiment's log streams into the notebook and is saved to
`<output_dir>/log_<name>.txt`.

**Outputs** (with the default `output_dir=/kaggle/working/audiodit_fixed`):

- `v5` — CSVs, WAVs and figures directly in `output_dir`. The headline table is
  `paper_results_table.csv`; `run_log.csv` accumulates replications (bump `run_tag` and `eval_seed`).
- `configd` — `configd_per_utterance.csv`, `configd_comparison.csv` and `meld_training_f0.csv` in
  `output_dir`; checkpoints in `checkpoints/configd_*`; all zipped to `/kaggle/working/configd_results.zip`.
- `addon` — `output_dir/addon/`, zipped to `/kaggle/working/addon_results.zip` (~150 MB with WAVs).
- `crossclone` — `output_dir/crossclone/`, zipped to `/kaggle/working/crossclone_results.zip`.

**Resuming:** `v5` resumes the main run from `resume_search_dir`. `configd` checkpoints every
500 steps; to continue an interrupted session, add its output as an input and set
`configd_resume_root` to the folder that contains `configd_D_train_only/` and `configd_C_pooled_matched/`.

## Reproducibility

Each evaluation block reseeds from `eval_seed` with a tag (`zeroshot`, `final_eval`,
`configd_eval`, `addon_cremad_cloned`, …), so a block's generation noise does not depend on what
ran before it. That is what lets `addon` regenerate exactly the v5 zero-shot and fine-tuned clips,
and `crossclone` regenerate exactly the add-on's voice-A clones. Running each experiment in a
fresh process keeps everything else (RNG state, GPU memory) the same as a fresh Kaggle session.
CUDA kernels are not bitwise deterministic, so expect agreement to within GPU numerical noise
rather than to the last digit.

## Running locally

```bash
pip install -r requirements.txt
```

Each experiment takes a JSON file of `Config` fields and/or `--set key=value` overrides.
`--install` also clones the LongCat-AudioDiT model code (it is not on PyPI) and installs the
pinned dependencies:

```bash
python scripts/run_experiment.py crossclone --install --set output_dir=outputs --set cremad_dir=data/AudioWAV --set longcat_repo_dir=third_party/LongCat-AudioDiT
```

Synthesise one sentence, optionally with a trained adapter:

```bash
python scripts/demo_inference.py --text "I cannot believe this is happening right now!" --emotion surprise --adapter outputs/lora_final --output surprise.wav
```

Available emotions: neutral, surprise, fear, sadness, joy, disgust, anger.

## Repository structure

```text
SenticDiT/
├── senticdit_kaggle.ipynb     # Kaggle runner: clone repo -> configure -> run experiments
├── senticdit/
│   ├── config.py              # Config dataclass: every setting, with rationale
│   ├── launcher.py            # runs each experiment in its own process (used by the notebook)
│   ├── setup_env.py           # pip installs + clone LongCat-AudioDiT
│   ├── context.py             # shared setup: MELD splits, model load, weight-loading guard
│   ├── experiments/
│   │   ├── v5.py              # main run + full evaluation
│   │   ├── configd.py         # data quantity at fixed compute + MELD register check
│   │   ├── addon.py           # CREMA-D cloned-voice MCD + second-opinion metrics
│   │   └── crossclone.py      # 2x2 cross-speaker cloning
│   ├── data.py                # MELD loading, held-out split, weights, duration model, Dataset
│   ├── model.py               # model loading, weight-loading guard, LoRA helpers
│   ├── training.py            # flow-matching loss, checkpoints/resume, training loop
│   ├── generation.py          # generate_speech, generate_and_score, prompt formats
│   ├── cremad.py              # CREMA-D discovery, target selection, voice cloning
│   ├── metrics.py             # generation-health metrics (completion, F0)
│   ├── stats.py               # bootstrap / paired bootstrap / Wilson / MDE
│   ├── ablations.py           # sampling, rank, target-module ablations
│   ├── report.py              # paper results table, cross-run log, manifest
│   └── evaluation/            # health, SER (wav2vec2, emotion2vec), MCD, UTMOS, WER, speaker
├── scripts/
│   ├── run_experiment.py      # run one experiment
│   ├── demo_inference.py      # synthesise one sentence
│   └── meld_eda.py            # dataset EDA figures
├── plots/                     # EDA figures and figures from the earlier v1 pipeline
├── Papers.zip                 # referenced papers
└── requirements.txt
```
