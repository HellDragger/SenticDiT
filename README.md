# SenticDiT 

This repository contains the codebase for fine-tuning the `meituan-longcat/LongCat-AudioDiT-1B` model to generate emotion-conditioned speech. By utilizing Low-Rank Adaptation (LoRA) and prefixing text prompts with emotion tokens, the model is trained to synthesize distinct emotional deliveries based on the Multimodal EmotionLines Dataset (MELD).

## Features

* **Emotion-Conditioned Generation:** Prefix text with emotion tokens to guide the audio synthesis.
* **LoRA Fine-tuning:** Efficient parameter-efficient fine-tuning on consumer hardware using `peft`.
* **Custom DiT Implementation:** Includes a from-scratch implementation of the Diffusion Transformer (DiT) architecture for educational and experimental purposes.

## Pipeline Overview

1. **Audio Processing**: Extract audio from MELD `.mp4` files, resample to 24 kHz mono, and convert to Mel-Spectrograms.
2. **Text Conditioning**: Prepend emotion tokens (e.g., `[JOY] Hello!`) and process via a frozen `google/flan-t5-small` text encoder.
3. **Model Architecture**: Diffusers-based AudioDiT UNet equipped with LoRA adapters (fp16 + gradient checkpointing).
4. **Vocoding**: Invert the generated Mel-spectrogram latents back to audio waveforms using Griffin-Lim.

## Repository Structure

```text
SenticDiT/
├── plots/                    # Plots
├── data/                     # MELD dataset (ignored in git)
├── src/
│   ├── config.py             # Global CFG setup
│   ├── data_loader.py        # MELDDataset and audio processing
│   ├── models.py             # Model wrappers and LoRA setup
│   ├── custom_dit.py         # Custom from-scratch DiT implementation
│   ├── trainer.py            # Diffusion loss and training loop
│   └── inference.py          # Generation and vocoder logic
├── main.py                   # Entry point for training
├── demo_inference.py         # Script to run generations
├── meld_eda.py               # Script to run eda on dataset
├── loss_plots.py             # Script to generate loss plot based on logs
├── Report.pdf
├── Papers.zip                # The Research papers that i have refered
├── AudioDiT-final.ipynb      # Actual implementation file that is run on Kaggle
├── requirements.txt          # Python dependencies
└── README.md
```

## Data
Please download/view the dataset from Kaggle and place them in the folder called data
MELD Dataset - https://www.kaggle.com/datasets/zaber666/meld-dataset

MELD Audio - https://www.kaggle.com/datasets/aryansharma26/meld-audio

Checkpoints - https://www.kaggle.com/datasets/aryansharma26/checkpoints

## Installation
Ensure you have Python 3.10+ installed, then run:

```Bash
pip install -r requirements.txt
Note: torchao is intentionally omitted to avoid PyTorch version conflicts on specific hardware like Kaggle T4 GPUs.
```

## Usage
### Training

To begin training, ensure your data paths in ```src/config.py``` point to the correct MELD dataset locations, then run:

```Bash
accelerate launch train.py
```
(Alternatively, you can just run ```python train.py``` if testing on a single GPU).

### Inference

You can generate speech from the command line using the ```demo_inference.py``` script. Point it to your saved LoRA checkpoint:

```Bash
python demo_inference.py \
    --text "I cannot believe this is happening right now!" \
    --emotion surprise \
    --checkpoint ./audiodit_meld_output/lora_weights \
    --output surprise_test.wav
Available emotions: neutral, surprise, fear, sadness, joy, disgust, anger.
```
## Evaluation
Model performance is evaluated using Mel Cepstral Distortion (MCD) against the ground-truth MELD audio using the pymcd library. See the notebook/training logs for detailed emotion-by-emotion MCD scores.