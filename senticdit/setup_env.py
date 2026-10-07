"""Environment bootstrap for Kaggle.

Must run before anything imports torch/transformers: it upgrades transformers in place and puts
the LongCat-AudioDiT source (the `audiodit` package, which is not on PyPI) on sys.path. This
module deliberately imports nothing heavy.
"""
import os
import subprocess
import sys

LONGCAT_REPO_URL = "https://github.com/meituan-longcat/LongCat-AudioDiT.git"
LONGCAT_REPO_DIR = "/kaggle/working/LongCat-AudioDiT"


def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + list(args), check=True)


def install_dependencies():
    # Kaggle images ship an old torchao that makes `peft`'s LoRA dispatcher raise ImportError
    # even though we never use torchao at all (peft checks its version unconditionally).
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

    pip("-U", "transformers>=5.3.0")
    pip("torchaudio>=2.0.0", "safetensors>=0.4.0", "librosa>=0.10.0", "soundfile>=0.12.0",
        "einops>=0.8.0", "sentencepiece", "peft>=0.11.0", "accelerate>=0.30.0")

    # Evaluation dependencies:
    #   pymcd/fastdtw -> DTW-aligned MCD
    #   pyworld       -> second F0 estimator, used to catch pyin octave errors
    #   scipy         -> normal quantiles for the minimum-detectable-effect calculation
    #   jiwer         -> WER
    # funasr (emotion2vec) is installed lazily by evaluation.ser with check=False, so a failure
    # there cannot abort the whole run.
    pip("pymcd", "fastdtw", "pyworld", "scipy", "jiwer")


def fetch_longcat(repo_dir=LONGCAT_REPO_DIR, repo_url=LONGCAT_REPO_URL):
    if not os.path.isdir(repo_dir):
        subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_dir], check=True)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def prepare_environment(install=True, longcat_dir=LONGCAT_REPO_DIR, longcat_url=LONGCAT_REPO_URL):
    """Install dependencies and fetch the real AudioDiT model code. Idempotent."""
    if install:
        install_dependencies()
    fetch_longcat(longcat_dir, longcat_url)
    print("Ready.")
