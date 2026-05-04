import os, csv, math, time, warnings, random, json, glob, subprocess, sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ── third-party ───────────────────────────────────────────────────────────────
# Install missing packages quietly (Kaggle may already have most of these)
def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                    *pkgs], check=False)

try:
    import soundfile as sf
except ImportError:
    _pip("soundfile"); import soundfile as sf

try:
    import librosa
except ImportError:
    _pip("librosa"); import librosa

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────────────────
MELD_ROOT  = Path("/kaggle/input/datasets/aryansharma2603/meld-dataset")
OUTPUT_DIR = Path("/kaggle/working/tts_dit_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 22050      # Hz  – keep at 22 050 to match most MELD files
N_FFT             = 1024
HOP_LENGTH        = 256
N_MELS            = 80
MAX_AUDIO_SECS    = 6          # seconds – MELD utterances rarely exceed 6 s
MAX_FRAMES        = math.ceil(MAX_AUDIO_SECS * SAMPLE_RATE / HOP_LENGTH) + 1

# ── Model ─────────────────────────────────────────────────────────────────────
TEXT_EMB_DIM      = 256        # output dim of the text encoder
DIT_DIM           = 512        # hidden dim inside every DiT block
DIT_DEPTH         = 8          # number of DiT blocks
DIT_HEADS         = 8          # attention heads per block
DIT_FF_MULT       = 4          # feed-forward expansion factor
TEXT_MAX_LEN      = 256        # max character sequence length

# ── Diffusion ─────────────────────────────────────────────────────────────────
DIFFUSION_STEPS   = 1000       # T in the forward process
INFERENCE_STEPS   = 50         # DDIM steps at generation time
BETA_START        = 1e-4
BETA_END          = 0.02

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE        = 16         # increase if GPU VRAM allows
LR                = 2e-4
WEIGHT_DECAY      = 1e-4
EPOCHS            = 100         # 30 epochs gives solid results on MELD
WARMUP_STEPS      = 500
MAX_TRAIN_SAMPLES = 9989       # full MELD train split
MAX_VAL_SAMPLES   = 1109       # full MELD dev split
SEED              = 42
GRAD_CLIP         = 1.0

# ── Runtime ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4 if DEVICE == "cuda" else 0

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

print(f"[Config] device={DEVICE} | PyTorch={torch.__version__}")
print(f"[Config] MELD root: {MELD_ROOT}")
print(f"[Config] Output dir: {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MELD DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _discover_meld_layout(root: Path, split: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Tries several common MELD folder layouts and returns (csv_path, audio_dir).
    Handles flat layout, nested by split, and common alternative naming.
    """
    split_aliases = {
        "train": ["train", "train_sent_emo"],
        "dev":   ["dev",   "dev_sent_emo"],
        "test":  ["test",  "test_sent_emo"],
    }
    csv_candidates = []
    for alias in split_aliases.get(split, [split]):
        csv_candidates += [
            root / f"{alias}_sent_emo.csv",
            root / f"{alias}.csv",
            root / split / f"{alias}_sent_emo.csv",
            root / split / f"{alias}.csv",
            root / "MELD.Raw" / f"{alias}_sent_emo.csv",
        ]

    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        # Last resort: glob
        matches = list(root.rglob(f"*{split}*.csv"))
        csv_path = matches[0] if matches else None

    # Audio folder: look for 'audio', split-named subfolder, or raw wav dirs
    audio_candidates = [
        root / "audio",
        root / split,
        root / f"{split}_audio",
        root / "MELD.Raw" / split,
        root,
    ]
    audio_dir = next((p for p in audio_candidates
                      if p.is_dir() and any(p.glob("*.wav"))), None)
    if audio_dir is None:
        # Try one level deeper
        for d in root.rglob("*.wav"):
            audio_dir = d.parent
            break

    return csv_path, audio_dir


def _wav_path_variants(audio_dir: Path, dia: str, utt: str) -> List[Path]:
    """Return candidate WAV filenames for a given dialogue/utterance pair."""
    return [
        audio_dir / f"dia{dia}_utt{utt}.wav",
        audio_dir / f"dia{dia}utt{utt}.wav",
        audio_dir / f"dia_{dia}_utt_{utt}.wav",
        audio_dir / f"{dia}_{utt}.wav",
        audio_dir / f"dia{dia}" / f"utt{utt}.wav",
    ]


class MeldDataset(Dataset):
    """
    Loads MELD utterances (text + audio) from the Kaggle dataset path.

    Each sample returns:
        text_ids  : LongTensor (TEXT_MAX_LEN,)   – character IDs
        mel       : FloatTensor (N_MELS, T)      – log-mel spectrogram
        wav       : FloatTensor (max_audio_len,) – raw waveform (for eval)
        text      : str                          – original utterance
        emotion   : str                          – MELD emotion label
    """

    MEL_TF = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS)

    def __init__(self, root: Path, split: str = "train",
                 max_samples: int = MAX_TRAIN_SAMPLES):
        self.max_wav_len = int(MAX_AUDIO_SECS * SAMPLE_RATE)
        self.samples: List[Dict] = []

        csv_path, audio_dir = _discover_meld_layout(root, split)

        if csv_path is None:
            raise FileNotFoundError(
                f"Could not locate {split} CSV in {root}. "
                f"Please verify the dataset was downloaded correctly.")

        print(f"[Dataset/{split}] CSV   → {csv_path}")
        print(f"[Dataset/{split}] Audio → {audio_dir}")

        n_found = n_missing = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text    = row.get("Utterance", "").strip().strip('"')
                dia     = row.get("Dialogue_ID", "").strip()
                utt     = row.get("Utterance_ID", "").strip()
                emotion = row.get("Emotion", "neutral").strip().lower()
                if not text:
                    continue

                wav_path = None
                if audio_dir:
                    for candidate in _wav_path_variants(audio_dir, dia, utt):
                        if candidate.exists():
                            wav_path = candidate
                            n_found += 1
                            break
                    else:
                        n_missing += 1
                else:
                    n_missing += 1

                self.samples.append(dict(
                    text=text, wav_path=wav_path, emotion=emotion,
                    dia=dia, utt=utt))
                if len(self.samples) >= max_samples:
                    break

        print(f"[Dataset/{split}] {len(self.samples)} utterances loaded  "
              f"({n_found} with audio, {n_missing} text-only)")
        if n_found == 0:
            print(f"[Dataset/{split}] ⚠️  No WAV files found – "
                  f"check that audio/ directory is present in {root}")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load_wav(self, path: Optional[Path]) -> torch.Tensor:
        if path is not None:
            try:
                wav, sr = torchaudio.load(str(path))
                if sr != SAMPLE_RATE:
                    wav = T.Resample(sr, SAMPLE_RATE)(wav)
                wav = wav.mean(0)          # stereo → mono
            except Exception as e:
                print(f"[warn] Could not load {path}: {e}")
                wav = self._silence()
        else:
            wav = self._silence()

        # Pad or crop to fixed length
        if wav.shape[-1] < self.max_wav_len:
            wav = F.pad(wav, (0, self.max_wav_len - wav.shape[-1]))
        else:
            wav = wav[..., :self.max_wav_len]
        return wav.float()

    @staticmethod
    def _silence() -> torch.Tensor:
        """Return near-zero audio when a WAV file is unavailable."""
        return torch.zeros(int(MAX_AUDIO_SECS * SAMPLE_RATE))

    @staticmethod
    def _to_mel(wav: torch.Tensor) -> torch.Tensor:
        mel = MeldDataset.MEL_TF(wav.unsqueeze(0)).squeeze(0)  # (N_MELS, T)
        return torch.log(mel.clamp(min=1e-9))

    @staticmethod
    def text_to_ids(text: str) -> torch.Tensor:
        ids = [min(ord(c), 255) for c in text[:TEXT_MAX_LEN]]
        ids += [0] * (TEXT_MAX_LEN - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s   = self.samples[idx]
        wav = self._load_wav(s["wav_path"])
        mel = self._to_mel(wav)

        # Pad / crop mel to MAX_FRAMES
        if mel.shape[-1] < MAX_FRAMES:
            mel = F.pad(mel, (0, MAX_FRAMES - mel.shape[-1]))
        else:
            mel = mel[..., :MAX_FRAMES]

        return dict(
            text_ids = self.text_to_ids(s["text"]),
            mel      = mel,
            wav      = wav,
            text     = s["text"],
            emotion  = s["emotion"],
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

# ── 2a. Character-level Text Encoder ─────────────────────────────────────────

class CharTextEncoder(nn.Module):
    """
    Encodes a character-ID sequence into a single fixed-size conditioning
    vector via a small Transformer encoder + mean pooling.
    """

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(257, TEXT_EMB_DIM, padding_idx=0)
        self.pos = nn.Embedding(TEXT_MAX_LEN, TEXT_EMB_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=TEXT_EMB_DIM, nhead=4,
            dim_feedforward=TEXT_EMB_DIM * 4,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.pool    = nn.AdaptiveAvgPool1d(1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, L)  →  (B, TEXT_EMB_DIM)"""
        B, L  = ids.shape
        pos   = torch.arange(L, device=ids.device).unsqueeze(0)
        x     = self.emb(ids) + self.pos(pos)
        mask  = (ids == 0)
        x     = self.encoder(x, src_key_padding_mask=mask)   # (B, L, D)
        return self.pool(x.transpose(1, 2)).squeeze(-1)       # (B, D)


# ── 2b. Sinusoidal Timestep Embedding ────────────────────────────────────────

class TimestepEmb(nn.Module):
    def __init__(self, dim: int = DIT_DIM):
        super().__init__()
        self.dim  = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freq  = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half)
        args  = t.float()[:, None] * freq[None]
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


# ── 2c. DiT Block with AdaLN ─────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    Diffusion Transformer block with Adaptive Layer Norm (AdaLN) conditioning.
    The conditioning signal (timestep + text) modulates scale, shift and gate
    for both the self-attention and feed-forward sub-layers.
    """

    def __init__(self):
        super().__init__()
        self.norm1  = nn.LayerNorm(DIT_DIM, elementwise_affine=False, eps=1e-6)
        self.norm2  = nn.LayerNorm(DIT_DIM, elementwise_affine=False, eps=1e-6)
        self.attn   = nn.MultiheadAttention(
            DIT_DIM, DIT_HEADS, dropout=0.0, batch_first=True)
        self.ff     = nn.Sequential(
            nn.Linear(DIT_DIM, DIT_DIM * DIT_FF_MULT),
            nn.GELU(),
            nn.Linear(DIT_DIM * DIT_FF_MULT, DIT_DIM))
        # AdaLN modulation: 6 * DIT_DIM params (shift1, scale1, gate1, ×2)
        self.adaLN  = nn.Sequential(
            nn.SiLU(), nn.Linear(DIT_DIM, 6 * DIT_DIM, bias=True))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, DIT_DIM) – time + text conditioning
        shift1, scale1, gate1, shift2, scale2, gate2 = \
            self.adaLN(cond).unsqueeze(1).chunk(6, dim=-1)

        xn = self.norm1(x) * (1 + scale1) + shift1
        ao, _ = self.attn(xn, xn, xn)
        x = x + gate1 * ao

        xn = self.norm2(x) * (1 + scale2) + shift2
        x  = x + gate2 * self.ff(xn)
        return x


# ── 2d. Full Diffusion Transformer ───────────────────────────────────────────

class DiffusionTransformer(nn.Module):
    """
    Predicts the noise ε in a noisy log-mel spectrogram given:
        mel_noisy : (B, N_MELS, T)   – noisy mel
        t         : (B,)             – diffusion timestep (long)
        text_emb  : (B, TEXT_EMB_DIM)
    Returns:
        noise_pred : (B, N_MELS, T)
    """

    def __init__(self):
        super().__init__()
        # Patch-in: project each mel frame to DIT_DIM
        self.patch_in  = nn.Linear(N_MELS, DIT_DIM)
        self.pos_emb   = nn.Embedding(MAX_FRAMES + 4, DIT_DIM)

        # Conditioning: time + text → single DIT_DIM vector
        self.time_emb  = TimestepEmb(DIT_DIM)
        self.cond_proj = nn.Sequential(
            nn.Linear(DIT_DIM + TEXT_EMB_DIM, DIT_DIM * 2),
            nn.SiLU(),
            nn.Linear(DIT_DIM * 2, DIT_DIM))

        # DiT blocks
        self.blocks    = nn.ModuleList([DiTBlock() for _ in range(DIT_DEPTH)])

        # Patch-out: project back to mel
        self.norm_out  = nn.LayerNorm(DIT_DIM, eps=1e-6)
        self.patch_out = nn.Linear(DIT_DIM, N_MELS)
        nn.init.zeros_(self.patch_out.weight)
        nn.init.zeros_(self.patch_out.bias)

    def forward(self, mel_noisy: torch.Tensor,
                t: torch.Tensor,
                text_emb: torch.Tensor) -> torch.Tensor:
        B, M, T = mel_noisy.shape
        x  = mel_noisy.permute(0, 2, 1)            # (B, T, M)
        x  = self.patch_in(x)                       # (B, T, D)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x  = x + self.pos_emb(pos)

        t_emb = self.time_emb(t)                    # (B, D)
        cond  = self.cond_proj(
            torch.cat([t_emb, text_emb], dim=-1))   # (B, D)

        for block in self.blocks:
            x = block(x, cond)

        x = self.norm_out(x)
        x = self.patch_out(x)                       # (B, T, M)
        return x.permute(0, 2, 1)                   # (B, M, T)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DDPM NOISE SCHEDULER + DDIM SAMPLER
# ══════════════════════════════════════════════════════════════════════════════

class DDPMScheduler:

    def __init__(self):
        betas        = torch.linspace(BETA_START, BETA_END, DIFFUSION_STEPS)
        alphas       = 1.0 - betas
        alpha_bar    = torch.cumprod(alphas, dim=0)
        self.T       = DIFFUSION_STEPS
        self._bufs   = dict(
            betas      = betas,
            alphas     = alphas,
            alpha_bar  = alpha_bar,
            sqrt_ab    = alpha_bar.sqrt(),
            sqrt_1m_ab = (1.0 - alpha_bar).sqrt(),
        )

    def to(self, device: str) -> "DDPMScheduler":
        self._bufs = {k: v.to(device) for k, v in self._bufs.items()}
        return self

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._bufs[name]
        except KeyError:
            raise AttributeError(name)

    # ── forward process (training) ────────────────────────────────────────────
    def add_noise(self, x0: torch.Tensor,
                  t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        s_ab  = self.sqrt_ab[t].view(-1, 1, 1)
        s_1m  = self.sqrt_1m_ab[t].view(-1, 1, 1)
        return s_ab * x0 + s_1m * noise, noise

    # ── DDIM deterministic sampling (inference) ───────────────────────────────
    @torch.no_grad()
    def ddim_sample(self, dit: nn.Module, text_emb: torch.Tensor,
                    shape: Tuple, device: str,
                    n_steps: int = INFERENCE_STEPS,
                    eta: float = 0.0) -> torch.Tensor:
        """
        eta=0   → deterministic DDIM
        eta>0   → stochastic (DDPM-like)
        """
        dit.eval()
        xt    = torch.randn(shape, device=device)
        steps = torch.linspace(self.T - 1, 0, n_steps, dtype=torch.long)

        for i, step in enumerate(steps):
            t_b   = torch.full((shape[0],), step.item(),
                               device=device, dtype=torch.long)
            eps   = dit(xt, t_b, text_emb)
            ab    = self.alpha_bar[step].to(device)
            x0_p  = (xt - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-8)
            x0_p  = x0_p.clamp(-4, 4)

            if i < len(steps) - 1:
                ab_prev = self.alpha_bar[steps[i + 1]].to(device)
                sigma   = eta * ((1 - ab_prev) / (1 - ab) *
                                 (1 - ab / ab_prev)).sqrt()
                xt = (ab_prev.sqrt() * x0_p
                      + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
                      + sigma * torch.randn_like(xt))
            else:
                xt = x0_p
        return xt


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders():
    train_ds = MeldDataset(MELD_ROOT, split="train",
                           max_samples=MAX_TRAIN_SAMPLES)
    val_ds   = MeldDataset(MELD_ROOT, split="dev",
                           max_samples=MAX_VAL_SAMPLES)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
                          drop_last=True, prefetch_factor=2 if NUM_WORKERS else None)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
                          drop_last=False, prefetch_factor=2 if NUM_WORKERS else None)
    return train_dl, val_dl, train_ds


def build_models():
    text_enc  = CharTextEncoder().to(DEVICE)
    dit       = DiffusionTransformer().to(DEVICE)
    scheduler = DDPMScheduler().to(DEVICE)
    n_params  = sum(p.numel() for p in
                    list(text_enc.parameters()) + list(dit.parameters())
                    if p.requires_grad)
    print(f"[Model] Trainable params: {n_params:,}  "
          f"({n_params/1e6:.1f} M)")
    return text_enc, dit, scheduler


def train():
    print("\n" + "="*66)
    print("  TEXT-TO-AUDIO DIFFUSION TRANSFORMER — MELD FINE-TUNING")
    print("="*66)

    train_dl, val_dl, train_ds = build_dataloaders()
    text_enc, dit, sched       = build_models()

    params    = list(text_enc.parameters()) + list(dit.parameters())
    optim     = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_dl)
    lr_sched  = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=LR, total_steps=total_steps,
        pct_start=WARMUP_STEPS / total_steps,
        anneal_strategy="cos")

    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    history   = []
    best_val  = float("inf")
    best_ckpt = OUTPUT_DIR / "dit_tts_meld_best.pt"

    for epoch in range(1, EPOCHS + 1):
        # ── train ─────────────────────────────────────────────────────────────
        text_enc.train(); dit.train()
        t_loss = 0.0
        pbar   = tqdm(train_dl, desc=f"Epoch {epoch:3d}/{EPOCHS}", leave=False)
        for batch in pbar:
            ids  = batch["text_ids"].to(DEVICE)
            mel  = batch["mel"].to(DEVICE)        # (B, N_MELS, T)

            B    = mel.shape[0]
            t    = torch.randint(0, sched.T, (B,), device=DEVICE)
            xt, noise = sched.add_noise(mel, t)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                text_emb = text_enc(ids)
                pred     = dit(xt, t, text_emb)
                loss     = F.mse_loss(pred, noise)

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            scaler.step(optim)
            scaler.update()
            lr_sched.step()

            t_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{lr_sched.get_last_lr()[0]:.2e}")

        avg_train = t_loss / len(train_dl)

        # ── validate ──────────────────────────────────────────────────────────
        text_enc.eval(); dit.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                ids  = batch["text_ids"].to(DEVICE)
                mel  = batch["mel"].to(DEVICE)
                B    = mel.shape[0]
                t    = torch.randint(0, sched.T, (B,), device=DEVICE)
                xt, noise = sched.add_noise(mel, t)
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    text_emb = text_enc(ids)
                    pred     = dit(xt, t, text_emb)
                    v_loss  += F.mse_loss(pred, noise).item()
        avg_val = v_loss / max(len(val_dl), 1)

        print(f"  Epoch {epoch:3d} | "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"lr={lr_sched.get_last_lr()[0]:.2e}")
        history.append(dict(epoch=epoch,
                            train_loss=avg_train, val_loss=avg_val))

        # Save best checkpoint
        if avg_val < best_val:
            best_val = avg_val
            torch.save(dict(text_enc=text_enc.state_dict(),
                            dit=dit.state_dict()), best_ckpt)

    # Final checkpoint
    final_ckpt = OUTPUT_DIR / "dit_tts_meld.pt"
    torch.save(dict(text_enc=text_enc.state_dict(),
                    dit=dit.state_dict(),
                    history=history), final_ckpt)
    print(f"\n[Checkpoint] Final  → {final_ckpt}")
    print(f"[Checkpoint] Best   → {best_ckpt}  (val={best_val:.4f})")
    return text_enc, dit, sched, train_ds, history


# ══════════════════════════════════════════════════════════════════════════════
# 5.  INFERENCE — GENERATE 6 SAMPLE WAVs
# ══════════════════════════════════════════════════════════════════════════════

# Six diverse MELD-style sample sentences covering different emotions
SAMPLE_TEXTS = [
    "I really can't believe what just happened to us.",          # surprise
    "Could you please pass me the salt over there?",             # neutral
    "She smiled and said everything was going to be alright.",   # joy
    "We are running out of time, please hurry up now!",          # fear/urgency
    "He was absolutely furious at the decision they made.",      # anger
    "Today has honestly been the best day of my entire life.",   # joy
]


def mel_to_wav(mel: torch.Tensor) -> np.ndarray:
    """
    Convert a log-mel spectrogram back to a waveform using Griffin-Lim.
    For production quality, replace with a HiFi-GAN or BigVGAN vocoder.
    """
    mel_np  = mel.squeeze().cpu().float().numpy()   # (N_MELS, T)
    mel_lin = np.exp(mel_np).astype(np.float32)

    wav = librosa.feature.inverse.mel_to_audio(
        mel_lin, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_iter=64)

    peak = np.abs(wav).max()
    if peak > 1e-6:
        wav = wav / peak * 0.9
    return wav.astype(np.float32)


def generate_samples(text_enc: nn.Module,
                     dit: nn.Module,
                     sched: DDPMScheduler) -> List[str]:
    print("\n[Inference] Generating 6 sample WAV files …")
    text_enc.eval(); dit.eval()
    helper   = MeldDataset.__new__(MeldDataset)  # avoid re-loading data
    wav_paths = []

    for i, text in enumerate(SAMPLE_TEXTS, 1):
        ids      = MeldDataset.text_to_ids(text).unsqueeze(0).to(DEVICE)
        text_emb = text_enc(ids)

        shape = (1, N_MELS, MAX_FRAMES)
        mel   = sched.ddim_sample(dit, text_emb, shape, DEVICE)

        wav   = mel_to_wav(mel[0])
        fpath = OUTPUT_DIR / f"sample_{i:02d}.wav"
        sf.write(str(fpath), wav, SAMPLE_RATE)
        wav_paths.append(str(fpath))
        print(f"  [{i}] '{text[:60]}' → {fpath.name}")

    return wav_paths


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION — MCD + FAD
# ══════════════════════════════════════════════════════════════════════════════

def compute_mcd(ref: np.ndarray, gen: np.ndarray) -> float:
    """
    Mel-Cepstral Distortion (MCD) in dB.
      MCD = (10/ln10) * sqrt(2 * Σ (mc_ref_k − mc_gen_k)²)
    Reference: Kubichek (1993).  Lower is better; typical good range: 4–8 dB.
    """
    n = min(len(ref), len(gen))
    ref, gen = ref[:n], gen[:n]

    n_mfcc = 25
    mc_ref = librosa.feature.mfcc(y=ref, sr=SAMPLE_RATE,
                                   n_mfcc=n_mfcc + 1)[1:]   # skip c0
    mc_gen = librosa.feature.mfcc(y=gen, sr=SAMPLE_RATE,
                                   n_mfcc=n_mfcc + 1)[1:]

    nf = min(mc_ref.shape[1], mc_gen.shape[1])
    diff = mc_ref[:, :nf] - mc_gen[:, :nf]
    return float((10.0 / np.log(10)) *
                 np.sqrt(2.0 * np.mean(np.sum(diff ** 2, axis=0))))


def compute_fad(real_wavs: List[np.ndarray],
                gen_wavs:  List[np.ndarray]) -> float:
    """
    Fréchet Audio Distance (approximation) using log-mel statistics.

    True FAD uses VGGish embeddings; this implementation uses 256-dim
    log-mel mean+std vectors as a proxy — same Fréchet formula, no
    external model weights required.

    FAD = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2·sqrtm(Σ_r Σ_g))
    Lower is better; good models typically score < 50 with this proxy.
    """
    def featurise(wavs: List[np.ndarray]) -> np.ndarray:
        out = []
        for w in wavs:
            m  = librosa.feature.melspectrogram(
                y=w, sr=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=128)
            lm = librosa.power_to_db(m + 1e-9)
            out.append(np.concatenate([lm.mean(1), lm.std(1)]))
        return np.array(out, dtype=np.float64)   # (N, 256)

    rf = featurise(real_wavs)
    gf = featurise(gen_wavs)

    mu_r, mu_g = rf.mean(0), gf.mean(0)
    Sr = np.cov(rf, rowvar=False) if len(rf) > 1 else np.eye(rf.shape[1])
    Sg = np.cov(gf, rowvar=False) if len(gf) > 1 else np.eye(gf.shape[1])

    diff    = mu_r - mu_g
    sq_dist = float(diff @ diff)

    # Matrix square-root via eigendecomposition (stable for semi-PD matrices)
    vals, vecs = np.linalg.eigh(Sr @ Sg)
    vals = np.maximum(vals, 0.0)
    sqrtm = vecs @ np.diag(np.sqrt(vals)) @ vecs.T

    tr = float(np.trace(Sr) + np.trace(Sg) - 2.0 * np.trace(sqrtm))
    return float(sq_dist + tr)


def evaluate(wav_paths: List[str],
             dataset: MeldDataset) -> Dict:
    print("\n[Evaluation] Computing MCD and FAD …")

    gen_wavs = [sf.read(p)[0].astype(np.float32) for p in wav_paths]

    # Use the first N real utterances as references
    ref_wavs: List[np.ndarray] = []
    for i in range(min(len(SAMPLE_TEXTS) * 3, len(dataset))):
        ref_wavs.append(dataset[i]["wav"].numpy().astype(np.float32))
        if len(ref_wavs) == len(gen_wavs):
            break
    # Pad if fewer real samples than generated
    while len(ref_wavs) < len(gen_wavs):
        ref_wavs.append(ref_wavs[0])

    # Per-sample MCD
    mcd_scores = []
    for i, (ref, gen) in enumerate(zip(ref_wavs, gen_wavs)):
        score = compute_mcd(ref, gen)
        mcd_scores.append(score)
        print(f"  Sample {i+1:2d}: MCD = {score:7.2f} dB  |  "
              f"'{SAMPLE_TEXTS[i][:50]}'")

    mean_mcd = float(np.mean(mcd_scores))
    fad      = compute_fad(ref_wavs, gen_wavs)

    print(f"\n  ┌{'─'*40}")
    print(f"  │  Mean MCD : {mean_mcd:.2f} dB   "
          f"(lower ↓ is better; target < 8 dB)")
    print(f"  │  FAD      : {fad:.4f}       "
          f"(lower ↓ is better; target < 50)")
    print(f"  └{'─'*40}")

    return dict(mcd_per_sample=mcd_scores, mean_mcd=mean_mcd, fad=fad)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  REPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_report(history: List[Dict],
                eval_res: Dict,
                wav_paths: List[str]) -> Path:
    cfg_snapshot = dict(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, max_audio_secs=MAX_AUDIO_SECS,
        text_emb_dim=TEXT_EMB_DIM, dit_dim=DIT_DIM,
        dit_depth=DIT_DEPTH, dit_heads=DIT_HEADS,
        diffusion_steps=DIFFUSION_STEPS, inference_steps=INFERENCE_STEPS,
        batch_size=BATCH_SIZE, lr=LR, epochs=EPOCHS,
        device=DEVICE,
    )
    report = dict(
        model="Diffusion Transformer (DiT) TTS fine-tuned on MELD",
        dataset=str(MELD_ROOT),
        config=cfg_snapshot,
        training_history=history,
        evaluation=eval_res,
        generated_samples=[str(p) for p in wav_paths],
    )
    json_path = OUTPUT_DIR / "training_report.json"
    json_path.write_text(json.dumps(report, indent=2))

    txt_path = OUTPUT_DIR / "summary.txt"
    W = 64
    with open(txt_path, "w") as f:
        def w(s=""): f.write(s + "\n")
        w("=" * W)
        w("  TEXT-TO-AUDIO DiT  —  TRAINING & EVALUATION SUMMARY")
        w("=" * W)
        w()
        w(f"  Architecture  : Diffusion Transformer (DiT)")
        w(f"  Text encoder  : Character-level Transformer (4 layers)")
        w(f"  DiT blocks    : {DIT_DEPTH} × AdaLN-conditioned blocks")
        w(f"  Parameters    : ~{sum(p.numel() for p in [])//1000}K  "
          f"(see model definition)")
        w(f"  Dataset       : MELD  ({MELD_ROOT})")
        w(f"  Device        : {DEVICE}")
        w(f"  Epochs        : {EPOCHS}")
        w(f"  Batch size    : {BATCH_SIZE}")
        w(f"  Learning rate : {LR}")
        w()
        w("-" * W)
        w("  TRAINING HISTORY")
        w("-" * W)
        w(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}")
        w(f"  {'─'*6}  {'─'*12}  {'─'*12}")
        for h in history:
            w(f"  {h['epoch']:6d}  {h['train_loss']:12.4f}  {h['val_loss']:12.4f}")
        w()
        w("-" * W)
        w("  EVALUATION METRICS  (on 6 generated samples vs real MELD)")
        w("-" * W)
        w(f"  Mean MCD  : {eval_res['mean_mcd']:.2f} dB  "
          f"(target < 8 dB after full training)")
        w(f"  FAD       : {eval_res['fad']:.4f}        "
          f"(target < 50 with log-mel proxy)")
        w()
        w(f"  {'Sample':>8}  {'MCD (dB)':>10}  Utterance")
        w(f"  {'─'*8}  {'─'*10}  {'─'*42}")
        for i, (mcd, text) in enumerate(zip(eval_res["mcd_per_sample"],
                                             SAMPLE_TEXTS), 1):
            w(f"  {i:8d}  {mcd:10.2f}  {text[:42]}")
        w()
        w("-" * W)
        w("  GENERATED SAMPLES")
        w("-" * W)
        for i, (text, p) in enumerate(zip(SAMPLE_TEXTS, wav_paths), 1):
            w(f"  {i}. {text}")
            w(f"     → {Path(p).name}")
            w()
        w("=" * W)

    print(f"[Report] summary.txt       → {txt_path}")
    print(f"[Report] training_report.json → {json_path}")
    return txt_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    # 1. Train
    text_enc, dit, sched, train_ds, history = train()

    # 2. Generate 6 sample WAV files
    wav_paths = generate_samples(text_enc, dit, sched)

    # 3. Compute MCD + FAD
    eval_res = evaluate(wav_paths, train_ds)

    # 4. Save report
    save_report(history, eval_res, wav_paths)

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n✅  Pipeline complete in {mins}m {secs}s")
    print(f"   Output dir : {OUTPUT_DIR}")
    print(f"   Checkpoint : dit_tts_meld.pt")
    print(f"   Best ckpt  : dit_tts_meld_best.pt")
    print(f"   WAV files  : {len(wav_paths)}")
    print(f"   Report     : summary.txt")