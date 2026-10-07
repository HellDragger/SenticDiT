"""Conditional flow-matching LoRA training with full, multi-session-resumable checkpoints.

Each checkpoint saves the full training state — LoRA adapter, optimizer (Adam momentum),
LR-scheduler state, step count, loss history and RNG state — so resuming genuinely continues
training rather than restarting it with an LR/momentum reset.

Multi-session workflow on Kaggle:
1. Run the notebook, let it checkpoint as far as it gets (checkpoints land in
   `<output_dir>/checkpoints/<config_tag>/`), then Save Version.
2. In a new session, Add Data -> the previous notebook's output.
3. Set `resume_search_dir` to that checkpoint path and re-run.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup


def flow_matching_loss(model, tokenizer, batch, cfg, device):
    """Rectified-flow objective (x1 = real VAE latent, x0 = noise, predict x1 - x0), counted only
    over frames within each sample's real (unpadded) length."""
    texts = batch["prompt"]
    wavs = batch["wav"].to(device)
    real_frame_lens = batch["real_frame_lens"].to(device)

    enc = tokenizer(texts, padding="longest", truncation=True, max_length=64,
                     return_tensors="pt").to(device)

    with torch.no_grad():
        text_condition = model.encode_text(enc.input_ids, enc.attention_mask)
        x1 = model.vae.encode(wavs)
    x1 = x1.permute(0, 2, 1)   # (B, T, latent_dim)
    B, T, D = x1.shape

    text_mask = enc.attention_mask.bool()
    text_len = enc.attention_mask.sum(1).float()
    frame_idx = torch.arange(T, device=device).unsqueeze(0)
    audio_mask = frame_idx < real_frame_lens.clamp(max=T).unsqueeze(1)   # (B, T) real content only
    latent_cond = torch.zeros_like(x1)

    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device) * (1 - 2e-3) + 1e-3
    t_ = t.view(B, 1, 1)
    xt = (1 - t_) * x0 + t_ * x1
    target_v = x1 - x0

    out = model.transformer(
        x=xt, text=text_condition, text_len=text_len, time=t,
        mask=audio_mask, cond_mask=text_mask, latent_cond=latent_cond,
    )
    pred_v = out["last_hidden_state"]

    mask_f = audio_mask.unsqueeze(-1).float()
    sq_err = (pred_v - target_v) ** 2 * mask_f
    valid_elements = (audio_mask.sum() * D).float().clamp(min=1.0)
    return sq_err.sum() / valid_elements


# ---------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------

def save_training_checkpoint(ckpt_dir, model, optimizer, lr_scheduler, global_step, losses):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.transformer.save_pretrained(str(ckpt_dir / "lora_adapter"))
    torch.save({
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": lr_scheduler.state_dict(),
        "global_step": global_step,
        "losses": losses,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
    }, ckpt_dir / "trainer_state.pt")
    print(f"  [checkpoint saved -> {ckpt_dir}  (step {global_step})]")


def find_latest_checkpoint(search_dir):
    if not search_dir:
        return None
    search_dir = Path(search_dir)
    if not search_dir.is_dir():
        return None
    candidates = sorted(search_dir.glob("step_*"), key=lambda p: p.name)
    for p in reversed(candidates):
        if (p / "trainer_state.pt").exists() and (p / "lora_adapter").exists():
            return p
    return None


def load_training_checkpoint(ckpt_path, model, base_transformer_state, cfg, lr, max_steps, device):
    from peft import PeftModel
    if hasattr(model.transformer, "unload"):
        model.transformer = model.transformer.unload()
    model.transformer.load_state_dict(base_transformer_state)
    model.transformer = PeftModel.from_pretrained(model.transformer, str(Path(ckpt_path) / "lora_adapter"),
                                                   is_trainable=True)
    model.transformer.to(device)

    # Load onto CPU first — Optimizer.load_state_dict() moves tensors to match each param's
    # device, and the RNG-state tensors must stay on CPU.
    # weights_only=False: PyTorch >=2.6 defaults to True, which refuses to unpickle the numpy
    # RNG-state tuples in our own checkpoint. Safe because we wrote this file ourselves — never
    # do this for checkpoints from an untrusted source.
    state = torch.load(Path(ckpt_path) / "trainer_state.pt", map_location="cpu", weights_only=False)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    optimizer.load_state_dict(state["optimizer_state"])
    warmup_steps = max(1, int(max_steps * cfg.warmup_ratio))
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    lr_scheduler.load_state_dict(state["scheduler_state"])

    torch.set_rng_state(state["rng_state"])
    if state.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    np.random.set_state(state["numpy_rng_state"])

    print(f"Resumed from {ckpt_path} at step {state['global_step']}/{max_steps}")
    return model, optimizer, lr_scheduler, state["global_step"], state["losses"]


# ---------------------------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------------------------

def train_lora(model, tokenizer, loader, cfg, device, max_steps, lr,
                ckpt_dir=None, ckpt_every=None, log_every=20,
                resume_search_dir=None, base_transformer_state=None):
    resumed_ckpt = find_latest_checkpoint(resume_search_dir) if resume_search_dir else None

    if resumed_ckpt is not None:
        model, optimizer, lr_scheduler, global_step, losses = load_training_checkpoint(
            resumed_ckpt, model, base_transformer_state, cfg, lr, max_steps, device)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
        warmup_steps = max(1, int(max_steps * cfg.warmup_ratio))
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
        global_step = 0
        losses = []

    if global_step >= max_steps:
        print(f"Checkpoint already at step {global_step} >= max_steps {max_steps}; nothing to do.")
        return losses

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    model.train()
    model.text_encoder.eval()
    model.vae.eval()

    micro_step = 0
    optimizer.zero_grad()

    while global_step < max_steps:
        for batch in loader:
            loss = flow_matching_loss(model, tokenizer, batch, cfg, device) / cfg.grad_accum_steps
            loss.backward()
            losses.append(loss.item() * cfg.grad_accum_steps)
            micro_step += 1

            if micro_step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    recent = losses[-log_every * cfg.grad_accum_steps:]
                    print(f"  step {global_step}/{max_steps} | loss {sum(recent)/len(recent):.4f}")

                if ckpt_dir and ckpt_every and global_step % ckpt_every == 0:
                    p = Path(ckpt_dir) / f"step_{global_step:06d}"
                    save_training_checkpoint(p, model, optimizer, lr_scheduler, global_step, losses)

                if global_step >= max_steps:
                    break
        if global_step >= max_steps:
            break

    if ckpt_dir:
        final_p = Path(ckpt_dir) / f"step_{global_step:06d}"
        save_training_checkpoint(final_p, model, optimizer, lr_scheduler, global_step, losses)

    return losses


# ---------------------------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------------------------

def smooth(xs, w=20):
    if len(xs) < w:
        return xs
    return [sum(xs[max(0, i - w):i + 1]) / len(xs[max(0, i - w):i + 1]) for i in range(len(xs))]


def plot_loss_curve(losses, out_path):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, alpha=0.3, label="raw")
    plt.plot(smooth(losses), label="smoothed")
    plt.xlabel("micro-batch")
    plt.ylabel("masked flow-matching MSE loss")
    plt.title("SenticDiT — main run loss (class-balanced sampling)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
