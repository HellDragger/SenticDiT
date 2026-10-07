"""Loading the real pretrained LongCat-AudioDiT-1B, the weight-loading guard, and LoRA helpers."""
import copy
import sys

import torch
from peft import LoraConfig, PeftModel, get_peft_model


def import_audiodit(cfg):
    """`audiodit` comes from the cloned LongCat-AudioDiT repo (see setup_env), not from PyPI.
    Importing it registers AudioDiTConfig / AudioDiTModel with transformers."""
    if cfg.longcat_repo_dir not in sys.path:
        sys.path.insert(0, cfg.longcat_repo_dir)
    import audiodit  # noqa: F401
    from audiodit import AudioDiTModel
    return AudioDiTModel


def load_pretrained(cfg, device, snapshot=True):
    """Returns (model, tokenizer, original_transformer_state). Overwrites cfg.sample_rate and
    cfg.latent_hop from the real model config. snapshot=False skips the clean-state copy (only
    needed when LoRA adapters are trained and reset)."""
    from transformers import AutoTokenizer

    AudioDiTModel = import_audiodit(cfg)
    model = AudioDiTModel.from_pretrained(cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

    cfg.sample_rate = model.config.sampling_rate
    cfg.latent_hop  = model.config.latent_hop

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {cfg.model_id}")
    print(f"Total params: {n_params/1e6:.1f}M | sample_rate={cfg.sample_rate} | latent_hop={cfg.latent_hop}")

    # Snapshot the clean, pre-LoRA transformer weights once — we reset back to this between the
    # ablation arms and before the main run, so LoRA experiments never contaminate each other
    # without needing to reload the whole 1B-param model.
    if not snapshot:
        return model, tokenizer, None
    original_state = copy.deepcopy(model.transformer.state_dict())
    print("Cached clean transformer state for later resets.")
    return model, tokenizer, original_state


def verify_weight_loading(model, cfg):
    """Compare a sample of loaded tensors against the checkpoint on disk and stop on mismatch.

    The v1 version of this work loaded nothing: a generic AutoModel call failed, a try/except
    substituted a freshly initialised network, and every downstream metric was computed on an
    untrained model — while still producing a plausible MCD number.
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors import safe_open

    assert type(model).__name__ == "AudioDiTModel", \
        f"Expected AudioDiTModel, got {type(model).__name__} — a fallback class was substituted."

    sd = model.state_dict()
    checked, mismatched = 0, []
    shard_files = [f for f in list_repo_files(cfg.model_id) if f.endswith(".safetensors")]
    for fn in shard_files:
        path = hf_hub_download(cfg.model_id, fn)          # already cached by from_pretrained
        with safe_open(path, framework="pt") as f:
            keys = [k for k in f.keys() if k in sd]
            # spread the sample across the file rather than taking the first few layers only
            for k in keys[:: max(1, len(keys) // 6)][:6]:
                ref = f.get_tensor(k).float()
                got = sd[k].detach().float().cpu()
                if ref.shape != got.shape or not torch.allclose(ref, got, rtol=1e-2, atol=1e-3):
                    mismatched.append(k)
                checked += 1
        if checked >= 24:
            break

    if checked == 0:
        print("WARNING: no checkpoint keys matched the model's state_dict names, so weights could")
        print("not be verified. Inspect key naming before trusting any result from this run.")
    elif mismatched:
        raise RuntimeError(f"Loaded weights do not match the checkpoint for {len(mismatched)}/{checked} "
                           f"sampled tensors (e.g. {mismatched[:3]}). Refusing to continue.")
    else:
        print(f"Weight-loading guard passed: {checked} sampled tensors match the checkpoint on disk.")


# ---------------------------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------------------------

def make_lora_config(r: int, cfg, include_adaln: bool = False) -> LoraConfig:
    targets = cfg.lora_target_modules_with_adaln if include_adaln else cfg.lora_target_modules
    return LoraConfig(r=r, lora_alpha=r * cfg.lora_alpha_mult, lora_dropout=cfg.lora_dropout,
                       target_modules=list(targets), bias="none")


def summarize_lora_targets(peft_model, verbose=True):
    '''Lists which modules actually got LoRA-wrapped -- don't trust a target_modules string/list
    blindly; confirm it matched something real in this specific checkpoint.'''
    matched = [name for name, module in peft_model.named_modules() if hasattr(module, "lora_A")]
    n_attn = sum(1 for m in matched if m.endswith(("to_q", "to_k", "to_v")))
    n_adaln = sum(1 for m in matched if m.endswith("mlp.1"))
    if verbose:
        print(f"LoRA-adapted modules: {len(matched)} total ({n_attn} attention Q/K/V, {n_adaln} AdaLN MLP)")
    return {"matched": matched, "n_attn": n_attn, "n_adaln": n_adaln}


def reset_transformer(model, original_state):
    """Remove any LoRA adapter and restore the clean pretrained transformer weights."""
    if isinstance(model.transformer, PeftModel):
        model.transformer = model.transformer.unload()
    model.transformer.load_state_dict(original_state)


def attach_lora(model, original_state, r: int, cfg, include_adaln: bool = False):
    """Fresh adapter on top of clean pretrained weights."""
    reset_transformer(model, original_state)
    model.transformer = get_peft_model(model.transformer, make_lora_config(r, cfg, include_adaln))
    return model
