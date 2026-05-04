import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from accelerate import Accelerator
from src.config import CFG

def diffusion_loss_step(
    audiodit_model, text_enc, noisy_scheduler: DDPMScheduler, batch: dict, device
) -> torch.Tensor:
    mel     = batch["mel"].to(device)
    prompts = batch["prompt"]
    B       = mel.size(0)

    noise     = torch.randn_like(mel)
    timesteps = torch.randint(
        0, noisy_scheduler.config.num_train_timesteps,
        (B,), device=device, dtype=torch.long
    )
    noisy_mel  = noisy_scheduler.add_noise(mel, noise, timesteps)

    with torch.no_grad():
        text_emb = text_enc(prompts, device)

    pred_noise = audiodit_model(noisy_mel, timesteps, text_emb)
    return F.mse_loss(pred_noise.float(), noise.float())

def train(
    audiodit_model, text_enc, sched, loader, opt, lr_sched, acc: Accelerator
):
    step_losses = []
    global_step = 0
    steps_per_epoch = len(loader)

    for epoch in range(1, CFG.num_epochs + 1):
        audiodit_model.train()

        for step, batch in enumerate(loader):
            with acc.accumulate(audiodit_model):
                loss = diffusion_loss_step(audiodit_model, text_enc, sched, batch, acc.device)
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(audiodit_model.parameters(), CFG.max_grad_norm)
                opt.step()
                lr_sched.step()
                opt.zero_grad()

            global_step += 1
            step_losses.append(loss.item())

            if global_step % 10 == 0:
                print(f"Ep {epoch}/{CFG.num_epochs} Step {step+1}/{steps_per_epoch} Loss={loss.item():.4f}")
    
    return step_losses