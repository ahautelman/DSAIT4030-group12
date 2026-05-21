import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import lpips
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler

from dataset_loader import load_dataset
from vae import VAE
from vae_losses import (
    PatchGAN, reconstruction_loss, perceptual_loss,
    generator_loss, discriminator_loss, adaptive_weight
)


DATASET              = "celeba"       # "celeba" or "imagenet"
MODE                 = "kl"           # "kl", "esm", or "dsm"

# Gradient Accumulation Hyperparameters
BATCH_SIZE           = 4            # Physical batch size that fits in 8GB VRAM
EFFECTIVE_BATCH_SIZE = 48            # Target mathematical batch size
ACCUMULATION_STEPS   = EFFECTIVE_BATCH_SIZE // BATCH_SIZE  # Processes 4 physical batches per step

NUM_WORKERS          = 0              
LR                   = 5e-5
WEIGHT_DECAY         = 0.005
LAMBDA1              = 0.5            # LPIPS weight
LAMBDA2              = 0.5            # GAN weight
DISC_START           = 50000          # step to start GAN training
TRAIN_STEPS          = 500000
LOG_EVERY            = 500
IMG_EVERY            = 1000
SAVE_EVERY           = 50000
SAVE_DIR             = "reconstructions"
CKPT_DIR             = "checkpoints"
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"


def save_images(x, recon, step):
    os.makedirs(SAVE_DIR, exist_ok=True)
    x = (x.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
    recon = (recon.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
    for i in range(min(4, x.size(0))):
        grid = torch.cat([x[i], recon[i]], dim=2)
        transforms.ToPILImage()(grid).save(f"{SAVE_DIR}/step_{step}_img_{i}.png")


def save_checkpoint(vae, disc, vae_opt, disc_opt, step):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({
        "step": step,
        "vae": vae.state_dict(),
        "disc": disc.state_dict(),
        "vae_opt": vae_opt.state_dict(),
        "disc_opt": disc_opt.state_dict(),
    }, f"{CKPT_DIR}/step_{step}.pt")
    print(f"Saved checkpoint at step {step}")


def train():
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET} | Mode: {MODE}")
    print(f"Physical Batch Size: {BATCH_SIZE} | Effective Batch Size: {EFFECTIVE_BATCH_SIZE}")

    # data loader configuration adjusted cleanly for Windows (num_workers=0)
    train_set = load_dataset(DATASET, split="train")
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    # models
    vae = VAE(mode=MODE).to(DEVICE)
    disc = PatchGAN().to(DEVICE)
    disc.apply(disc.init_weights)
    lpips_model = lpips.LPIPS(net="vgg").eval().to(DEVICE)

    print(f"VAE params:  {sum(p.numel() for p in vae.parameters()):,}")
    print(f"Disc params: {sum(p.numel() for p in disc.parameters()):,}")

    # optimizers
    vae_opt = torch.optim.AdamW(vae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.5, 0.9))
    disc_opt = torch.optim.AdamW(disc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.5, 0.9))

    # mixed precision scaler
    scaler = GradScaler(device="cuda")

    step = 0
    data_start = time.time()

    # Clear gradients cleanly to begin accumulation loop
    vae_opt.zero_grad()
    disc_opt.zero_grad()

    while step < TRAIN_STEPS:
        for batch_idx, batch in enumerate(train_loader):
            if step >= TRAIN_STEPS:
                break

            data_time = time.time() - data_start
            x = batch["images"].to(DEVICE)

            # --------------------------------------------------------
            # Step 1: Forward & Backward VAE (Generator)
            # --------------------------------------------------------
            vae.train()
            disc.eval()

            forward_start = time.time()
            with autocast(device_type="cuda"):
                out = vae(x)
                img_target = out["img_target"]
                recon = out["reconstruction"]
                reg_loss = out["reg_loss"]

                recon_loss_val = reconstruction_loss(img_target, recon)
                
                # OPTIMIZATION: Isolating LPIPS forward pass graph generation
                with torch.no_grad():
                    percep_loss_val = perceptual_loss(img_target, recon, lpips_model)

                if step >= DISC_START:
                    fake_scores = disc(recon)
                    gen_loss = generator_loss(fake_scores)
                    adap_w = adaptive_weight(recon_loss_val + percep_loss_val, gen_loss, vae.last_layer)
                    gan_term = LAMBDA2 * adap_w * gen_loss
                else:
                    gan_term = torch.tensor(0.0, device=DEVICE)

                # Scale the step loss down by accumulation ratio
                vae_loss = (recon_loss_val + LAMBDA1 * percep_loss_val + gan_term + reg_loss) / ACCUMULATION_STEPS

            scaler.scale(vae_loss).backward()
            forward_time = time.time() - forward_start

            # --------------------------------------------------------
            # Step 2: Forward & Backward Discriminator
            # --------------------------------------------------------
            if step >= DISC_START:
                disc.train()
                vae.eval()

                with autocast(device_type="cuda"):
                    real_scores = disc(img_target.detach())
                    fake_scores = disc(recon.detach())
                    disc_loss = discriminator_loss(real_scores, fake_scores) / ACCUMULATION_STEPS

                scaler.scale(disc_loss).backward()

            # --------------------------------------------------------
            # Step 3: Optimizer Step (Only runs every N sub-batches)
            # --------------------------------------------------------
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                # Step VAE weights
                scaler.unscale_(vae_opt)
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                scaler.step(vae_opt)
                
                # Step Discriminator weights
                if step >= DISC_START:
                    scaler.unscale_(disc_opt)
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
                    scaler.step(disc_opt)

                # Single unified scaler synchronization block
                scaler.update()
                
                # Reset tracking gradients
                vae_opt.zero_grad()
                disc_opt.zero_grad()

            # --------------------------------------------------------
            # Logging & Visual Checks
            # --------------------------------------------------------
            if step % LOG_EVERY == 0:
                # OPTIMIZATION: .item() handles CPU-GPU sync only when printing
                print(
                    f"step {step} | "
                    f"loss: {vae_loss.item() * ACCUMULATION_STEPS:.4f} | "
                    f"recon: {recon_loss_val.item():.4f} | "
                    f"percep: {percep_loss_val.item():.4f} | "
                    f"data: {data_time:.2f}s | "
                    f"fwd: {forward_time:.2f}s"
                )

            if step % IMG_EVERY == 0:
                save_images(x, recon, step)

            if step % SAVE_EVERY == 0 and step > 0:
                save_checkpoint(vae, disc, vae_opt, disc_opt, step)

            step += 1
            data_start = time.time()


if __name__ == "__main__":
    train()