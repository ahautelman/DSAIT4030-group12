import os
import sys
from sys import platform

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

if platform == "linux" or platform == "linux2":
    # We assume that the project folder is located in the home directory
    home_dir = os.path.expanduser("~")
    sys.path.insert(0, os.path.abspath(os.path.join(home_dir, 'DSAIT4030-group12')))

elif platform == "win32":
    # Set import root to project root, to find dataset_loader and vae 
    sys.path.insert(0, os.path.abspath(os.path.join("..")))

from diffuser.diffuser_ddpm_linear_schedule import Diffuser_DDPM_linear_schedule 
from diffuser.style_loss import VGGGramStyleLoss #ALP-ADDITION
from diffuser.unet import DiffusionUNet

# ALP-ADDITION: To be used when we add validation
import csv
from diffuser.frequency_mse import compute_per_frequency_denoising_mse

from dataset_loader import load_dataset

# Config:
from diffuser.unet_config import DiffuserConfig

def save_checkpoint(iteration, checkpoint_path):
    
    checkpoint = {
        'iteration': iteration,
        'unet': unet.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)

# Loads data and uses VAE to generate latents
def load_data(NUM_WORKERS, BATCH_SIZE, DATASET, percentage=1.0):
    train_set = load_dataset(DATASET, split="train")

    if percentage < 1.0:
        num_samples = int(len(train_set) * percentage)
        train_set = Subset(train_set, range(num_samples))

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    return train_loader
                                                                # ALP-ADDITION: Style loss params for training
def train_step(data_iter, batch_size, model, ddpm, fold_factor=1, use_style_loss=False, style_loss_fn=None, style_loss_weight=0.0, style_loss_t_max=300):
    
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    data_samples = 0

    while data_samples < batch_size:

        batch = next(data_iter)

        data_minibatch = batch["images"].to(device, non_blocking=True)

        if fold_factor > 1:
            data_minibatch = F.pixel_unshuffle(data_minibatch, downscale_factor=fold_factor)

        # Randomly sample diffusion timesteps for each data sample in the minibatch.
        B = data_minibatch.shape[0]
        t = torch.randint(0, ddpm.total_timesteps, (B,), device=device, dtype=torch.long).view(-1)

        # Perform a forward diffusion step to time t and retrieve the corresponding noisy sample x_t and the true added noise.
        x_t, true_noise = ddpm.forward_diffusion(data_minibatch, t)
        
        # Let the prediction model predict the added noise given the noisy sample x_t and the diffusion timestep t.
        pred_noise = model(x_t, t)

        # ALP-ADDITION: Style loss added
        # # Compute loss and backprop (normalized by total batch for stability)
        # loss = F.mse_loss(pred_noise, true_noise)
        # Standard DDPM noise prediction loss.
        ddpm_loss = F.mse_loss(pred_noise, true_noise)
        loss = ddpm_loss

        # Optional Gatys-style Gram matrix loss
        if use_style_loss and style_loss_fn is not None and style_loss_weight > 0.0:
            alpha_bar_t = ddpm.alpha_bars[t].to(device).view(-1, 1, 1, 1)

            x0_hat = (
                x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise
            ) / torch.sqrt(alpha_bar_t)

            if fold_factor > 1:
                x0_hat_rgb = F.pixel_shuffle(x0_hat, upscale_factor=fold_factor)
            else:
                x0_hat_rgb = x0_hat

            # Only apply style loss to lower/no-moderate noise timesteps.
            # At very high t, x0_hat can be too unstable for VGG style supervision.
            style_mask = t <= style_loss_t_max

            if style_mask.any():
                # style_loss_value = style_loss_fn(x0_hat_rgb[style_mask])

                x0_hat_rgb_for_style = (x0_hat_rgb.clamp(-1.0, 1.0) + 1.0) / 2.0 # Convert from diffusion training range [-1, 1] to VGG input range [0, 1]
                with torch.amp.autocast(device_type=device, enabled=False): 
                    style_loss_value = style_loss_fn(
                        x0_hat_rgb_for_style[style_mask].float()
                    )
                loss = loss + style_loss_weight * style_loss_value

    
        accumulation_steps = batch_size / B
        scaled_loss = loss / accumulation_steps

        scaled_loss.backward()

        total_loss += loss.detach().item() * B
        data_samples += B

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    return total_loss / max(data_samples, 1)

#############################################################################
## ALP-ADDITION
# NEW NOISES
noise_mode = "white"      # "white", "hf", or "lf"
noise_strength = 1.0

# Set the path to the VAE checkpoint
checkpoint_dir = "../checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
vae_checkpoint_path = f"{checkpoint_dir}/step_100000.pt"

iterations = 10000
batch_size = 256
minibatch_size = 4
num_workers = 0

save_checkpoint_every = 100
save_checkpoint_milestone_every = 1000
print_loss_every = 25
fold_factor = 2

## ALP-ADDITION:
#STYLE LOSS CONFIG
use_style_loss = False
style_image_path = "style_images/starry_night.jpg"
style_loss_weight = 0.01 # can be reduced to make it more stable
style_loss_t_max = 300
style_image_size = 64

## ALP-ADDITION:
if use_style_loss:
    diffusion_checkpoint_path = f"{checkpoint_dir}/diffusion_ddpm_{noise_mode}_strength{noise_strength}_checkpoint_with_styleloss.pt"
else:
    diffusion_checkpoint_path = f"{checkpoint_dir}/diffusion_ddpm_{noise_mode}_strength{noise_strength}_checkpoint.pt"


#############################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
np.random.seed(0)

# Create DDPM model with a linear beta schedule                                                     ## ALP-ADDITION: New params
ddpm_model = Diffuser_DDPM_linear_schedule(total_timesteps=1000, beta_start=0.0001, beta_end=0.02, noise_mode=noise_mode, noise_strength=noise_strength)
ddpm_model.betas = ddpm_model.betas.to(device)
ddpm_model.alphas = ddpm_model.alphas.to(device)
ddpm_model.alpha_bars = ddpm_model.alpha_bars.to(device)


## ALP-ADDITION:
style_loss_fn = None
if use_style_loss:
    style_loss_fn = VGGGramStyleLoss(
        style_image_path=style_image_path,
        device=device,
        image_size=style_image_size,
    ).to(device)

    style_loss_fn.eval()

folded_channels = 3 * (fold_factor ** 2)

# Create U-Net model
config = DiffuserConfig()
unet = DiffusionUNet(
    config=config,
    model_in_channels=folded_channels,
    model_out_channels=folded_channels
).to(device)

# Instantiate AdamW optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-6)

start_iteration = 0

if os.path.exists(diffusion_checkpoint_path):

    checkpoint = torch.load(diffusion_checkpoint_path, map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint["unet"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_iteration = checkpoint["iteration"] + 1

    print(f"Loaded checkpoint and starting from iteration {start_iteration}!")

train_loader = load_data(num_workers, minibatch_size, "celeba", percentage=0.35)
#train_loader = load_data(num_workers, minibatch_size, "celeba", percentage=1.0) ## for the smoke test
data_i = iter(train_loader)

# Use start_step from checkpoint (if resumed) otherwise 0
for i in range(start_iteration, iterations):

    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
    
        try:                                                                                    ## ALP-ADDITION: new params
            loss = train_step(data_i, batch_size, unet, ddpm_model, fold_factor=fold_factor, use_style_loss=use_style_loss, style_loss_fn=style_loss_fn, style_loss_weight=style_loss_weight, style_loss_t_max=style_loss_t_max)

        except:
            data_i = iter(train_loader)                                                                                     ## ALP-ADDITION: new params
            loss = train_step(data_i, batch_size, unet, ddpm_model, fold_factor=fold_factor, use_style_loss=use_style_loss, style_loss_fn=style_loss_fn, style_loss_weight=style_loss_weight, style_loss_t_max=style_loss_t_max)

    if i % print_loss_every == 0:
        print(f"step {i:5d} | loss {loss:.4f}")

    # Periodic checkpoint save
    if save_checkpoint_every and save_checkpoint_every > 0 and (i % save_checkpoint_every == 0) and i != start_iteration:
        save_checkpoint(i, diffusion_checkpoint_path)

    if save_checkpoint_milestone_every and save_checkpoint_milestone_every > 0 and (i % save_checkpoint_milestone_every == 0) and i != start_iteration:
        save_checkpoint(i, f"{diffusion_checkpoint_path[0:-3]}_{i}_.pt")
