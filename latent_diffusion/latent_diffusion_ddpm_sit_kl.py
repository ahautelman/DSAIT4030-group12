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
    # Requires you to clone https://github.com/willisma/SiT into your home folder
    sys.path.insert(0, os.path.abspath(os.path.join(home_dir, 'SiT')))

elif platform == "win32":
    # Set import root to project root, to find dataset_loader and vae 
    sys.path.insert(0, os.path.abspath(os.path.join("..")))
    
from diffuser.diffuser_ddpm_linear_schedule import Diffuser_DDPM_linear_schedule 
from diffuser.unet import DiffusionUNet

from dataset_loader import load_dataset
from vae.vae import VAE

from models import SiT_models

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

def train_step(data_iter, batch_size, model, ddpm):
    
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    data_samples = 0

    while data_samples < batch_size:

        batch = next(data_iter)

        data_minibatch = batch["images"].to(device, non_blocking=True)
        
        with torch.no_grad():
            data_minibatch = vae.encode(data_minibatch) * 0.18215

        # Randomly sample diffusion timesteps for each data sample in the minibatch.
        B = data_minibatch.shape[0]
        t = torch.randint(0, ddpm.total_timesteps, (B,), device=device, dtype=torch.long).view(-1)
        y = torch.full((B,), 1000, device=device, dtype=torch.long)

        # Perform a forward diffusion step to time t and retrieve the corresponding noisy sample x_t and the true added noise.
        x_t, true_noise = ddpm.forward_diffusion(data_minibatch, t)
        
        # Let the prediction model predict the added noise given the noisy sample x_t and the diffusion timestep t.
        model_output = model(x_t, t, y)

        if model_output.shape[1] == true_noise.shape[1] * 2:
            pred_noise, _ = model_output.chunk(2, dim=1)
        else:
            pred_noise = model_output
        
        # Compute loss and backprop (normalized by total batch for stability)
        loss = F.mse_loss(pred_noise, true_noise)

        accumulation_steps = batch_size / B
        scaled_loss = loss / accumulation_steps

        scaled_loss.backward()

        total_loss += loss.detach().item() * B
        data_samples += B

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    return total_loss / max(data_samples, 1)

#############################################################################
# Set the path to the VAE checkpoint
checkpoint_dir = "../checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
vae_checkpoint_path = f"{checkpoint_dir}/VAE_KL_step_200000.pt"
diffusion_checkpoint_path = f"{checkpoint_dir}/latent_diffusion_ddpm_kl_sit_checkpoint.pt"

iterations = 20000
batch_size = 128
minibatch_size = 64
num_workers = 0

save_checkpoint_every = 100
save_checkpoint_milestone_every = 2500
print_loss_every = 25
#############################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
np.random.seed(0)

# Create DDPM model with a linear beta schedule
ddpm_model = Diffuser_DDPM_linear_schedule(total_timesteps=1000, beta_start=0.0001, beta_end=0.02)
ddpm_model.betas = ddpm_model.betas.to(device)
ddpm_model.alphas = ddpm_model.alphas.to(device)
ddpm_model.alpha_bars = ddpm_model.alpha_bars.to(device)

# Create U-Net model
config = DiffuserConfig()
unet = SiT_models['SiT-L/2'](
    input_size=32, 
    in_channels=4
).to(device)

# Instantiate AdamW optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-6)

# Create VAE model and load checkpoint
vae = VAE(mode="kl").to(device)
checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
vae.load_state_dict(checkpoint["vae"], strict=False)

start_iteration = 0

if os.path.exists(diffusion_checkpoint_path):

    checkpoint = torch.load(diffusion_checkpoint_path, map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint["unet"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_iteration = checkpoint["iteration"] + 1

    print(f"Loaded checkpoint and starting from iteration {start_iteration}!")

train_loader = load_data(num_workers, minibatch_size, "celeba", percentage=0.35)
data_i = iter(train_loader)

# Use start_step from checkpoint (if resumed) otherwise 0
for i in range(start_iteration, iterations):

    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
    
        try:
            loss = train_step(data_i, batch_size, unet, ddpm_model)

        except:
            data_i = iter(train_loader)
            loss = train_step(data_i, batch_size, unet, ddpm_model)

    if i % print_loss_every == 0:
        print(f"step {i:5d} | loss {loss:.4f}")

    # Periodic checkpoint save
    if save_checkpoint_every and save_checkpoint_every > 0 and (i % save_checkpoint_every == 0) and i != start_iteration:
        save_checkpoint(i, diffusion_checkpoint_path)

    if save_checkpoint_milestone_every and save_checkpoint_milestone_every > 0 and (i % save_checkpoint_milestone_every == 0) and i != start_iteration:
        save_checkpoint(i, f"{diffusion_checkpoint_path[0:-3]}_{i}_.pt")
