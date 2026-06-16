import os
import sys
from sys import platform

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

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
from diffuser.frequency_mse import compute_per_frequency_denoising_mse
from diffuser.frequency_weighted_noise import frequency_weighted_gaussian_noise

from dataset_loader import load_dataset
from diffuser.metrics import compute_FID
from cleanfid import fid
from diffuser.metrics import store_FID_baseline

from vae.vae import VAE

from diffuser.unet_config import DiffuserConfig

from models import SiT_models

def sample_ddpm(model, ddpm, shape, noise_mode, noise_strength):
    model.eval()

    # 1. Start from pure noise
    dummy_x = torch.zeros(shape, device=device)
    x = frequency_weighted_gaussian_noise(dummy_x, mode=noise_mode, strength=noise_strength)

    # 2. Iterate backward from T-1 down to 0
    for t in reversed(range(0, ddpm.total_timesteps)):
        ts = torch.full((shape[0],), t, device=device, dtype=torch.long)
        y = torch.full((shape[0],), 1000, device=device, dtype=torch.long)

        with torch.no_grad():
            model_output = model(x, ts, y)

        if model_output.shape[1] == shape[1] * 2:
            pred_noise, _ = model_output.chunk(2, dim=1)
        else:
            pred_noise = model_output

        # Grab the schedule variables for this timestep
        alpha_t = ddpm.alphas[ts].view(-1, 1, 1, 1)
        alpha_bar_t = ddpm.alpha_bars[ts].view(-1, 1, 1, 1)
        beta_t = ddpm.betas[ts].view(-1, 1, 1, 1)

        # 3. Add noise for Langevin dynamics (except on the final step t=0)
        if t > 0:
            noise = frequency_weighted_gaussian_noise(x, mode=noise_mode, strength=noise_strength)
        else:
            noise = torch.zeros_like(x)

        # 4. The standard DDPM reverse step formula
        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        ) + torch.sqrt(beta_t) * noise

    return x

def save_images(img_batch, filepath, title=None):
    save_image(img_batch, filepath, normalize=True, value_range=(-1.0, 1.0))

#############################################################################
# Set the path to the VAE checkpoint
checkpoint_dir = "../checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

vae_checkpoint_path = f"{checkpoint_dir}/VAE_ESM_step_200000.pt"
diffusion_checkpoint_path = f"/media/remcohuijsen/Expansion/generative_modeling_checkpoints/SiT_ESM_lf/latent_diffusion_ddpm_esm_sit_lf_checkpoint_10000_.pt"
#diffusion_checkpoint_path = f"{checkpoint_dir}/latent_diffusion_ddpm_repa_checkpoint.pt"

# FID calculations
FID_BASELINE_NAME = "celeba256"
GENERATED_DIR = f"{checkpoint_dir}/generated"

# Use the celeba dataset directory for baseline stats
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DATA_DIR = os.path.join(BASE_DIR, "data", "celeba", "validation")

fid_images_num = 1
process_images_per_it = 1
#############################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

ddpm_model = Diffuser_DDPM_linear_schedule(total_timesteps=1000, beta_start=0.0001, beta_end=0.02, noise_mode="lf")
ddpm_model.betas = ddpm_model.betas.to(device)
ddpm_model.alphas = ddpm_model.alphas.to(device)
ddpm_model.alpha_bars = ddpm_model.alpha_bars.to(device)

# Create U-Net model
config = DiffuserConfig()
unet = SiT_models['SiT-L/2'](
    input_size=32, 
    in_channels=4
).to(device)
unet.eval()

vae = VAE(mode="esm").to(device)
checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
vae.load_state_dict(checkpoint["vae"], strict=False)

if os.path.exists(diffusion_checkpoint_path):
    checkpoint = torch.load(diffusion_checkpoint_path, map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint["unet"]) 
    start_iteration = checkpoint["iteration"] + 1
    print(start_iteration)
else:
    sys.exit("Checkpoint not found...")

# Get directory for generated images
os.makedirs(GENERATED_DIR, exist_ok=True)

it = fid_images_num // process_images_per_it

for i in range(it):
    print(i)
    with torch.no_grad():
        latent_samples = sample_ddpm(unet, ddpm_model, (process_images_per_it, 4, 32, 32), 'lf', 1.0)
        latent_samples = latent_samples / 0.18215
        samples = vae.decode(latent_samples)

        for j in range(process_images_per_it):
            # Add unsqueeze(0) to keep the batch dimension for your save_images function
            save_images(samples[j].unsqueeze(0), f"{GENERATED_DIR}/Sample_{i*process_images_per_it+j}.png")

# # Store baseline for fid calculations if baseline does not already exist
# if not fid.test_stats_exists(FID_BASELINE_NAME, "clean"):
#     print(f"FID baseline '{FID_BASELINE_NAME}' not found. Creating it...")

#     # Check if we need to store the evaluation images 
#     if not os.path.exists(BASELINE_DATA_DIR) or len(os.listdir(BASELINE_DATA_DIR)) == 0:
#         print(f"Extracting evaluation images from celebA from Huggingface")
#         os.makedirs(BASELINE_DATA_DIR, exist_ok=True)
    
#         val_dataset = load_dataset("celeba", split="validation", img_size=256)

#         # Extract and Save images
#         for i in range(len(val_dataset)):
#             batch = val_dataset[i]
#             img = (batch["images"])
#             save_images(img.unsqueeze(0), os.path.join(BASELINE_DATA_DIR, f"img_{i:06d}.png"))

#             if (i + 1) % 100 == 0:
#                 print(f"Progress: extracted {i+1} images")

#         print("Done extracting images")

#     store_FID_baseline(
#         baseline_stats_name=FID_BASELINE_NAME,
#         image_dir=BASELINE_DATA_DIR,
#         device=device,
#     )
#     print("FID baseline created.")
# else:
#     print(f"Using existing FID baseline '{FID_BASELINE_NAME}'.")

# # gFID calculation
# fid = compute_FID(baseline_stats_name="celeba256", 
#                   dir=GENERATED_DIR, 
#                   device=device,
#                   resolution=256
#                   )

# print(fid)

# ====================================================================
# ALP-ADDITION: Frequency MSE Evaluation
# ====================================================================

print("\n--- Starting Frequency MSE Evaluation ---")

# 2. Create a wrapper to handle SiT's class labels and chunked variance outputs
def sit_frequency_wrapper(x_t, t):
    y = torch.full((x_t.shape[0],), 1000, device=device, dtype=torch.long)
    model_output = unet(x_t, t, y)
    
    if model_output.shape[1] == x_t.shape[1] * 2:
        pred_noise, _ = model_output.chunk(2, dim=1)
        return pred_noise
    return model_output

# 3. Load Validation Data via DataLoader
# Using the existing dataset logic from your file
val_dataset_eval = load_dataset("celeba", split="validation", img_size=256)
val_loader = DataLoader(val_dataset_eval, batch_size=32, shuffle=False, num_workers=2)

# Select which timesteps you want to evaluate the spectral error on
timesteps_to_evaluate = [100, 500, 900]
frequency_results = {t: [] for t in timesteps_to_evaluate}

# 4. Evaluation Loop
unet.eval()
vae.eval()

# To save time, we evaluate the first 5 batches. Increase max_batches for a full validation pass.
max_batches = 100 

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        # if i >= max_batches:
        #     break
            
        real_images = batch["images"].to(device)
        
        # CRITICAL: Project images into latent space before calculating metric
        latents = vae.encode(real_images) * 0.18215
        
        for t_val in timesteps_to_evaluate:
            rad_avg_error = compute_per_frequency_denoising_mse(
                model=sit_frequency_wrapper,
                ddpm=ddpm_model,
                images=latents,
                t_value=t_val,
                device=device
            )
            frequency_results[t_val].append(rad_avg_error)

# 5. Aggregate and display results
for t_val in timesteps_to_evaluate:
    # Stack all batches and compute the mean across them
    avg_rad_error = torch.stack(frequency_results[t_val]).mean(dim=0)
    
    print(f"\nTimestep {t_val} - Mean Radial Spectral Error (first 10 radii):")
    for radius, err in enumerate(avg_rad_error[:10]):
        print(f"  Radius {radius}: {err.item():.4f}")
