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

from dataset_loader import load_dataset
from diffuser.metrics import compute_FID
from cleanfid import fid
from diffuser.metrics import store_FID_baseline

from vae.vae import VAE

from diffusers import DPMSolverMultistepScheduler

from diffuser.unet_config import DiffuserConfig

from models import SiT_models

# Initialize the DPM-Solver++ scheduler using your DDPM parameters
dpm_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear",
    algorithm_type="dpmsolver++",
    solver_order=2
)

def sample_diffusers(model, scheduler, shape, fixed_noise=None, steps=20):
    model.eval()

    x = torch.randn(shape, device=device)
    if fixed_noise is not None:
        x = fixed_noise

    # Tell the scheduler how many steps we want to take (e.g., 20 instead of 1000)
    scheduler.set_timesteps(steps, device=device)

    # The scheduler pre-calculates the exact timesteps to use
    for t in scheduler.timesteps:
        # Broadcast the timestep to the batch size for your model
        ts = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # Unconditional class label
        y = torch.full((shape[0],), 1000, device=device, dtype=torch.long)
        
        # Forward pass through your SiT model
        model_output = model(x, ts, y)

        # Handle variance outputs if your model predicts them
        if model_output.shape[1] == shape[1] * 2:
            pred_noise, _ = model_output.chunk(2, dim=1)
        else:
            pred_noise = model_output
            
        # Let the scheduler compute the previous image sample x_{t-1}
        x = scheduler.step(pred_noise, t, x).prev_sample

    return x

def save_images(img_batch, filepath, title=None):
    save_image(img_batch, filepath, normalize=True, value_range=(-1.0, 1.0))

#############################################################################
# Set the path to the VAE checkpoint
checkpoint_dir = "../checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

vae_checkpoint_path = f"{checkpoint_dir}/step_200000.pt"
diffusion_checkpoint_path = f"/media/remcohuijsen/Expansion/generative_modeling_checkpoints/SiT_ESM_colab/latent_diffusion_ddpm_sit_checkpoint_10000_.pt"
#diffusion_checkpoint_path = f"{checkpoint_dir}/latent_diffusion_ddpm_repa_checkpoint.pt"

show_images_num = 16
process_images_per_it = 4
#############################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

# torch.manual_seed(0)
# np.random.seed(0)

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

it = show_images_num // process_images_per_it

for i in range(it):
    with torch.no_grad():
        latent_samples = sample_diffusers(unet, dpm_scheduler, (process_images_per_it, 4, 32, 32), steps=20)
        
        latent_samples = latent_samples / 0.18215
        
        samples = vae.decode(latent_samples)

        for j in range(process_images_per_it):
            # Add unsqueeze(0) to keep the batch dimension for your save_images function
            save_images(samples[j].unsqueeze(0), f"{checkpoint_dir}/Sample_{i*process_images_per_it+j}.png")

#############################################################################
# FID calculations
FID_BASELINE_NAME = "celeba256"
GENERATED_DIR = f"{checkpoint_dir}/generated"

# Use the celeba dataset directory for baseline stats
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DATA_DIR = os.path.join(BASE_DIR, "data", "celeba", "validation")

fid_images_num = 10000
process_images_per_it = 40
#############################################################################

# Get directory for generated images
os.makedirs(GENERATED_DIR, exist_ok=True)

it = fid_images_num // process_images_per_it

for i in range(it):
    print(i)
    with torch.no_grad():
        latent_samples = sample_diffusers(unet, dpm_scheduler, (process_images_per_it, 4, 32, 32), steps=20)
        latent_samples = latent_samples / 0.18215
        samples = vae.decode(latent_samples)

        for j in range(process_images_per_it):
            # Add unsqueeze(0) to keep the batch dimension for your save_images function
            save_images(samples[j].unsqueeze(0), f"{GENERATED_DIR}/Sample_{i*process_images_per_it+j}.png")

# Store baseline for fid calculations if baseline does not already exist
if not fid.test_stats_exists(FID_BASELINE_NAME, "clean"):
    print(f"FID baseline '{FID_BASELINE_NAME}' not found. Creating it...")

    # Check if we need to store the evaluation images 
    if not os.path.exists(BASELINE_DATA_DIR) or len(os.listdir(BASELINE_DATA_DIR)) == 0:
        print(f"Extracting evaluation images from celebA from Huggingface")
        os.makedirs(BASELINE_DATA_DIR, exist_ok=True)
    
        val_dataset = load_dataset("celeba", split="validation", img_size=256)

        # Extract and Save images
        for i in range(len(val_dataset)):
            batch = val_dataset[i]
            img = (batch["images"])
            save_images(img.unsqueeze(0), os.path.join(BASELINE_DATA_DIR, f"img_{i:06d}.png"))

            if (i + 1) % 100 == 0:
                print(f"Progress: extracted {i+1} images")

        print("Done extracting images")

    store_FID_baseline(
        baseline_stats_name=FID_BASELINE_NAME,
        image_dir=BASELINE_DATA_DIR,
        device=device,
    )
    print("FID baseline created.")
else:
    print(f"Using existing FID baseline '{FID_BASELINE_NAME}'.")

# gFID calculation
fid = compute_FID(baseline_stats_name="celeba256", 
                  dir=GENERATED_DIR, 
                  device=device,
                  resolution=256
                  )

print(fid)
