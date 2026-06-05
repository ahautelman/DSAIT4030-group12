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
from diffuser.unet import DiffusionUNet

from dataset_loader import load_dataset
from vae.vae import VAE

# Config:
from diffuser.unet_config import DiffuserConfig

def sample(model, ddpm, shape, fixed_noise=None):
    model.eval()

    x = torch.randn(shape, device=device)

    if fixed_noise != None:
        x = fixed_noise

    for t in range(ddpm.total_timesteps - 1, 0, -1):
        ts = torch.full((shape[0],), t, device=device, dtype=torch.long).view(-1)
        pred_noise = model(x, ts)
        x = ddpm.reverse_diffusion(x, ts, pred_noise)

    return x

def save_images(img_batch, filepath, title=None):
    img_batch = img_batch.detach().cpu()
    fig, axes = plt.subplots(1, img_batch.shape[0], figsize=(4 * img_batch.shape[0], 4))
    if img_batch.shape[0] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        img = img_batch[i]
        img = (img + 1.0) / 2.0
        img = (img * 255.0).clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.savefig(filepath, bbox_inches='tight')
    
    plt.close(fig)

#############################################################################
# Set the path to the VAE checkpoint
checkpoint_dir = "../checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
vae_checkpoint_path = f"{checkpoint_dir}/step_100000.pt"
diffusion_checkpoint_path = f"{checkpoint_dir}/latent_diffusion_ddpm_checkpoint.pt"

show_images_num = 16
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
unet = DiffusionUNet(
    config=config,
    model_in_channels=4,
    model_out_channels=4
).to(device)
unet.eval()

# Create VAE model and load checkpoint
vae = VAE(mode="kl").to(device)
checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
vae.load_state_dict(checkpoint["vae"], strict=False)

if os.path.exists(diffusion_checkpoint_path):

    checkpoint = torch.load(diffusion_checkpoint_path, map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint["unet"])
    start_iteration = checkpoint["iteration"] + 1

else:

    sys.exit("Checkpoint not found...")

for i in range(show_images_num):
    with torch.no_grad():
        latent_samples = sample(unet, ddpm_model, (1, 4, 32, 32))
        samples = vae.decode(latent_samples)
        save_images(samples, f"{checkpoint_dir}/Sample_{i}.png")

