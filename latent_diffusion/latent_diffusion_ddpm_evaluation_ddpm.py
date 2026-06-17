import os
import sys
from sys import platform
import torch
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
from diffuser.frequency_weighted_noise import frequency_weighted_gaussian_noise

from dataset_loader import load_dataset
from diffuser.metrics import compute_FID
from cleanfid import fid
from diffuser.metrics import store_FID_baseline

from vae.vae import VAE

from diffuser.unet_config import DiffuserConfig

from models import SiT_models

def sample_ddpm(model, ddpm, shape, noise_mode='white', noise_strength=1.0, model_type="unet", sit_learn_sigma=False):
    
    model.eval()

    x = frequency_weighted_gaussian_noise(torch.zeros(shape, device=device), mode=noise_mode, strength=noise_strength)

    for t in range(ddpm.total_timesteps - 1, 0, -1):

        ts = torch.full((shape[0],), t, device=device)
    
        if model_type != "unet":

            # SiT expects a class label tensor
            y = torch.full((shape[0],), 1000, device=device)

            # Predict added noise per sample
            eps_hat = model(x, ts, y)

            if sit_learn_sigma:
                if eps_hat.shape[1] == shape[1] * 2:
                    eps_hat, _ = eps_hat.chunk(2, dim=1)

        else:
            
            # Predict added noise per sample
            eps_hat = model(x, ts)

        x = ddpm.reverse_diffusion(x, ts, eps_hat)

    return x


#############################################################################
# Checkpoint paths
vae_checkpoint_path = f"../checkpoints/VAE_DSM_step_200000.pt"
training_checkpoint_path = f"/media/remcohuijsen/Expansion/generative_modeling_checkpoints/UNet_ESM_colab/latent_diffusion_ddpm_unet_esm_checkpoint_10000_.pt"

# General settings
vae_type = "esm" # Use kl, esm or dsm
noise_prediction_model = "unet" # Use unet or sit_l_2
sit_learn_sigma = True # SiT parameter to turn on/off to learn sigma, flip if there are issues with loading/using the model
noise_mode = "white" # Use white, lf or hf
noise_strength = 1.0

# FID calculations
FID_BASELINE_NAME = "celeba256"
GENERATED_DIR = f"../generated"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DATA_DIR = os.path.join(BASE_DIR, "data", "celeba", "validation")
fid_images_num = 10000
process_images_per_it = 1
#############################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

ddpm_model = Diffuser_DDPM_linear_schedule(total_timesteps=1000, beta_start=0.0001, beta_end=0.02, noise_mode=noise_mode, noise_strength=noise_strength)
ddpm_model.betas = ddpm_model.betas.to(device)
ddpm_model.alphas = ddpm_model.alphas.to(device)
ddpm_model.alpha_bars = ddpm_model.alpha_bars.to(device)

vae = VAE(mode=vae_type).to(device)
checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
vae.load_state_dict(checkpoint["vae"], strict=False)
vae.eval()

if noise_prediction_model == "unet":

    config = DiffuserConfig()
    model = DiffusionUNet(config=config, model_in_channels=4, model_out_channels=4).to(device)

elif noise_prediction_model == "sit_l_2":

    if sit_learn_sigma:

        model = SiT_models['SiT-L/2'](input_size=32, in_channels=4, learn_sigma=True).to(device)

    else:
        
        model = SiT_models['SiT-L/2'](input_size=32, in_channels=4, learn_sigma=False).to(device)

model.eval()

if os.path.exists(training_checkpoint_path):

    checkpoint = torch.load(training_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"]) 
    start_iteration = checkpoint["iteration"] + 1

else:

    sys.exit("Checkpoint not found...")

os.makedirs(GENERATED_DIR, exist_ok=True)

it = fid_images_num // process_images_per_it

for i in range(it):

    print(i)

    with torch.no_grad():

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):

            # Sample, denormalize and decode latents to images
            z = sample_ddpm(model, ddpm_model, (process_images_per_it, 4, 32, 32), noise_mode=noise_mode, noise_strength=noise_strength, model_type=noise_prediction_model, sit_learn_sigma=sit_learn_sigma)
            z = z / 0.18215
            imgs = vae.decode(z)

        for j in range(process_images_per_it):
            save_image(imgs[j].unsqueeze(0), f"{GENERATED_DIR}/Sample_{i*process_images_per_it+j}.png", normalize=True, value_range=(-1.0, 1.0))

# Store baseline for fid calculations if baseline does not already exist
if not fid.test_stats_exists(FID_BASELINE_NAME, "clean"):

    # Check if we need to store the evaluation images 
    if not os.path.exists(BASELINE_DATA_DIR) or len(os.listdir(BASELINE_DATA_DIR)) == 0:
        os.makedirs(BASELINE_DATA_DIR, exist_ok=True)
    
        val_dataset = load_dataset("celeba", split="validation", img_size=256)

        for i in range(len(val_dataset)):
            batch = val_dataset[i]
            img = (batch["images"])
            save_image(img.unsqueeze(0), os.path.join(BASELINE_DATA_DIR, f"img_{i:06d}.png"), normalize=True, value_range=(-1.0, 1.0))

    store_FID_baseline(baseline_stats_name=FID_BASELINE_NAME, image_dir=BASELINE_DATA_DIR,device=device)

fid = compute_FID(baseline_stats_name="celeba256", dir=GENERATED_DIR, device=device, resolution=256)
print(fid)
