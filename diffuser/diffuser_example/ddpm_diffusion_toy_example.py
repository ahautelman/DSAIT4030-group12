import os, sys
from sys import platform
import numpy as np
import torch
import torchvision.utils as utils
import torch.nn.functional as F

# Check platform for platform specific operations
if platform == "linux" or platform == "linux2":
    # We assume that the project folder is located in the home directory
    home_dir = os.path.expanduser("~")
    sys.path.insert(0, os.path.abspath(os.path.join(home_dir, 'DSAIT4030-group12')))

elif platform == "win32":
    # Set import root to project root, to find dataset_loader and vae 
    sys.path.insert(0, os.path.abspath(os.path.join("..")))
    # TO DO: Windows integration of loading SiT

from dataset_loader import load_dataset

from diffuser.diffuser_example.toy_image_generator import get_toy_image_example_batch
from diffuser.diffuser_ddpm_linear_schedule import Diffuser_DDPM_linear_schedule
from diffuser.unet import DiffusionUNet
from diffuser.diffuser_example.unet_example_config import DiffuserConfig

def get_toy_image_batch(device, batch_size, x_dim, y_dim):
    start_hue_1s = np.random.randint(0, 180)
    hue_1_offsets = np.linspace(0, 180, num=batch_size, endpoint=False)
    
    hue_1s = (start_hue_1s + hue_1_offsets) % 180
    hue_2s = (hue_1s + 90) % 180

    saturation_1s = np.full(batch_size, 255)
    saturation_2s = np.full(batch_size, 255)
    
    value_1s = np.full(batch_size, 255)
    value_2s = np.full(batch_size, 255)
    
    hue_3s = np.zeros(batch_size)
    saturation_3s = np.zeros(batch_size)
    value_3s = np.zeros(batch_size)

    start_angle = np.random.rand() * 360
    angle_offsets = np.linspace(0, 360, num=batch_size, endpoint=False)
    angles = (start_angle + angle_offsets) % 360
    
    max_dim = max(x_dim, y_dim)
    line_widths = np.random.randint(max_dim//4, max_dim//2, size=batch_size)

    batch = get_toy_image_example_batch(
        x_dim=x_dim, y_dim=y_dim, # Image settings
        hue_1s=hue_1s, saturation_1s=saturation_1s, value_1s=value_1s, # Colours of region 1
        hue_2s=hue_2s, saturation_2s=saturation_2s, value_2s=value_2s, # Colours of region 2
        hue_3s=hue_3s, saturation_3s=saturation_3s, value_3s=value_3s, # Colours of line
        angles=angles, line_widths=line_widths, # Line settings
    )
    return batch.to(device)

def training_procedure_general(device, batch_size, minibatch_size, model, diffusion_model, x_dim=64, y_dim=64):
    
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_samples = 0

    minibatch_num = torch.div(batch_size, minibatch_size)

    while total_samples < batch_size:

        # Load the next minibatch
        minibatch = get_toy_image_batch(device, minibatch_size, x_dim, y_dim)

        # Randomly sample diffusion timesteps and perform forward diffusion
        t = torch.randint(0, diffusion_model.total_timesteps, (minibatch_size,), device=device).view(-1)
        x_t, eps_t = diffusion_model.forward_diffusion(minibatch, t)
            
        # Predict added noise per sample
        eps_t_hat = model(x_t, t)
        
        # Compute L2 loss
        loss = F.mse_loss(eps_t_hat, eps_t)

        # Normalize by the amount of minibatches
        norm_loss = torch.div(loss, minibatch_num)

        # Perform backwards pass per minibatch
        norm_loss.backward()

        total_loss = torch.add(total_loss, norm_loss.item())
        total_samples = torch.add(total_samples, minibatch_size)

    # Perform one forward step after accumilating all the backward steps
    optimizer.step()
    
    return total_loss

def sample(model, ddpm, shape, device, fixed_noise=None):

    model.eval()

    x = torch.randn(shape, device=device) if fixed_noise is None else fixed_noise

    for t in range(ddpm.total_timesteps - 1, 0, -1):

        ts = torch.full((shape[0],), t, device=device).view(-1)

        eps = model(x, ts)

        x = ddpm.reverse_diffusion(x, ts, eps)

    return x

###############################################################################################
# Diffusion model settings
diffusion_T = 1000
diffusion_beta_start = 0.0001
diffusion_beta_end = 0.02
noise_mode = "white" # Use white, lf or hf
noise_strength = 1.0

# Image settings
x_dim = 64
y_dim = 64

# General optimization settings
iterations = 10000
batch_size = 256
minibatch_size = 64 # Change based on what your GPU can handle :)
learning_rate = 1e-4
weight_decay = 1e-6

# Iterative settings
print_loss_every = 25
save_image_every = 25
###############################################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define DDPM diffusion model and put it on the device
ddpm_model = Diffuser_DDPM_linear_schedule(total_timesteps=diffusion_T, beta_start=diffusion_beta_start, beta_end=diffusion_beta_end, noise_mode=noise_mode, noise_strength=noise_strength)
ddpm_model.betas = ddpm_model.betas.to(device)
ddpm_model.alphas = ddpm_model.alphas.to(device)
ddpm_model.alpha_bars = ddpm_model.alpha_bars.to(device)

# Load the smaller U-Net
config = DiffuserConfig()
model = DiffusionUNet(config=config, model_in_channels=3, model_out_channels=3).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

start_iteration = 0

for i in range(start_iteration, iterations):
    
    # Autocast to bfloat16 for less demanding compute
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
    
        loss = training_procedure_general(device, batch_size, minibatch_size, model, ddpm_model, x_dim=x_dim, y_dim=y_dim)

        # Iterative checks
        with torch.no_grad():
            if i % print_loss_every == 0:
                print(f"step {i} | loss {loss}")
            if (save_image_every and save_image_every > 0 and ((i+1) % save_image_every == 0)):
                samples = sample(model, ddpm_model, (1, 3, 64, 64), device)
                utils.save_image(samples, f"../img_{i+1}.png", normalize=True, value_range=(-1, 1))