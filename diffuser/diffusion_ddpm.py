import os, sys
from sys import platform

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Check platform for platform specific operations
if platform == "linux" or platform == "linux2":
    # We assume that the project folder is located in the home directory
    home_dir = os.path.expanduser("~")
    sys.path.insert(0, os.path.abspath(os.path.join(home_dir, 'DSAIT4030-group12')))
    # Requires you to clone https://github.com/willisma/SiT into your home folder
    sys.path.insert(0, os.path.abspath(os.path.join(home_dir, 'SiT')))

elif platform == "win32":
    # Set import root to project root, to find dataset_loader and vae 
    sys.path.insert(0, os.path.abspath(os.path.join("..")))
    # TO DO: Windows integration of loading SiT

from dataset_loader import load_dataset

from diffuser.diffuser_ddpm_linear_schedule import Diffuser_DDPM_linear_schedule 
from diffuser.style_loss import VGGGramStyleLoss
from diffuser.unet import DiffusionUNet
from diffuser.unet_config import DiffuserConfig

from models import SiT_models

def load_dataloader(batch_size, dataset, percentage=1.0):

    training_set = load_dataset(dataset, split="train")
    n_samples = int(len(training_set) * percentage)

    training_loader = DataLoader(
        Subset(training_set, range(n_samples)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return training_loader

def training_procedure_general(data_i, batch_size, minibatch_size, model, diffusion_model, model_type="unet"):
    
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_samples = 0

    minibatch_num = torch.div(batch_size, minibatch_size)

    while total_samples < batch_size:

        # Load the next minibatch
        minibatch = next(data_i)["images"].to(device)

        # Randomly sample diffusion timesteps and perform forward diffusion
        t = torch.randint(0, diffusion_model.total_timesteps, (minibatch_size,), device=device).view(-1)
        x_t, eps_t = diffusion_model.forward_diffusion(minibatch, t)
        
        if model_type != "unet":

            # SiT expects a class label tensor
            y = torch.full((minibatch_size,), 1000, device=device)

            # Predict added noise per sample
            eps_t_hat = model(x_t, t, y)

        else:
            
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

def training_procedure_with_style_transfer(data_i, batch_size, minibatch_size, model, diffusion_model, style_loss_fn=None, style_loss_weight=0.0, style_loss_t_max=300, model_type="unet"):
    
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_mse_loss = 0.0
    total_style_loss = 0.0
    samples = 0

    minibatch_num = torch.div(batch_size, minibatch_size)

    while samples < batch_size:
        
        style_loss = torch.tensor(0.0, device=device)
        
        # Load the next minibatch
        minibatch = next(data_i)["images"].to(device)

        # Randomly sample diffusion timesteps and perform forward diffusion
        t = torch.randint(0, diffusion_model.total_timesteps, (minibatch_size,), device=device).view(-1)
        x_t, eps_t = diffusion_model.forward_diffusion(minibatch, t)

        if model_type != "unet":

            # SiT expects a class label tensor
            y = torch.full((minibatch_size,), 1000, device=device)

            # Predict added noise per sample
            eps_t_hat = model(x_t, t, y)

        else:
            
            # Predict added noise per sample
            eps_t_hat = model(x_t, t)
        
        # Compute L2 loss
        mse_loss = F.mse_loss(eps_t_hat, eps_t)

        alpha_bar_t = diffusion_model.alpha_bars[t].to(device).view(-1, 1, 1, 1)
        x0_hat = torch.div(torch.sub(x_t, torch.mul(torch.sqrt(torch.sub(1.0, alpha_bar_t)), eps_t_hat)), torch.sqrt(alpha_bar_t))
        
        # Only compute style loss for valid t values
        style_mask = t <= style_loss_t_max

        if style_mask.any():

            with torch.amp.autocast(device_type=device, enabled=False):

                decoded_images = torch.div(torch.add(x0_hat[style_mask], 1.0), 2.0)
                style_loss = style_loss_fn(decoded_images)
                
            batch_loss = torch.add(mse_loss, torch.mul(style_loss_weight, style_loss))
        
        else:

            batch_loss = mse_loss

        # Normalize by the amount of minibatches
        norm_loss = torch.div(batch_loss, minibatch_num)
        norm_loss.backward()

        total_loss = torch.add(total_loss, norm_loss.item())
        total_mse_loss = torch.add(total_mse_loss, mse_loss.item())
        total_style_loss = torch.add(total_style_loss, style_loss.item())
        samples = torch.add(samples, minibatch_size)

    # Perform one forward step after accumilating all the backward steps
    optimizer.step()
    
    return total_loss, total_mse_loss, total_style_loss

###############################################################################################
# Checkpoint paths
training_checkpoint_path = f"../checkpoints/diffusion_ddpm_test_checkpoint.pt"

# General settings
noise_prediction_model = "sit_l_2" # Use unet or sit_l_2

# Style loss settings
use_style_loss = False
style_image_path = "../style_images/starry_night.jpg"
style_loss_weight = 1.0 # Can be reduced to make it more stable
style_loss_t_max = 300
style_image_size = 64

# Diffusion model settings
diffusion_T = 1000
diffusion_beta_start = 0.0001
diffusion_beta_end = 0.02
noise_mode = "white" # Use white, lf or hf
noise_strength = 1.0

# General optimization settings
iterations = 10000
batch_size = 1
minibatch_size = 1 # Change based on what your GPU can handle :)
learning_rate = 1e-4
weight_decay = 1e-6

# Dataset settings
celeba_dataset_percentage = 0.35

# Iterative settings
save_checkpoint_every = 200
save_checkpoint_milestone_every = 5000
print_loss_every = 1
###############################################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define DDPM diffusion model and put it on the device
ddpm_model = Diffuser_DDPM_linear_schedule(total_timesteps=diffusion_T, beta_start=diffusion_beta_start, beta_end=diffusion_beta_end, noise_mode=noise_mode, noise_strength=noise_strength)
ddpm_model.betas = ddpm_model.betas.to(device)
ddpm_model.alphas = ddpm_model.alphas.to(device)
ddpm_model.alpha_bars = ddpm_model.alpha_bars.to(device)

# Only load the VGGGramStyleLoss if we do style loss
if use_style_loss:
    style_loss_fn = VGGGramStyleLoss(style_image_path=style_image_path, device=device, image_size=style_image_size).to(device)
    style_loss_fn.eval()

# Load either a U-Net or a SiT
if noise_prediction_model == "unet":

    config = DiffuserConfig()
    model = DiffusionUNet(config=config, model_in_channels=3, model_out_channels=3).to(device)

elif noise_prediction_model == "sit_l_2":

    model = SiT_models['SiT-L/2'](input_size=256, in_channels=3, learn_sigma=False).to(device)

# Initialize AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Either we start from iteration zero, or this gets overwritten when a checkpoint exists
start_iteration = 0

if os.path.exists(training_checkpoint_path):

    checkpoint = torch.load(training_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"]) # Prediction model
    optimizer.load_state_dict(checkpoint["optimizer"]) # Optimizer
    start_iteration = checkpoint["iteration"] + 1 # Start iteration

train_loader = load_dataloader(minibatch_size, "celeba", percentage=celeba_dataset_percentage)
data_i = iter(train_loader)

# If we use style loss
if use_style_loss:

    for i in range(start_iteration, iterations):

        # Autocast to bfloat16 for less demanding compute
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        
            # Try to perform a training iteration, if it fails we presume we are out of training samples and retrieve new data
            try:
                loss, mse_loss, style_loss = training_procedure_with_style_transfer(data_i, batch_size, minibatch_size, model, ddpm_model, style_loss_fn=style_loss_fn, style_loss_weight=style_loss_weight, style_loss_t_max=style_loss_t_max, model_type=noise_prediction_model)
            except StopIteration:
                data_i = iter(train_loader)
                loss, mse_loss, style_loss = training_procedure_with_style_transfer(data_i, batch_size, minibatch_size, model, ddpm_model, style_loss_fn=style_loss_fn, style_loss_weight=style_loss_weight, style_loss_t_max=style_loss_t_max, model_type=noise_prediction_model)

        # Iterative functions
        if (i+1) % print_loss_every == 0:
            print(f"step {(i+1)} | loss {loss} | mse loss {mse_loss} | style loss {style_loss}")
            
        save_condition_1 = (save_checkpoint_every and save_checkpoint_every > 0 and ((i+1) % save_checkpoint_every == 0) and i != start_iteration)
        save_condition_2 = (save_checkpoint_milestone_every and save_checkpoint_milestone_every > 0 and ((i+1) % save_checkpoint_milestone_every == 0) and i != start_iteration)
        
        if save_condition_1 or save_condition_2:

            checkpoint = {'iteration': i, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        
            if save_condition_1:
                torch.save(checkpoint, training_checkpoint_path)

            if save_condition_2:
                torch.save(checkpoint, f"{training_checkpoint_path[0:-3]}_{(i+1)}_.pt")

# Some more general training procedure
else:

    for i in range(start_iteration, iterations):

        # Autocast to bfloat16 for less demanding compute
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        
            # Try to perform a training iteration, if it fails we presume we are out of training samples and retrieve new data
            try:
                loss = training_procedure_general(data_i, batch_size, minibatch_size, model, ddpm_model, model_type=noise_prediction_model)
            except StopIteration:
                data_i = iter(train_loader)
                loss = training_procedure_general(data_i, batch_size, minibatch_size, model, ddpm_model, model_type=noise_prediction_model)

        if (i+1) % print_loss_every == 0:
            print(f"step {(i+1)} | loss {loss}")

        save_condition_1 = (save_checkpoint_every and save_checkpoint_every > 0 and ((i+1) % save_checkpoint_every == 0) and i != start_iteration)
        save_condition_2 = (save_checkpoint_milestone_every and save_checkpoint_milestone_every > 0 and ((i+1) % save_checkpoint_milestone_every == 0) and i != start_iteration)
        
        if save_condition_1 or save_condition_2:

            checkpoint = {'iteration': i, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        
            if save_condition_1:
                torch.save(checkpoint, training_checkpoint_path)

            if save_condition_2:
                torch.save(checkpoint, f"{training_checkpoint_path[0:-3]}_{(i+1)}_.pt")