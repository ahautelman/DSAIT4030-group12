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

from latent_diffusion.latent_diffusion_vae_wrapper import VAE_wrapper

from repa.config import ExperimentConfig
from repa.models.wrapper import REPAWrapper
from repa.train import DiffusionTrainer
from repa.models.vae import BaseVAEWrapper
from repa.models.factory import build_student_model

from vae.vae import VAE

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

def training_procedure_general(data_i, batch_size, minibatch_size, model, diffusion_model, vae, model_type="unet"):
    
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_samples = 0

    minibatch_num = torch.div(batch_size, minibatch_size)

    while total_samples < batch_size:

        # Load the next minibatch
        minibatch = next(data_i)["images"].to(device)
        
        # Encode images to latent space and normalize the latents for diffusion
        with torch.no_grad():
            minibatch = vae.encode(minibatch) * 0.18215

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

def training_procedure_with_style_transfer(data_i, batch_size, minibatch_size, model, diffusion_model, vae, style_loss_fn=None, style_loss_weight=0.0, style_loss_t_max=300, model_type="unet"):
    
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
        
        # Encode images to latent space and normalize the latents for diffusion
        with torch.no_grad():
            minibatch = torch.mul(vae.encode(minibatch), 0.18215)

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

                # Denormalize latents and decode to image
                z0_denorm = torch.div(x0_hat[style_mask].float(), 0.18215)
                decoded_images = vae.decode(z0_denorm)
                decoded_images = torch.div(torch.add(decoded_images, 1.0), 2.0)
                
                # Compute style loss value
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

def training_procedure_with_repa_like(data_i, batch_size, minibatch_size, trainer):

    trainer.optimizer.zero_grad()
    
    total_loss = 0.0
    diffusion_loss = 0.0
    alignment_loss = 0.0
    samples = 0

    minibatch_num = torch.div(batch_size, minibatch_size)

    while samples < batch_size:

        # Load the next minibatch
        minibatch = next(data_i)["images"].to(device)
        
        # Perform backwards pass
        losses = trainer.minibatch_backward_step(minibatch, accumulation_steps=minibatch_num)

        total_loss = torch.add(total_loss, torch.mul(losses['loss_total'], minibatch_size))
        diffusion_loss = torch.add(diffusion_loss, torch.mul(losses['loss_diff'], minibatch_size))
        alignment_loss = torch.add(alignment_loss, torch.mul(losses['loss_repa'], minibatch_size))
        samples = torch.add(samples, minibatch_size)

    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    
    return torch.div(total_loss, samples), torch.div(diffusion_loss, samples), torch.div(alignment_loss, samples)

###############################################################################################
# Checkpoint paths
vae_checkpoint_path = f"../checkpoints/VAE_DSM_step_200000.pt"
training_checkpoint_path = f"../checkpoints/latent_diffusion_ddpm_test_checkpoint.pt"

# General settings
vae_type = "dsm" # Use kl, esm or dsm
noise_prediction_model = "sit_l_2" # Use unet or sit_l_2
alignment = "none" # Use none, repa, irepa or dog
alignment_lambda = 1.0 # Use 0.4 for REPA, 1.0 for iREPA and DoG

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
batch_size = 4
minibatch_size = 1 # Change based on what your GPU can handle :)
learning_rate = 1e-4
weight_decay = 1e-6

# Dataset settings
celeba_dataset_percentage = 0.35

# Iterative settings
save_checkpoint_every = 200
save_checkpoint_milestone_every = 5000
print_loss_every = 25
###############################################################################################

# Small checks for options that do not go together

if use_style_loss and alignment != "none":
    print("Currently style loss only works without alignment!")
assert not (use_style_loss and alignment != "none")

if (noise_mode != "white" or noise_strength != 1.0) and alignment != "none":
    print("Standard non-white noise only works without alignment!")
assert not ((noise_mode != "white" or noise_strength != 1.0) and alignment != "none")

device = "cuda" if torch.cuda.is_available() else "cpu"

if alignment != "none":

    # Wrap our VAE for REPAWrapper to work with our trained VAE
    vae_wrapper = VAE_wrapper(
        checkpoint_path=vae_checkpoint_path, 
        vae_type=vae_type,
        device=device, 
        compute_dtype=torch.bfloat16
    )

    # Options here an be changed, for our experiments these were fixed with thwe following settings
    config = ExperimentConfig(
        data_dir="../data",
        dataset_name="celeba",
        output_dir="../results/unet_dog",
        max_steps=30000,
        batch_size=batch_size,
        lr=learning_rate,
        model_type=noise_prediction_model,
        mode=alignment,
        lambda_repa=alignment_lambda,
        num_evals=40,
        num_eval_images=2000,
        vae_model_id="none"
    )

    # Build a trainer for the optimization
    student_model, meta = build_student_model(config.model_type)
    wrapper = REPAWrapper(student_model, meta, config, custom_vae=vae_wrapper)
    wrapper.student.train()
    trainer = DiffusionTrainer(wrapper, learning_rate=learning_rate, lambda_repa=alignment_lambda)

else:

    # Define DDPM diffusion model and put it on the device
    ddpm_model = Diffuser_DDPM_linear_schedule(total_timesteps=diffusion_T, beta_start=diffusion_beta_start, beta_end=diffusion_beta_end, noise_mode=noise_mode, noise_strength=noise_strength)
    ddpm_model.betas = ddpm_model.betas.to(device)
    ddpm_model.alphas = ddpm_model.alphas.to(device)
    ddpm_model.alpha_bars = ddpm_model.alpha_bars.to(device)

    # Load a VAE from checkpoint and put it on evaluation
    vae = VAE(mode=vae_type).to(device)
    checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint["vae"], strict=False)
    vae.eval()

    # Only load the VGGGramStyleLoss if we do style loss
    if use_style_loss:
        style_loss_fn = VGGGramStyleLoss(style_image_path=style_image_path, device=device, image_size=style_image_size).to(device)
        style_loss_fn.eval()

    # Load either a U-Net or a SiT
    if noise_prediction_model == "unet":

        config = DiffuserConfig()
        model = DiffusionUNet(config=config, model_in_channels=4, model_out_channels=4).to(device)

    elif noise_prediction_model == "sit_l_2":

        model = SiT_models['SiT-L/2'](input_size=32, in_channels=4, learn_sigma=False).to(device)

    # Initialize AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Either we start from iteration zero, or this gets overwritten when a checkpoint exists
start_iteration = 0

if os.path.exists(training_checkpoint_path):

    # Load checkpoint, with slightly different checkpoints depending on if we use a REPA-like
    checkpoint = torch.load(training_checkpoint_path, map_location=device, weights_only=False)
    
    if alignment != "none":
    
        wrapper.student.load_state_dict(checkpoint["student"]) # Prediction model
        trainer.optimizer.load_state_dict(checkpoint["optimizer"]) # Optimizer
        start_iteration = checkpoint["iteration"] + 1 # Start iteration

    else:

        model.load_state_dict(checkpoint["model"]) # Prediction model
        optimizer.load_state_dict(checkpoint["optimizer"]) # Optimizer
        start_iteration = checkpoint["iteration"] + 1 # Start iteration

train_loader = load_dataloader(minibatch_size, "celeba", percentage=celeba_dataset_percentage)
data_i = iter(train_loader)

# If we use a REPA-like alignment method:
if alignment != "none":

    for i in range(start_iteration, iterations):
        
        # Try to perform a training iteration, if it fails we presume we are out of training samples and retrieve new data
        try:
            total_loss, diffusion_loss, alignment_loss = training_procedure_with_repa_like(data_i, batch_size, minibatch_size, trainer)
        except StopIteration:
            data_i = iter(train_loader)
            total_loss, diffusion_loss, alignment_loss = training_procedure_with_repa_like(data_i, batch_size, minibatch_size, trainer)

        # Iterative functions
        if (i+1) % print_loss_every == 0:
            print(f"step {(i+1)} | loss {total_loss} | diff {diffusion_loss} | repa {alignment_loss}")

        save_condition_1 = (save_checkpoint_every and save_checkpoint_every > 0 and ((i+1) % save_checkpoint_every == 0) and i != start_iteration)
        save_condition_2 = (save_checkpoint_milestone_every and save_checkpoint_milestone_every > 0 and ((i+1) % save_checkpoint_milestone_every == 0) and i != start_iteration)
        
        if save_condition_1 or save_condition_2:

            checkpoint = {'iteration': i, 'student': wrapper.student.state_dict(), 'optimizer': trainer.optimizer.state_dict()}
        
            if save_condition_1:
                torch.save(checkpoint, training_checkpoint_path)

            if save_condition_2:
                torch.save(checkpoint, f"{training_checkpoint_path[0:-3]}_{(i+1)}_.pt")

# If we use style loss
elif alignment == "none" and use_style_loss:

    for i in range(start_iteration, iterations):

        # Autocast to bfloat16 for less demanding compute
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        
            # Try to perform a training iteration, if it fails we presume we are out of training samples and retrieve new data
            try:
                loss, mse_loss, style_loss = training_procedure_with_style_transfer(data_i, batch_size, minibatch_size, model, ddpm_model, vae, style_loss_fn=style_loss_fn, style_loss_weight=style_loss_weight, style_loss_t_max=style_loss_t_max, model_type=noise_prediction_model)
            except StopIteration:
                data_i = iter(train_loader)
                loss, mse_loss, style_loss = training_procedure_with_style_transfer(data_i, batch_size, minibatch_size, model, ddpm_model, vae, style_loss_fn=style_loss_fn, style_loss_weight=style_loss_weight, style_loss_t_max=style_loss_t_max, model_type=noise_prediction_model)

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
                loss = training_procedure_general(data_i, batch_size, minibatch_size, model, ddpm_model, vae, model_type=noise_prediction_model)
            except StopIteration:
                data_i = iter(train_loader)
                loss = training_procedure_general(data_i, batch_size, minibatch_size, model, ddpm_model, vae, model_type=noise_prediction_model)

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
