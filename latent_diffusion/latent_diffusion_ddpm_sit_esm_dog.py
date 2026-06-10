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

from dataset_loader import load_dataset

from diffuser.unet import DiffusionUNet

from repa.config import ExperimentConfig
from repa.models.wrapper import REPAWrapper
from repa.train import DiffusionTrainer
from repa.models.vae import BaseVAEWrapper
from repa.models.factory import build_student_model

from vae.vae import VAE

# Config:
from diffuser.unet_config import DiffuserConfig

class MockLatentDist:
    def __init__(self, tensor):
        self.tensor = tensor
    def sample(self):
        return self.tensor

class MockVAEOutput:
    def __init__(self, tensor):
        self.latent_dist = MockLatentDist(tensor)

class CustomVAEWrapper(BaseVAEWrapper):
    def __init__(self, checkpoint_path: str, device: str, compute_dtype: torch.dtype):
        super().__init__()
        
        self.vae_model = VAE(mode="esm").to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.vae_model.load_state_dict(checkpoint["vae"], strict=False)
        
        self.vae_model.to(compute_dtype)
        self.vae_model.eval()
        for param in self.vae_model.parameters():
            param.requires_grad = False

    def encode(self, x: torch.Tensor):
        raw_latents = self.vae_model.encode(x)
        return MockVAEOutput(raw_latents)

    def decode(self, z: torch.Tensor):
        return self.vae_model.decode(z)

    @property
    def scaling_factor(self) -> float:
        return 0.18215

    @property
    def latent_channels(self) -> int:
        return 4

def save_checkpoint(iteration, checkpoint_path):
    
    checkpoint = {
        'iteration': iteration,
        'student': wrapper.student.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
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

def train_step(data_iter, batch_size, trainer):

    trainer.optimizer.zero_grad(set_to_none=True)
    
    total_loss = 0.0
    diff_loss = 0.0
    repa_loss = 0.0

    data_samples = 0

    while data_samples < batch_size:

        batch = next(data_iter)

        data_minibatch = batch["images"].to(device, non_blocking=True)

        B = data_minibatch.shape[0]
        accumulation_steps = batch_size / B
        
        losses = trainer.minibatch_backward_step(data_minibatch, accumulation_steps=accumulation_steps)

        total_loss += losses['loss_total'] * B
        diff_loss += losses['loss_diff'] * B
        repa_loss += losses['loss_repa'] * B

        data_samples += B

    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    trainer.optimizer.zero_grad(set_to_none=True)
    
    return total_loss / max(data_samples, 1), diff_loss / max(data_samples, 1), repa_loss / max(data_samples, 1)

#############################################################################
# Set the path to the VAE checkpoint
checkpoint_dir = "../checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
vae_checkpoint_path = f"{checkpoint_dir}/VAE_ESM_step_200000.pt"
diffusion_checkpoint_path = f"{checkpoint_dir}/latent_diffusion_ddpm_sit_esm_repa_checkpoint.pt"

iterations = 10000
batch_size = 128
minibatch_size = 64
num_workers = 2

save_checkpoint_every = 100
save_checkpoint_milestone_every = 5000
print_loss_every = 25
#############################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
np.random.seed(0)

# Config for REPA-DoG
config = ExperimentConfig(
    data_dir="../data",
    dataset_name="celeba",
    output_dir="../results/unet_dog",
    max_steps=30000,
    batch_size=batch_size,
    lr=1e-4,
    model_type="sit_l_2",
    mode="dog",
    lambda_repa=1.0,
    num_evals=40,
    num_eval_images=2000,
    vae_model_id="none"
)

custom_vae_wrapper = CustomVAEWrapper(
    checkpoint_path=vae_checkpoint_path, 
    device=device, 
    compute_dtype=torch.bfloat16
)

student_model, meta = build_student_model(config.model_type)

wrapper = REPAWrapper(student_model, meta, config, custom_vae=custom_vae_wrapper)
wrapper.student.train()

start_iteration = 0

if os.path.exists(diffusion_checkpoint_path):

    checkpoint = torch.load(diffusion_checkpoint_path, map_location=device, weights_only=False)
    wrapper.student.load_state_dict(checkpoint["student"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    start_iteration = checkpoint["iteration"] + 1

    print(f"Loaded checkpoint and starting from iteration {start_iteration}!")

train_loader = load_data(num_workers, minibatch_size, "celeba", percentage=0.35)
data_i = iter(train_loader)

# Use start_step from checkpoint (if resumed) otherwise 0
for i in range(start_iteration, iterations):
    
    try:
        total_loss, diff_loss, repa_loss = train_step(data_i, batch_size, trainer)

    except:
        data_i = iter(train_loader)
        total_loss, diff_loss, repa_loss = train_step(data_i, batch_size, trainer)

    if i % print_loss_every == 0:
        print(f"step {i:5d} | loss {total_loss:.4f} | diff {diff_loss:.4f} | repa {repa_loss:.4f}")

    # Periodic checkpoint save
    if save_checkpoint_every and save_checkpoint_every > 0 and (i % save_checkpoint_every == 0) and i != start_iteration:
        save_checkpoint(i, diffusion_checkpoint_path)

    if save_checkpoint_milestone_every and save_checkpoint_milestone_every > 0 and (i % save_checkpoint_milestone_every == 0) and i != start_iteration:
        save_checkpoint(i, f"{diffusion_checkpoint_path[0:-3]}_{i}_.pt")
