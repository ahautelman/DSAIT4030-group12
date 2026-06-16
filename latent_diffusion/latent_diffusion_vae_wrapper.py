import os
import sys
from sys import platform
from types import SimpleNamespace
import torch

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

from repa.models.vae import BaseVAEWrapper
from vae.vae import VAE

# Class based on repa.models.vae.BaseVAEWrapper 
class VAE_wrapper(BaseVAEWrapper):
    def __init__(self, checkpoint_path: str, vae_type: str, device: str, compute_dtype: torch.dtype):
        super().__init__()
        
        self.vae_model = VAE(mode=vae_type).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.vae_model.load_state_dict(checkpoint["vae"], strict=False)
        
        self.vae_model.to(compute_dtype)
        self.vae_model.eval()

        for param in self.vae_model.parameters():
            param.requires_grad = False

    def encode(self, x: torch.Tensor):
        raw_latents = self.vae_model.encode(x)
        # Trickery since something in the REPA code expects a latent distribution component, which we fake like this.
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: raw_latents))

    def decode(self, z: torch.Tensor):
        return self.vae_model.decode(z)

    @property
    def scaling_factor(self) -> float:
        return 0.18215

    @property
    def latent_channels(self) -> int:
        return 4