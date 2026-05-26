import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class BaseVAEWrapper(nn.Module):
    """Lightweight abstraction to support custom VAE plugging in the future."""

    def encode(self, x: torch.Tensor):
        raise NotImplementedError

    def decode(self, z: torch.Tensor):
        raise NotImplementedError

    @property
    def scaling_factor(self) -> float:
        raise NotImplementedError

    @property
    def latent_channels(self) -> int:
        raise NotImplementedError


class DiffusersVAEWrapper(BaseVAEWrapper):
    """Standard VAE wrapper using Diffusers AutoencoderKL."""

    def __init__(self, model_id: str, compute_dtype: torch.dtype):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=compute_dtype)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    def encode(self, x: torch.Tensor):
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor):
        return self.vae.decode(z)

    @property
    def scaling_factor(self) -> float:
        return self.vae.config.scaling_factor

    @property
    def latent_channels(self) -> int:
        return self.vae.config.latent_channels