import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from vae_encoder import VAEEncoder
from vae_decoder import VAEDecoder
from vae_losses import kl_loss, esm_loss, dsm_loss
from dsm_helpers import apply_dsm_mask


class VAE(nn.Module):
    """
    Variational Autoencoder for Latent Diffusion.

    Supports three training modes:
        - "kl"  : Standard SD-VAE with KL regularization
        - "esm" : Encoding Spectrum Matching (replaces KL with ESM loss)
        - "dsm" : Decoding Spectrum Matching (masked inputs + DSM loss)

    Default config is f8d4:
        - Input:   (B, 3, 256, 256)
        - Latent:  (B, 4, 32, 32)
        - Output:  (B, 3, 256, 256)

    Args:
        in_channels:              Number of input image channels (default 3)
        latent_channels:          Number of latent channels (default 4 for f8d4)
        out_channels:             Number of output image channels (default 3)
        channels_per_block:       Channel progression (default (128, 256, 512, 512))
        residual_layers_per_block: Number of residual blocks per level (default 2)
        num_attention_layers:     Number of attention layers in mid block (default 1)
        dropout_p:                Dropout probability (default 0.0)
        groupnorm_groups:         Number of GroupNorm groups (default 32)
        norm_eps:                 GroupNorm epsilon (default 1e-6)
        downsample_factor:        Spatial downsampling factor (default 2)
        downsample_kernel_size:   Kernel size for downsampling (default 3)
        upsample_factor:          Spatial upsampling factor (default 2)
        upsample_kernel_size:     Kernel size for upsampling (default 3)
        scale_factor:             Scale factor for latent (default 1.0)
        mode:                     Training mode: "kl", "esm", or "dsm" (default "kl")
        esm_delta:                ESM flattening factor delta (default 1.0)
        dsm_mask_n:               DSM triangular mask diagonal (default 8)
        dsm_block_size:           DSM DCT block size (default 8)
    """

    def __init__(
        self,
        in_channels=3,
        latent_channels=4,
        out_channels=3,
        channels_per_block=(64, 128, 256, 256),
        residual_layers_per_block=2,
        num_attention_layers=1,
        dropout_p=0.0,
        groupnorm_groups=32,
        norm_eps=1e-6,
        downsample_factor=2,
        downsample_kernel_size=3,
        upsample_factor=2,
        upsample_kernel_size=3,
        scale_factor=1.0,
        mode="kl",
        esm_delta=1.0,
        esm_mode="standard",
        esm_transform="dct",
        dsm_mask_n=8,
        dsm_block_size=8,
    ):
        super().__init__()

        assert mode in ["kl", "esm", "dsm"], \
            f"mode must be 'kl', 'esm', or 'dsm', got '{mode}'"

        self.latent_channels = latent_channels
        self.scale_factor = scale_factor
        self.mode = mode
        self.esm_delta = esm_delta
        self.esm_mode = esm_mode
        self.esm_transform = esm_transform
        self.dsm_mask_n = dsm_mask_n
        self.dsm_block_size = dsm_block_size
        self.dsm_mask_family = [0, 8, 10, 12]

        self.encoder = VAEEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            double_z=True,
            channels_per_block=channels_per_block,
            residual_layers_per_block=residual_layers_per_block,
            num_attention_layers=num_attention_layers,
            dropout_p=dropout_p,
            groupnorm_groups=groupnorm_groups,
            norm_eps=norm_eps,
            downsample_factor=downsample_factor,
            downsample_kernel_size=downsample_kernel_size,
        )

        self.decoder = VAEDecoder(
            latent_channels=latent_channels,
            out_channels=out_channels,
            channels_per_block=channels_per_block,
            residual_layers_per_block=residual_layers_per_block,
            num_attention_layers=num_attention_layers,
            dropout_p=dropout_p,
            groupnorm_groups=groupnorm_groups,
            norm_eps=norm_eps,
            upsample_factor=upsample_factor,
            upsample_kernel_size=upsample_kernel_size,
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        Allows gradients to flow through the sampling operation.

        Args:
            mu:     Mean of latent distribution (B, C, H, W)
            logvar: Log variance of latent distribution (B, C, H, W)

        Returns:
            z: Sampled latent (B, C, H, W)
        """
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x):
        """
        Encode image to latent space.
        Used during diffusion training — returns scaled latent.

        Args:
            x: Input image (B, 3, H, W) in [-1, 1]

        Returns:
            z: Scaled latent (B, latent_channels, h, w)
        """
        moments = self.encoder(x)
        mu, logvar = torch.chunk(moments, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return z * self.scale_factor

    def decode(self, z):
        """
        Decode latent back to image space.
        Used during diffusion inference — unscales latent first.

        Args:
            z: Scaled latent (B, latent_channels, h, w)

        Returns:
            x: Reconstructed image (B, 3, H, W) in [-1, 1]
        """
        z = z / self.scale_factor
        return self.decoder(z)

    def forward(self, x, discriminator=None, lpips_model=None):
        """
        Full forward pass for VAE training.

        Returns a dict with everything needed to compute losses:
            - reconstruction: decoded image
            - mu, logvar:     encoder statistics
            - z:              sampled latent
            - reg_loss:       KL, ESM, or DSM regularization loss
            - img_target:     target image (x or x_M for DSM)
            - recon_target:   reconstruction target (x_hat or x_hat_M for DSM)

        Args:
            x:             Input image (B, 3, H, W)
            discriminator: PatchGAN discriminator (needed for DSM)
            lpips_model:   LPIPS model (needed for DSM)
        """
        # encode
        moments = self.encoder(x)
        mu, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        z = self.reparameterize(mu, logvar)

        output = {"mu": mu, "logvar": logvar, "z": z}

        if self.mode == "kl":
            # standard VAE — decode full latent, KL regularization
            reconstruction = self.decoder(z)
            output["reconstruction"] = reconstruction
            output["img_target"] = x
            output["reg_loss"] = kl_loss(mu, logvar)

        elif self.mode == "esm":
            # ESM-AE — decode full latent, ESM regularization instead of KL
            reconstruction = self.decoder(z)
            output["reconstruction"] = reconstruction
            output["img_target"] = x
            output["reg_loss"] = esm_loss(x, z, delta=self.esm_delta, mode=self.esm_mode, transform=self.esm_transform)

        elif self.mode == "dsm":
            # DSM-AE — apply spectral mask to both x and z
            # decode masked latent, reconstruct masked image
            n = random.choice(self.dsm_mask_family)
            x_M, z_M = apply_dsm_mask(x, z, n=n, block_size=self.dsm_block_size)        
            reconstruction = self.decoder(z_M)
            output["reconstruction"] = reconstruction
            output["img_target"] = x_M
            output["z_masked"] = z_M

            # DSM has no KL or ESM — regularization comes from the masking itself
            output["reg_loss"] = torch.tensor(0.0, device=x.device)

        return output

    @property
    def last_layer(self):
        """
        Returns the last layer weights of the decoder.
        Used for adaptive weight computation in GAN training.
        """
        return self.decoder.conv_out.weight


if __name__ == "__main__":
    # test all three modes
    x = torch.randn(2, 3, 256, 256)

    for mode in ["kl", "esm", "dsm"]:
        vae = VAE(mode=mode)
        out = vae(x)
        print(f"\nMode: {mode}")
        print(f"  reconstruction: {out['reconstruction'].shape}")  # (2, 3, 256, 256)
        print(f"  img_target:     {out['img_target'].shape}")       # (2, 3, 256, 256)
        print(f"  reg_loss:       {out['reg_loss'].item():.4f}")

    # test encode/decode (for diffusion)
    vae = VAE()
    z = vae.encode(x)
    print(f"\nEncode: {x.shape} -> {z.shape}")   # (2, 4, 32, 32)
    x_hat = vae.decode(z)
    print(f"Decode: {z.shape} -> {x_hat.shape}") # (2, 3, 256, 256)
