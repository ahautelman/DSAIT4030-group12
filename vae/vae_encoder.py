import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae_layers import ResidualBlock, DownSampleBlock, VAEMidBlock
from .vae_transformer import VAESelfAttention



class EncoderBlock(nn.Module):
    """
    One resolution level of the VAE encoder.

    Structure:
        ResidualBlock(s)
        optional DownSampleBlock

    The residual blocks can change the number of channels.
    The downsampling block reduces H and W by downsample_factor,
    while keeping the number of channels fixed.

    Example:
        (B, 128, 256, 256)
        -> residual blocks
        -> (B, 128, 256, 256)
        -> downsample
        -> (B, 128, 128, 128)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_residual_blocks=2,
        add_downsample=True,
        downsample_factor=2,
        downsample_kernel_size=3,
        dropout_p=0.0,
        groupnorm_groups=32,
        norm_eps=1e-6,
    ):
        super().__init__()

        self.residual_blocks = nn.ModuleList()

        for i in range(num_residual_blocks):
            block_in_channels = in_channels if i == 0 else out_channels

            self.residual_blocks.append(
                ResidualBlock(
                    in_channels=block_in_channels,
                    out_channels=out_channels,
                    dropout_p=dropout_p,
                    groupnorm_groups=groupnorm_groups,
                    norm_eps=norm_eps,
                )
            )

        if add_downsample:
            self.downsample = DownSampleBlock(
                in_channels=out_channels,
                kernel_size=downsample_kernel_size,
                downsample_factor=downsample_factor,
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        for block in self.residual_blocks:
            x = block(x)

        x = self.downsample(x)
        return x


class VAEEncoder(nn.Module):
    """
    SD-VAE-style convolutional encoder.

    Default architecture for 256x256 input:

        Input:
            (B, 3, 256, 256)

        conv_in:
            3 -> 128

        encoder blocks:
            128 -> 128, downsample: 256 -> 128
            128 -> 256, downsample: 128 -> 64
            256 -> 512, downsample: 64 -> 32
            512 -> 512, no downsample

        mid block:
            residual + self-attention + residual

        output:
            GroupNorm + SiLU + Conv2d

        If double_z=True and latent_channels=4:
            output has 8 channels:
                first 4  = mu
                second 4 = logvar

        Output:
            (B, 8, 32, 32) for f8d4
    """

    def __init__(
        self,
        in_channels=3,
        latent_channels=4,
        double_z=True,
        channels_per_block=(64, 128, 256, 256),
        residual_layers_per_block=2,
        num_attention_layers=1,
        dropout_p=0.0,
        groupnorm_groups=32,
        norm_eps=1e-6,
        downsample_factor=2,
        downsample_kernel_size=3,
        attention_head_dim=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.double_z = double_z
        self.channels_per_block = channels_per_block

        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels_per_block[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.encoder_blocks = nn.ModuleList()

        current_channels = channels_per_block[0]

        for i, out_channels in enumerate(channels_per_block):
            is_final_block = i == len(channels_per_block) - 1

            self.encoder_blocks.append(
                EncoderBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    num_residual_blocks=residual_layers_per_block,
                    add_downsample=not is_final_block,
                    downsample_factor=downsample_factor,
                    downsample_kernel_size=downsample_kernel_size,
                    dropout_p=dropout_p,
                    groupnorm_groups=groupnorm_groups,
                    norm_eps=norm_eps,
                )
            )

            current_channels = out_channels

        self.mid_block = VAEMidBlock(
            channels=channels_per_block[-1],
            num_attention_layers=num_attention_layers,
            dropout_p=dropout_p,
            groupnorm_groups=groupnorm_groups,
            norm_eps=norm_eps,
            attention_head_dim=attention_head_dim,
        )

        self.out_norm = nn.GroupNorm(
            num_groups=groupnorm_groups,
            num_channels=channels_per_block[-1],
            eps=norm_eps,
            affine=True,
        )

        out_channels = 2 * latent_channels if double_z else latent_channels

        self.conv_out = nn.Conv2d(
            in_channels=channels_per_block[-1],
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = self.conv_in(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.mid_block(x)

        x = self.out_norm(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


if __name__ == "__main__":
    encoder = VAEEncoder(
        in_channels=3,
        latent_channels=4,
        double_z=True,
        channels_per_block=(64, 128, 256, 256),
        residual_layers_per_block=2,
        num_attention_layers=1,
    )

    x = torch.randn(2, 3, 256, 256)
    y = encoder(x)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)

    mu, logvar = torch.chunk(y, chunks=2, dim=1)
    print("mu shape:    ", mu.shape)
    print("logvar shape:", logvar.shape)
