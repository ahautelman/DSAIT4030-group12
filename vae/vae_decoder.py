import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_layers import ResidualBlock, UpSampleBlock, VAEMidBlock
from vae_transformer import VAESelfAttention


class DecoderBlock(nn.Module):
    """
    One resolution level of the VAE decoder.

    Structure:
        ResidualBlock(s)
        optional UpSampleBlock

    Example:
        (B, 512, 32, 32)
        -> residual blocks
        -> (B, 256, 32, 32)
        -> upsample
        -> (B, 256, 64, 64)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_residual_blocks=2,
        add_upsample=True,
        upsample_factor=2,
        upsample_kernel_size=3,
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

        if add_upsample:
            self.upsample = UpSampleBlock(
                in_channels=out_channels,
                kernel_size=upsample_kernel_size,
                upsample_factor=upsample_factor,
            )
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        for block in self.residual_blocks:
            x = block(x)
        x = self.upsample(x)
        return x
    

class VAEDecoder(nn.Module):
    """
    SD-VAE-style convolutional decoder.

    Default architecture for 32x32 latent -> 256x256 output:

        Input:
            (B, 4, 32, 32)

        conv_in:
            4 -> 512

        mid block:
            residual + self-attention + residual

        decoder blocks:
            512 -> 512, upsample: 32 -> 64
            512 -> 256, upsample: 64 -> 128
            256 -> 128, upsample: 128 -> 256
            128 -> 128, no upsample

        output:
            GroupNorm + SiLU + Conv2d -> 3 channels (RGB)

        Output:
            (B, 3, 256, 256)
    """

    def __init__(
        self,
        latent_channels=4,
        out_channels=3,
        channels_per_block=(128, 256, 512, 512),
        residual_layers_per_block=2,
        num_attention_layers=1,
        dropout_p=0.0,
        groupnorm_groups=32,
        norm_eps=1e-6,
        upsample_factor=2,
        upsample_kernel_size=3,
        attention_head_dim=None,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.out_channels = out_channels

        # reverse channels for decoder
        channels_per_block = list(reversed(channels_per_block))

        self.conv_in = nn.Conv2d(
            in_channels=latent_channels,
            out_channels=channels_per_block[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if attention_head_dim is None:
            attention_head_dim = channels_per_block[0]

        self.mid_block = VAEMidBlock(
            channels=channels_per_block[0],
            num_attention_layers=num_attention_layers,
            dropout_p=dropout_p,
            groupnorm_groups=groupnorm_groups,
            norm_eps=norm_eps,
            attention_head_dim=attention_head_dim,
        )

        self.decoder_blocks = nn.ModuleList()
        current_channels = channels_per_block[0]

        for i, out_ch in enumerate(channels_per_block):
            is_final_block = i == len(channels_per_block) - 1

            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    num_residual_blocks=residual_layers_per_block + 1,
                    add_upsample=not is_final_block,
                    upsample_factor=upsample_factor,
                    upsample_kernel_size=upsample_kernel_size,
                    dropout_p=dropout_p,
                    groupnorm_groups=groupnorm_groups,
                    norm_eps=norm_eps,
                )
            )

            current_channels = out_ch

        self.out_norm = nn.GroupNorm(
            num_groups=groupnorm_groups,
            num_channels=channels_per_block[-1],
            eps=norm_eps,
            affine=True,
        )

        self.conv_out = nn.Conv2d(
            in_channels=channels_per_block[-1],
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)

        for block in self.decoder_blocks:
            x = block(x)

        x = self.out_norm(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


if __name__ == "__main__":
    decoder = VAEDecoder(
        latent_channels=4,
        out_channels=3,
        channels_per_block=(128, 256, 512, 512),
        residual_layers_per_block=2,
        num_attention_layers=1,
    )

    x = torch.randn(2, 4, 32, 32)
    y = decoder(x)

    print("Input shape: ", x.shape)   # (2, 4, 32, 32)
    print("Output shape:", y.shape)   # (2, 3, 256, 256)