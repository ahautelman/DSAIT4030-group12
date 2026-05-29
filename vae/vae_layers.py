import torch
import torch.nn as nn
import torch.nn.functional as F
from .vae_transformer import VAESelfAttention


class UpSampleBlock(nn.Module):
    """
    Spatially upsample a feature map by a given factor.

    (B, C, H, W) -> (B, C, H*factor, W*factor)

    Uses nearest-neighbor interpolation followed by a learnable convolution
    to refine the upsampled features. Channels are unchanged.

    Args:
        in_channels:     Number of input (and output) channels
        kernel_size:     Kernel size for the refinement convolution
        upsample_factor: Factor by which to increase spatial dimensions
    """

    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 upsample_factor=2):

        super(UpSampleBlock, self).__init__()

        self.factor = upsample_factor

        self.upsample = nn.Upsample(scale_factor=upsample_factor, mode="nearest")
        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=kernel_size,
                              stride=1,
                              padding="same")

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class DownSampleBlock(nn.Module):
    """
    Spatially downsample a feature map by a given factor.

    (B, C, H, W) -> (B, C, H/factor, W/factor)

    Uses a strided convolution so downsampling is learnable.
    Channels are unchanged.

    Args:
        in_channels:       Number of input (and output) channels
        kernel_size:       Kernel size for the strided convolution
        downsample_factor: Factor by which to reduce spatial dimensions
    """

    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 downsample_factor=2):

        super(DownSampleBlock, self).__init__()

        self.factor = downsample_factor

        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=kernel_size,
                              stride=downsample_factor,
                              padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    """
    Core residual block for the VAE: two convolutions with GroupNorm + SiLU,
    plus a skip connection from input to output.

    No time or class conditioning — VAE only.

    If in_channels != out_channels, a 1x1 conv aligns the skip connection.

    Args:
        in_channels:      Input channel count
        out_channels:     Output channel count
        dropout_p:        Dropout probability between the two convolutions
        groupnorm_groups: Number of groups in GroupNorm
        norm_eps:         Epsilon for GroupNorm stability
    """

    def __init__(self, in_channels, out_channels, dropout_p=0.0, groupnorm_groups=32, norm_eps=1e-6):

        super(ResidualBlock, self).__init__()

        self.norm1 = nn.GroupNorm(num_groups=groupnorm_groups,
                                  num_channels=in_channels,
                                  eps=norm_eps,
                                  affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding="same")

        self.norm2 = nn.GroupNorm(num_groups=groupnorm_groups,
                                  num_channels=out_channels,
                                  eps=norm_eps,
                                  affine=True)
        self.dropout = nn.Dropout(dropout_p)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding="same")

        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):

        residual = x

        # first conv block
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        # second conv block
        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)

        # residual connection
        x = x + self.skip(residual)

        return x
    
class VAEMidBlock(nn.Module):
    """
    Bottleneck block of the VAE encoder.

    This is the SD-VAE-style mid block:
        ResidualBlock
        Self-Attention
        ResidualBlock

    It does not change spatial resolution or channel count.

    Example:
        (B, 512, 32, 32) -> (B, 512, 32, 32)
    """

    def __init__(
        self,
        channels,
        num_attention_layers=1,
        dropout_p=0.0,
        groupnorm_groups=32,
        norm_eps=1e-6,
        attention_head_dim=None,
    ):
        super().__init__()

        if attention_head_dim is None:
            # This gives one attention head by default.
            # For channels=512, head_dim=512 -> num_heads=1.
            attention_head_dim = channels

        self.initial_residual = ResidualBlock(
            in_channels=channels,
            out_channels=channels,
            dropout_p=dropout_p,
            groupnorm_groups=groupnorm_groups,
            norm_eps=norm_eps,
        )

        self.attention_blocks = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        for _ in range(num_attention_layers):
            self.attention_blocks.append(
                VAESelfAttention(
                    in_channels=channels,
                    head_dim=attention_head_dim,
                    dropout_p=dropout_p,
                    groupnorm_groups=groupnorm_groups,
                    norm_eps=norm_eps,
                )
            )

            self.residual_blocks.append(
                ResidualBlock(
                    in_channels=channels,
                    out_channels=channels,
                    dropout_p=dropout_p,
                    groupnorm_groups=groupnorm_groups,
                    norm_eps=norm_eps,
                )
            )

    def forward(self, x):
        x = self.initial_residual(x)

        for attention, residual in zip(self.attention_blocks, self.residual_blocks):
            x = attention(x)
            x = residual(x)

        return x