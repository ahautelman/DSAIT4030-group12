import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffuserAttentionBlock(nn.Module):
    '''
    Self-attention block for the diffuser.

    Takes a 2D feature map (B, C, H, W), flattens spatial dimensions
    into a sequence, applies self-attention across all spatial positions,
    then reshapes back to (B, C, H, W).

    Uses GroupNorm before attention and a residual connection after.

    Args:
        in_channels:      Number of input channels (acts as embedding dim)
        config:   
            attention_head_dim:     Embedding dimension per attention head
            dropout:              Dropout probability in attention
            groupnorm_groups:       Number of groups in GroupNorm 
            norm_eps:               Epsilon for GroupNorm stability
    '''

    def __init__(self,
                 config,
                 in_channels):

        super(DiffuserAttentionBlock, self).__init__()

        assert in_channels % config.groupnorm_groups == 0
        assert in_channels % config.attention_head_dim == 0, f"in_channels ({in_channels}) must be divisible by head_dim ({head_dim})"

        self.in_channels = in_channels
        self.head_dim = config.attention_head_dim
        self.num_heads = in_channels // config.attention_head_dim
        self.dropout = config.dropout

        self.norm = nn.GroupNorm(num_groups=config.groupnorm_groups,
                                 num_channels=in_channels,
                                 eps=config.norm_eps)

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)

        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):

        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        x = x.reshape(B, C, H * W).transpose(1, 2)
        seq_len = H * W

        q = self.q_proj(x)  # (B, H*W, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention( q, k, v, dropout_p=self.dropout if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, seq_len, C)
        attn_out = self.out_proj(attn_out)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)

        return attn_out + residual
    
class UpSampleBlock(nn.Module):
    '''
    Spatially upsample a feature map by a given factor.

    (B, C, H, W) -> (B, C, H*factor, W*factor)

    Uses nearest-neighbor interpolation followed by a learnable convolution
    to refine the upsampled features. Channels are unchanged.

    Args:
        in_channels:     Number of input (and output) channels
        config:
            kernel_size:     Kernel size for the refinement convolution
            upsample_factor: Factor by which to increase spatial dimensions
    '''

    def __init__(self,
                 config,
                 in_channels):

        super(UpSampleBlock, self).__init__()

        self.factor = config.unet_up_down_factor

        self.upsample = nn.Upsample(scale_factor=config.unet_up_down_factor, mode="nearest")
        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=config.unet_up_down_kernel_size,
                              stride=1,
                              padding="same")

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class DownSampleBlock(nn.Module):
    '''
    Spatially downsample a feature map by a given factor.

    (B, C, H, W) -> (B, C, H/factor, W/factor)

    Uses a strided convolution so downsampling is learnable.
    Channels are unchanged.

    Args:
        in_channels:       Number of input (and output) channels
        config: 
            kernel_size:       Kernel size for the strided convolution
            downsample_factor: Factor by which to reduce spatial dimensions
    '''

    def __init__(self, 
                 config,
                 in_channels):

        super(DownSampleBlock, self).__init__()

        self.factor = config.unet_up_down_factor

        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=config.unet_up_down_kernel_size,
                              stride=config.unet_up_down_factor,
                              padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class DiffuserResidualBlock(nn.Module):
    '''
    Residual block for the diffuser: two convolutions with GroupNorm + SiLU,
    plus a skip connection from input to output. Uses time conditioning.

    If in_channels != out_channels, a 1x1 conv aligns the skip connection.

    Args:
        in_channels:      Input channel count
        out_channels:     Output channel count
        config:
            dropout:        Dropout probability between the two convolutions
            groupnorm_groups: Number of groups in GroupNorm
            norm_eps:         Epsilon for GroupNorm stability
    '''

    def __init__(self, config, in_channels, out_channels):

        super(DiffuserResidualBlock, self).__init__()

        self.time_projection = nn.Linear(
            config.time_embed_proj_dim,
            out_channels
        )

        # Assert shapes
        assert in_channels % config.groupnorm_groups == 0
        assert out_channels % config.groupnorm_groups == 0

        self.norm1 = nn.GroupNorm(num_groups=config.groupnorm_groups,
                                  num_channels=in_channels,
                                  eps=config.norm_eps,
                                  affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding="same")

        self.norm2 = nn.GroupNorm(num_groups=config.groupnorm_groups,
                                  num_channels=out_channels,
                                  eps=config.norm_eps,
                                  affine=True)
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding="same")
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, time_embedding):

        assert time_embedding.ndim == 2, (f"time_embedding must be (B, D), got {time_embedding.shape}")
        residual = x

        # first conv block
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        # Time embedding
        t = self.time_projection(time_embedding)
        assert t.ndim == 2, (f"Projected time embedding must be (B, D), got {t.shape}")
        t = t[:, :, None, None]
        x = x + t

        # second conv block
        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)

        # residual connection
        x = x + self.skip(residual)

        return x