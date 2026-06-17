import torch
import torch.nn as nn
import torch.nn.functional as F


class VAESelfAttention(nn.Module):
    """
    Self-attention module for the VAE bottleneck.

    Takes a 2D feature map (B, C, H, W), flattens spatial dimensions
    into a sequence, applies self-attention across all spatial positions,
    then reshapes back to (B, C, H, W).

    Uses GroupNorm before attention and a residual connection after.

    Args:
        in_channels:      Number of input channels (acts as embedding dim)
        head_dim:         Embedding dimension per attention head
        dropout_p:        Dropout probability in attention
        groupnorm_groups: Number of groups in GroupNorm
        norm_eps:         Epsilon for GroupNorm stability
    """

    def __init__(self,
                 in_channels,
                 head_dim=1,
                 dropout_p=0.0,
                 groupnorm_groups=32,
                 norm_eps=1e-6):

        super(VAESelfAttention, self).__init__()

        assert in_channels % head_dim == 0, \
            f"in_channels ({in_channels}) must be divisible by head_dim ({head_dim})"

        self.in_channels = in_channels
        self.head_dim = head_dim
        self.num_heads = in_channels // head_dim
        self.dropout_p = dropout_p

        self.norm = nn.GroupNorm(num_groups=groupnorm_groups,
                                 num_channels=in_channels,
                                 eps=norm_eps)

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

        attn_out = F.scaled_dot_product_attention( q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, seq_len, C)
        attn_out = self.out_proj(attn_out)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)

        return attn_out + residual