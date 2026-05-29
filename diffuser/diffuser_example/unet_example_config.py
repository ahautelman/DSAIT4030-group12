from dataclasses import dataclass
from typing import Tuple

@dataclass
class DiffuserConfig:

    ###################
    ### UNET CONFIG ###
    ###################

    ### UNET Parts Config ###
    down_block_types: Tuple = ("AttnDown", "AttnDown", "AttnDown") # AttnDown/Down
    mid_block_types: str = "AttnMid"                                       # AttnMid/Mid
    up_block_types: Tuple = ("Up", "AttnUp", "AttnUp")           # AttnUp/Up
    unet_channels_per_block = (64, 128, 256)
    unet_residual_layers_per_block: int = 2
    unet_up_down_factor: int = 2
    unet_up_down_kernel_size: int = 3

    ### Time Embeddings Config ###
    time_embed_start_dim: int = 64
    time_embed_proj_dim: int = 256

    ### Attention Config ###
    attention_head_dim: int = 64

    ######################
    ### GENERAL CONFIG ###
    ######################
    groupnorm_groups: int = 32
    norm_eps: float = 1e-6
    dropout: float = 0.0