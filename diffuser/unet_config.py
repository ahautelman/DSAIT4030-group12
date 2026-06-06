from dataclasses import dataclass
from typing import Tuple

# this was for the smoke test
"""# toy version:
@dataclass
class DiffuserConfig:
    down_block_types: Tuple = ("Down", "Down")
    mid_block_types: str = "Mid"
    up_block_types: Tuple = ("Up", "Up")

    unet_channels_per_block: Tuple = (32, 64)
    unet_residual_layers_per_block: int = 1
    unet_up_down_factor: int = 2
    unet_up_down_kernel_size: int = 3

    time_embed_start_dim: int = 32
    time_embed_proj_dim: int = 64

    attention_head_dim: int = 8

    groupnorm_groups: int = 8
    norm_eps: float = 1e-6
    dropout: float = 0.0
"""

@dataclass
class DiffuserConfig:

    ###################
    ### UNET CONFIG ###
    ###################

    ### UNET Parts Config ###
    down_block_types: Tuple = ("AttnDown", "AttnDown", "AttnDown", "Down") # AttnDown/Down
    mid_block_types: str = "AttnMid"                                       # AttnMid/Mid
    up_block_types: Tuple = ("Up", "AttnUp", "AttnUp", "AttnUp")           # AttnUp/Up
    unet_channels_per_block: Tuple = (320, 640, 1280, 1280)
    unet_residual_layers_per_block: int = 2
    unet_up_down_factor: int = 2
    unet_up_down_kernel_size: int = 3

    ### Time Embeddings Config ###
    time_embed_start_dim: int = 320
    time_embed_proj_dim: int = 1280

    ### Attention Config ###
    attention_head_dim: int = 8

    ######################
    ### GENERAL CONFIG ###
    ######################
    groupnorm_groups: int = 32
    norm_eps: float = 1e-6
    dropout: float = 0.0
