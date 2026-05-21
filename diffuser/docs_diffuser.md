# Documentation for usage of the UNET implementation
## Creation and application
The entire UNET can be created and controlled with two files: 

- unet.py: contains the class 'DiffusionUNet' for the unet architecture
- unet_config.py: dataclass 'DiffuserConfig' describing the configuration of the UNet

The class 'DiffusionUNet' takes three arguments: 

- config: an instance of the 'DiffuserConfig' class 
- model_in_channels: number of input channels to the unet
- model_out_channels: number of output channels of the unet

## Config description
As described, the config file contains all variables for the unet. Each variable controls a part of the architecture. These variables and their choices were based on the original Spectrum Matching paper, such that their config file can be used in this place. 

### UNET Parts Config
- down_block_types: block types for each stage in the downsample part of the network. "AttnDown" uses self-attention, "Down" is plain downsample.
- mid_block_types: Block type for the bottleneck. "AttnMid" enables attention in the middle.                            
- up_block_types: block types for each stage in the upsample part of the network. "AttnUp" uses self-attention. 
- unet_channels_per_block: Output channel count for each block level. 
- unet_residual_layers_per_block: Number of residual (DiffuserResidualBlock) layers inside each block.
- unet_up_down_factor: Spatial up/down sampling factor between successive resolutions (e.g., 2 = half/ double spatial size).
- unet_up_down_kernel_size: Kernel size used by the up/down sampling convolutions.

### Time Embeddings Config
- time_embed_start_dim: Base dimension for time embeddings produced by TimeEmbedder.
- time_embed_proj_dim: Projected time-embedding dimension fed into residual blocks.

### Attention Config
- attention_head_dim: Per-head embedding size for self-attention. Number of heads = channels / attention_head_dim.

### General Config
- groupnorm_groups: number of groups for groupNorm layers
- norm_eps: added in groupNorm for stability
- dropout: the dropout probability inside residual and attention blocks

## Blocks
Below are the building blocks of the UNET, with their respective class names: 

### Base blocks
- DiffuserResidualBlock: Base residual block
- DownSampleBlock: Base downsample block
- UpSampleBlock: Base upsample block
- DiffuserAttentionBlock: Base attention block

### Higher level blocks
- UNetDownBlock: 
    - unet_residual_layers_per_block * DiffuserResidualBlock
    - Possible self attention block based on "Attndown"/"Down" block in down_block_types
    - Downsample block
- UNetBottleneckBlock: 
    - unet_residual_layers_per_block * DiffuserResidualBlock
    - Possible self attention block based on "AttnMid"/"Mid" block in mid_block_types
- UNetUpBlock: 
    - unet_residual_layers_per_block * DiffuserResidualBlock
    - Possible self attention block based on "AttnUp"/"Up" block in up_block_types
    - Upsample block

### UNET 
- Creates UNetDownBlock based on unet_channels_per_block variable and stores skip connection values.
- Creates a single UNetBottleneckBlock.
- Create UNetUpBlock based on unet_channels_per_block variable and uses the skip connections.
- Ends with a single GroupNorm, SiLU and Conv2d block. 