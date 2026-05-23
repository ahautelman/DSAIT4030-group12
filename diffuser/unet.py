import torch
import torch.nn as nn
import torch.nn.functional as F

from diffuser.time_embedding import TimeEmbedder
from diffuser.diffuser_blocks import *

class DiffusionUNet(nn.Module):

    def __init__(self, config,
                 model_in_channels,
                 model_out_channels):
    
        super().__init__()

        # Init
        self.time_embedder = TimeEmbedder(config)
        block_channels = list(config.unet_channels_per_block)

        # input layer
        self.init_conv = nn.Conv2d(
            in_channels=model_in_channels,
            out_channels=block_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # ------------------------
        # Down path 
        # -------------------------
        self.down_blocks = nn.ModuleList()
        current_channels = block_channels[0]

        for i, out_channels in enumerate(block_channels):
            block_type = config.down_block_types[i]
            use_attention = "Attn" in block_type
            downsample = (i != len(block_channels) - 1)

            self.down_blocks.append(
                UNetDownBlock(
                    config=config,
                    in_channels=current_channels,
                    out_channels=out_channels,
                    num_res_layers=config.unet_residual_layers_per_block,
                    use_attention=use_attention,
                    downsample=downsample
                )
            )

            current_channels = out_channels

        # ------------------------
        # Bottleneck
        # -------------------------
        self.mid_block = UNetBottleneckBlock(
            config=config,
            channels=block_channels[-1],
            num_res_layers=config.unet_residual_layers_per_block,
            use_attention="Attn" in config.mid_block_types
        )

        # ------------------------
        # Up Path 
        # -------------------------
        self.up_blocks = nn.ModuleList()
        skip_channels_list = list(reversed(block_channels))
        current_channels = block_channels[-1]

        for i, block_type in enumerate(config.up_block_types):
            use_attention="Attn" in config.up_block_types[i]
            skip_channels = skip_channels_list[i]
            out_channels = skip_channels

            # The last block does not upsample
            upsample = (i != len(skip_channels_list) - 1)

            self.up_blocks.append(
                UNetUpBlock(
                    config=config,
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    num_res_layers=config.unet_residual_layers_per_block,
                    use_attention=use_attention,
                    upsample=upsample,
                )
            )

            current_channels = out_channels

        # Output layers
        self.out_norm = nn.GroupNorm(
            num_groups=config.groupnorm_groups,
            num_channels=block_channels[0],
            eps=config.norm_eps,
            affine=True
        )
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(
            in_channels=block_channels[0],
            out_channels=model_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, t):
        '''
        Takes an image x and timestep t and runs it through the Unet to learn 
        the conditional distribution of noise in the image. 

        :param x:   input image 
        :param t:   shape (B,)
        '''

        # Get time embeddings
        time_embedding = self.time_embedder(t)
        skips = []

        # input
        x = self.init_conv(x)

        # down
        for down_block in self.down_blocks:
            x, skip = down_block(x, time_embedding)
            skips.append(skip)
        
        # bottleneck
        x = self.mid_block(x, time_embedding)

        # up
        for up_block in self.up_blocks:
            skip = skips.pop()
            x = up_block(x, skip, time_embedding)

        # output
        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x)

        return x

#################
# Unet blocks
#################

class UNetDownBlock(nn.Module):
    '''
    A single down block for the Unet. 
    '''

    def __init__(self, config, in_channels, out_channels, 
                 use_attention=False, 
                 num_res_layers=2, 
                 downsample=True):

        super(UNetDownBlock, self).__init__()

        # First residual layer, changes #channels
        self.resnets = nn.ModuleList([
            DiffuserResidualBlock(
                    config=config,
                    in_channels=in_channels,
                    out_channels=out_channels
        )])    

        # Rest of 'num_res_layers' residual blocks 
        self.resnets.extend([
            DiffuserResidualBlock(
                    config=config,
                    in_channels=out_channels,
                    out_channels=out_channels
        ) for _ in range(num_res_layers - 1)])    

        # Possible Self attention block
        self.attn = (
            DiffuserAttentionBlock(in_channels=out_channels, config=config)
            if use_attention 
            else nn.Identity()
        )

        # Possible Downsample block 
        self.downsample = (
            DownSampleBlock(in_channels=out_channels, config=config)
            if downsample 
            else nn.Identity()
        )

    def forward(self, x, time_embedding):
        for resnet in self.resnets:
            x = resnet.forward(x, time_embedding)
        
        x = self.attn(x)
        skip = x
        x = self.downsample(x)

        return x, skip


class UNetBottleneckBlock(nn.Module):

    def __init__(self, config, channels,
                 use_attention=False, 
                 num_res_layers=2 ):

        super().__init__()

        # First residual layer, which sees skip layers
        self.resnets = nn.ModuleList([
            DiffuserResidualBlock(
                    config=config,
                    in_channels=channels,
                    out_channels=channels
        )])    

        # Extend with list of 'num_res_layers - 1' residual blocks 
        self.resnets.extend([
            DiffuserResidualBlock(
                    config=config,
                    in_channels=channels,
                    out_channels=channels
        ) for _ in range(num_res_layers - 1)])    

        # Possible Self attention block
        self.attn = (
            DiffuserAttentionBlock(in_channels=channels, config=config)
            if use_attention 
            else nn.Identity()
        )

    def forward(self, x, time_embedding):
        for idx, resnet in enumerate(self.resnets):
            x = resnet(x, time_embedding)
            if idx == 0:
                x = self.attn(x)

        return x
    
class UNetUpBlock(nn.Module):

    
    def __init__(self, config, in_channels, skip_channels, out_channels, 
                 use_attention=False, 
                 num_res_layers=2, 
                 upsample=True):

        super().__init__()

        # First residual layer, which sees skip layers
        self.resnets = nn.ModuleList([
            DiffuserResidualBlock(
                    config=config,
                    in_channels=in_channels + skip_channels,
                    out_channels=out_channels
        )])    

        # Extend with list of 'num_res_layers - 1' residual blocks 
        self.resnets.extend([
            DiffuserResidualBlock(
                    config=config,
                    in_channels=out_channels,
                    out_channels=out_channels
        ) for _ in range(num_res_layers - 1)])    

        # Possible Self attention block
        self.attn = (
            DiffuserAttentionBlock(in_channels=out_channels, config=config)
            if use_attention 
            else nn.Identity()
        )

        # Possible Downsample block 
        self.upsample = (
            UpSampleBlock(in_channels=out_channels, config=config)
            if upsample 
            else nn.Identity()
        )

    def forward(self, x, skip, time_embedding):

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)

        for resnet in self.resnets:
            x = resnet.forward(x, time_embedding)

        x = self.attn(x)
        x = self.upsample(x)

        return x