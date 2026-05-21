import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Define MLP for mapping the time embedding to a useful format for the diffuser
        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(config.time_embed_start_dim, config.time_embed_proj_dim),
            nn.SiLU(),
            nn.Linear(config.time_embed_proj_dim, config.time_embed_proj_dim),
            nn.SiLU()
        )

    def forward(self, timesteps):
        embedding = self.get_time_embedding(timesteps, self.config.time_embed_start_dim)
        return self.time_embedding_mlp(embedding)
        
    def get_time_embedding(self, timesteps, dim):
        '''
        Returns Sinusoidal time embedding

        :param timesteps: shape (B,) integer or float tensor
        :param dim: size of second dimension of the return tensor, must be even
        
        returns shape (B, dim)
        '''

        assert dim % 2 == 0 

        device = timesteps.device
        half_dim = dim // 2

        # Frequencies
        exponent = torch.arange(half_dim, device=device, dtype=torch.float32)
        exponent = -math.log(10000.0) * exponent / max((half_dim - 1),1)
        freqs = torch.exp(exponent)

        # Broadcast: (B, 1) * (1, half_dim) -> (B, half_dim)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)

        encoding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        return encoding