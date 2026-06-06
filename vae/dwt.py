import torch as th
from torch.nn import functional as F

class DWT:
    """2D Discrete wavelet transform (Haar)
    """
    def __init__(self, device):
        ll = th.tensor([[1, 1], [1, 1]]) / 2
        lh = th.tensor([[1, 1], [-1, -1]]) / 2
        hl = th.tensor([[1, -1], [1, -1]]) / 2
        hh = th.tensor([[1, -1], [-1, 1]]) / 2
 
        self.conv_kernel = th.stack([ll,lh,hl,hh]).unsqueeze(1).to(device)
        self.device = device

    def forward(self, X ):
        return F.conv2d(X, self.conv_kernel, stride=2)
 
    def inverse(self, Y ):
        return F.conv2d(Y, self.conv_kernel.T, stride=2)

