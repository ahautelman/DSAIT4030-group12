import torch as th
import torch.nn.functional as F

class DWT:
    """2D Discrete wavelet transform (Haar)
    """
    def __init__(self, shape, device):
        ll = th.tensor([[1, 1], [1, 1]]) / 2
        lh = th.tensor([[1, 1], [-1, -1]]) / 2
        hl = th.tensor([[1, -1], [1, -1]]) / 2
        hh = th.tensor([[1, -1], [-1, 1]]) / 2
 
        self.channels = shape[1]
        self.conv_kernel = th.stack([ll,lh,hl,hh]).unsqueeze(1).to(device)
        self.conv_kernel = self.conv_kernel.repeat(self.channels, 1, 1, 1)
        self.device = device

    def forward(self, X ):
        return F.conv2d(X, self.conv_kernel, stride=2, groups=self.channels)
 
    def inverse(self, Y ):
        return F.conv_transpose2d(Y, self.conv_kernel, stride=2, groups=self.channels)

