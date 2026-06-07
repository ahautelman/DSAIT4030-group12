from torch import nn
import torch.nn.functional as F
import torch
import lpips
import torch_dct as dct
from dwt import DWT

class PatchGAN(nn.Module):
    """
    PatchGAN discriminator as defined in Image to Image Translation 
    w/ Conditional Adversarial Networks
    https://arxiv.org/pdf/1611.07004
    
    Scores local patches of an image as real or fake rather than
    giving a single score for the whole image.

    """

    def __init__(self, input_channels=3, start_dim=64, depth=3, kernel_size=4, padding=1, leaky_relu_slope=0.2):

        super(PatchGAN, self).__init__()

        current_filters = start_dim
        layers = nn.ModuleList([])
        layers.append(nn.Conv2d(input_channels, current_filters, kernel_size=kernel_size, stride=2, padding=padding))
        layers.append(nn.LeakyReLU(leaky_relu_slope))

        for i in range(depth):
            stride = 2 if i != depth - 1 else 1
            out_channels = current_filters * 2
            layers.append(nn.Conv2d(current_filters, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            current_filters = out_channels

        layers.append(nn.Conv2d(current_filters, 1, kernel_size=kernel_size, stride=1, padding=padding))

        self.model = nn.Sequential(*layers)

    def init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.2)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0.0)

    def forward(self, x):
        return self.model(x)

def hinge_loss(real, fake):
    """
    Hinge Loss for discriminator training.

    """
    loss_real = torch.mean(F.relu(1.0 - real))
    loss_fake = torch.mean(F.relu(1.0 + fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_loss(real, fake):
    """
    Vanilla GAN loss using softplus.
    Alternative to hinge_loss for discriminator training.

    """
    return 0.5 * (torch.mean(F.softplus(-real)) +
                  torch.mean(F.softplus(fake)))

def reconstruction_loss(real, fake, loss_type="l1"):
    """
    Pixel-wise reconstruction loss between real and reconstructed images.
    """
    if loss_type == "l1":
        return F.l1_loss(fake, real)
    elif loss_type == "l2":
        return F.mse_loss(fake, real)

def perceptual_loss(real, fake, lpips_model):
    """
    Perceptual loss using a pretrained feature extractor (e.g. VGG).
    Compares high-level features of real and fake images.
    """
    return lpips_model(real, fake).mean()

def generator_loss(fake):
    """
    Generator loss for adversarial training.
    Encourages generator to produce outputs that discriminator classifies as real.
    """
    return -torch.mean(fake)

def discriminator_loss(real, fake, loss_type="hinge"):
    """
    Discriminator loss for adversarial training.
    Can use either hinge loss or vanilla GAN loss.
    """
    if loss_type == "hinge":
        return hinge_loss(real, fake)
    elif loss_type == "vanilla":
        return vanilla_loss(real, fake)


def adaptive_weight(perceptual_loss, generator_loss, last_layer):
    """
    Dynamically balances perceptual and generator losses by comparing
    their gradients with respect to the last layer of the decoder.

    """
    perceptual_grad = torch.autograd.grad(perceptual_loss, last_layer, retain_graph=True)[0].norm()
    generator_grad = torch.autograd.grad(generator_loss, last_layer, retain_graph=True)[0].norm()

    weight = perceptual_grad / (generator_grad + 1e-4)
    weight = torch.clamp(weight, 0.0, 1e4).detach()

    return weight

def kl_loss(mu, logvar):
    """
    KL Divergence loss for VAE regularization.
    
    Measures how much the learned latent distribution deviates 
    from a standard normal distribution N(0, 1).
    """
    var = torch.exp(logvar)
    loss = -0.5 * torch.sum(1 + logvar - mu**2 - var, dim=[1, 2, 3])
    return loss.mean()

"""
def radial_power_spectrum(x, n_bins=16, remove_dc=True, eps=1e-8):
    
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    x = x - x.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    power = (fft.real**2 + fft.imag**2).mean(dim=1)  # (B, H, W)

    fy = torch.fft.fftfreq(H, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx**2 + yy**2)

    edges = torch.linspace(0.0, radius.max() + eps, n_bins + 1, device=device, dtype=dtype)
    bins = []

    for i in range(n_bins):
        mask = (radius >= edges[i]) & (radius < edges[i + 1])
        if remove_dc and i == 0:
            mask = mask.clone()
            mask[0, 0] = False

        if mask.sum() == 0:
            bins.append(torch.zeros(B, device=device, dtype=dtype))
        else:
            bins.append(power[:, mask].mean(dim=-1))

    return torch.stack(bins, dim=-1) + eps
"""

def radial_power_spectrum(x, n_bins=16, remove_dc=True, eps=1e-8, mode="standard", transform="dct"):
    """
    Radially averaged 2D power spectrum using DCT or Haar DWT.
    Input:  x -> (B, C, H, W)
    Output: spectrum -> (B, n_bins)
    """
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    x = x - x.mean(dim=(-2, -1), keepdim=True)

    if transform == "dct":
        x_tf = dct.dct_2d(x, norm="ortho")  # (B, C, H, W) real valued
    elif transform == "wavelet":
        dwt = DWT(x.device)
        x_tf = dwt.forward(x)

    if mode=="standard":
        # DCT instead of FFT
        power = (x_tf ** 2).mean(dim=1)     # (B, H, W) — no .real needed, DCT is real
    elif mode=="inverse":
        x_tf_max = x_tf          # Low filtering max - current_value is equivalent to High pass filtering current_value
        power = ( ( x_tf_max - x_tf )** 2).mean(dim=1)

    fy = torch.arange(H, device=device, dtype=dtype)
    fx = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx**2 + yy**2)

    edges = torch.linspace(0.0, radius.max() + eps, n_bins + 1, device=device, dtype=dtype)
    bins = []

    for i in range(n_bins):
        mask = (radius >= edges[i]) & (radius < edges[i + 1])
        if remove_dc and i == 0:
            mask = mask.clone()
            mask[0, 0] = False

        if mask.sum() == 0:
            bins.append(torch.zeros(B, device=device, dtype=dtype))
        else:
            bins.append(power[:, mask].mean(dim=-1))

    return torch.stack(bins, dim=-1) + eps


def normalize_spectrum(spectrum, eps=1e-8):
    spectrum = spectrum.clamp_min(eps)
    return spectrum / spectrum.sum(dim=-1, keepdim=True).clamp_min(eps)


def flatten_spectrum(spectrum, delta=1.0, eps=1e-8):
    """
    Flatten image spectrum using frequency^delta.
    delta=0 gives original spectrum.
    delta>0 gives flatter target.
    """
    n_bins = spectrum.shape[-1]
    freq = torch.arange(1, n_bins + 1, device=spectrum.device, dtype=spectrum.dtype)
    return spectrum * freq.pow(delta).unsqueeze(0) + eps


def esm_loss(x, z, n_bins=16, delta=1.0, remove_dc=True, eps=1e-8, mode="standard", transform="dct"):
    """
    Encoding Spectrum Matching loss:
        L_ESM = KL(flatten(PSD(x)) || PSD(z))
    """

    assert mode in ["standard", "inverse"], \
        f"ESM is only defined as standard or inverse, choose a valid mode to operate it (got: {mode})"
    assert transform in ["dct", "dwt"], \
        f"ESM is only usable with Discrete Cosine Transform (dct) and Discrete Wavelet Transform (dwt) transforms please choose either of those (got: {transform})"
    sx = radial_power_spectrum(x, n_bins=n_bins, remove_dc=remove_dc, eps=eps, mode=mode, transform=transform)
    sz = radial_power_spectrum(z, n_bins=n_bins, remove_dc=remove_dc, eps=eps, mode=mode, transform=transform)

    sx = normalize_spectrum(flatten_spectrum(sx, delta=delta, eps=eps), eps=eps)
    sz = normalize_spectrum(sz, eps=eps)

    return torch.sum(sx * (torch.log(sx + eps) - torch.log(sz + eps)), dim=-1).mean()


    # probably wont be used and will be handled in train vae instead, but here for completeness
def dsm_loss(x_M, x_hat_M, discriminator, lpips_model, lambda1=0.5, lambda2=0.5):
 
    recon = reconstruction_loss(x_M, x_hat_M)
    percep = perceptual_loss(x_M, x_hat_M, lpips_model)
    gen = generator_loss(discriminator(x_hat_M))

    return recon + lambda1 * percep + lambda2 * gen
