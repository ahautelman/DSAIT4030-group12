## ALP-ADDITION: To create the new noises

import torch

def _frequency_radius(height, width, device):
    fy = torch.fft.fftfreq(height, device=device).view(height, 1)
    fx = torch.fft.fftfreq(width, device=device).view(1, width)
    radius = torch.sqrt(fx ** 2 + fy ** 2)
    radius = radius / (radius.max() + 1e-8)
    return radius


def frequency_weighted_gaussian_noise(
    x_like,
    mode="white",
    strength=1.0,
    eps=1e-8,
):
    """
    Returns Gaussian noise with a frequency-shaped spectrum.

    mode:
      - "white": normal torch.randn_like noise
      - "hf": amplify high-frequency noise
      - "lf": amplify low-frequency noise

    strength:
      Controls how strongly frequencies are reweighted.
    """
    noise = torch.randn_like(x_like)

    if mode == "white":
        return noise

    b, c, h, w = noise.shape
    device = noise.device
    dtype = noise.dtype

    # FFT in float32 for stability, then cast back.
    noise_f = torch.fft.fft2(noise.float(), dim=(-2, -1))

    radius = _frequency_radius(h, w, device).view(1, 1, h, w)

    if mode == "hf":
        # Low at center, high at outer frequencies.
        weight = 1.0 + strength * radius
    elif mode == "lf":
        # High at center, lower toward outer frequencies.
        weight = 1.0 + strength * (1.0 - radius)
    else:
        raise ValueError(f"Unknown noise mode: {mode}")

    noise_f = noise_f * weight
    shaped = torch.fft.ifft2(noise_f, dim=(-2, -1)).real

    # keep the total noise scale comparable across modes.
    shaped = shaped - shaped.mean(dim=(-2, -1), keepdim=True)
    shaped = shaped / (shaped.std(dim=(-2, -1), keepdim=True) + eps)

    return shaped.to(dtype)