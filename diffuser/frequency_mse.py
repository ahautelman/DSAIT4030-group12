## ALP-ADDITION: This loss is to-be-used in the validation set
### Explanation: When we use different noises instead of Gaussian, how good is the model synthesizing different frequencies?

import torch
import torch.nn.functional as F


def radial_average(power: torch.Tensor) -> torch.Tensor:
    """
    power: [B, C, H, W]
    returns: [R], average spectral error per frequency radius.
    """
    _, _, h, w = power.shape
    device = power.device

    y = torch.arange(h, device=device) - h // 2
    x = torch.arange(w, device=device) - w // 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    radius = torch.sqrt(xx.float() ** 2 + yy.float() ** 2).long()
    max_radius = int(radius.max().item())

    values = []
    for r in range(max_radius + 1):
        mask = radius == r
        if mask.any():
            values.append(power[:, :, mask].mean())
        else:
            values.append(torch.tensor(0.0, device=device))

    return torch.stack(values)


@torch.no_grad()
def compute_per_frequency_denoising_mse(
    model,
    ddpm,
    images: torch.Tensor,
    t_value: int,
    device: torch.device,
    fold_factor: int = 1,
) -> torch.Tensor:
    """
    Validation-only metric.

    Steps:
      x0 -> x_t using ddpm.forward_diffusion
      model predicts noise
      reconstruct x0_hat from predicted noise
      compute Fourier-domain radial MSE of x0 - x0_hat
    """
    model.eval()

    x0 = images.to(device)

    if fold_factor > 1:
        x0 = F.pixel_unshuffle(x0, downscale_factor=fold_factor)

    batch_size = x0.shape[0]
    t = torch.full(
        (batch_size,),
        t_value,
        device=device,
        dtype=torch.long,
    )

    x_t, _true_noise = ddpm.forward_diffusion(x0, t)
    pred_noise = model(x_t, t)

    alpha_bar_t = ddpm.alpha_bars[t].to(device).view(-1, 1, 1, 1)

    x0_hat = (
        x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise
    ) / torch.sqrt(alpha_bar_t)

    error = x0 - x0_hat

    fft_error = torch.fft.fftshift(
        torch.fft.fft2(error.float(), dim=(-2, -1)),
        dim=(-2, -1),
    )

    power_error = torch.abs(fft_error) ** 2

    return radial_average(power_error)