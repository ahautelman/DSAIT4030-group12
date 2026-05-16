import os
import torch
import logging
from tqdm import tqdm
from diffusers import DDIMScheduler
from torchvision.utils import save_image
from cleanfid import fid


@torch.no_grad()
def generate_and_save_images(
        wrapper: torch.nn.Module,
        num_images: int,
        batch_size: int,
        device: torch.device,
        output_dir: str,
        num_inference_steps: int = 50
):
    """
    Generates images using DDIMScheduler (for speed) and saves them directly to disk.
    Saving to disk is mandatory for large-scale (50k) FID calculations to prevent OOM.
    """
    os.makedirs(output_dir, exist_ok=True)
    wrapper.eval()

    # Initialize DDIM Scheduler for fast evaluation (skips Markov steps)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        clip_sample=False
    )
    scheduler.set_timesteps(num_inference_steps)

    latent_channels = wrapper.vae.config.latent_channels
    # CelebA 256x256 -> VAE downscales by 8 -> 32x32 latents
    latent_h, latent_w = 32, 32

    num_batches = (num_images + batch_size - 1) // batch_size
    image_counter = 0

    logging.info(f"Generating {num_images} images for evaluation using DDIM ({num_inference_steps} steps)...")

    for _ in tqdm(range(num_batches), desc="Generating Images"):
        current_batch_size = min(batch_size, num_images - image_counter)

        # 1. Start with pure Gaussian noise
        latents = torch.randn(
            (current_batch_size, latent_channels, latent_h, latent_w),
            device=device
        )

        # 2. DDIM Denoising Loop
        for t in scheduler.timesteps:
            # Predict noise using the student DiT
            noise_pred = wrapper.student(latents, t, class_labels=None).sample
            # Take a step backward in time
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 3. Decode latents back to pixel space using VAE
        latents = latents / wrapper.vae.config.scaling_factor
        images = wrapper.vae.decode(latents).sample

        # 4. Post-process from [-1, 1] to [0, 1] and save
        images = (images / 2 + 0.5).clamp(0, 1)

        for i in range(current_batch_size):
            save_image(images[i], os.path.join(output_dir, f"gen_{image_counter:05d}.png"))
            image_counter += 1


def compute_fid(real_stats_name: str, generated_dir: str, resolution: int = 256) -> float:
    """
    Computes FID comparing the generated folder against pre-cached real dataset statistics.
    Using 'cleanfid' ensures exact academic reproducibility.
    """
    logging.info(f"Computing FID against cached stats: {real_stats_name}")
    score = fid.compute_fid(
        gen_folder=generated_dir,
        dataset_name=real_stats_name,
        dataset_res=resolution,
        dataset_split="custom"
    )
    return score