import os
import torch
from tqdm import tqdm
from diffusers import DPMSolverMultistepScheduler
from torchvision.utils import save_image
from cleanfid import fid

@torch.no_grad()
def generate_and_save_images(
        wrapper: torch.nn.Module,
        num_images: int,
        batch_size: int,
        device: torch.device,
        output_dir: str,
        num_inference_steps: int = 20 # Dropped from 50
):
    os.makedirs(output_dir, exist_ok=True)
    wrapper.eval()

    # DPM Solver for rapid inference
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        algorithm_type="dpmsolver++"
    )
    scheduler.set_timesteps(num_inference_steps)

    latent_channels = wrapper.vae.config.latent_channels
    latent_h, latent_w = 32, 32
    num_batches = (num_images + batch_size - 1) // batch_size
    image_counter = 0

    for _ in tqdm(range(num_batches), desc="Generating Fast Eval Images"):
        current_batch_size = min(batch_size, num_images - image_counter)
        latents = torch.randn((current_batch_size, latent_channels, latent_h, latent_w), device=device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for t in scheduler.timesteps:
                noise_pred = wrapper.student(latents, t, class_labels=None).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / wrapper.vae.config.scaling_factor
        images = wrapper.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        for i in range(current_batch_size):
            save_image(images[i], os.path.join(output_dir, f"gen_{image_counter:05d}.png"))
            image_counter += 1

def compute_fid(real_stats_name: str, generated_dir: str, resolution: int = 256) -> float:
    score = fid.compute_fid(
        gen_folder=generated_dir, dataset_name=real_stats_name,
        dataset_res=resolution, dataset_split="custom"
    )
    return score