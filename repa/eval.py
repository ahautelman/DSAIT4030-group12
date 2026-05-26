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
        num_inference_steps: int = 20
):
    os.makedirs(output_dir, exist_ok=True)
    wrapper.eval()

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        algorithm_type="dpmsolver++"
    )

    latent_channels = wrapper.vae.latent_channels
    latent_h, latent_w = 32, 32
    num_batches = (num_images + batch_size - 1) // batch_size
    image_counter = 0

    # Inherit compute dtype from the wrapper to ensure safe execution
    compute_dtype = wrapper.compute_dtype

    for _ in tqdm(range(num_batches), desc="Generating Fast Eval Images"):
        current_batch_size = min(batch_size, num_images - image_counter)
        latents = torch.randn((current_batch_size, latent_channels, latent_h, latent_w), device=device)

        # Reset the scheduler's internal state and timesteps for this specific batch
        scheduler.set_timesteps(num_inference_steps)
        with torch.autocast(device_type=device.type, dtype=compute_dtype):
            for t in scheduler.timesteps:
                t_batch = torch.full(
                    (current_batch_size,),
                    t.item() if isinstance(t, torch.Tensor) else t,
                    device=device,
                    dtype=torch.long
                )
                class_labels = torch.full(
                    (current_batch_size,),
                    1000,
                    device=device,
                    dtype=torch.long
                )
                student_output = wrapper.forward_student(latents, timesteps=t_batch, class_labels=class_labels)
                
                # Safe Chunking during inference
                output_sample = student_output.sample if hasattr(student_output, "sample") else student_output
                if output_sample.shape[1] == latent_channels * 2:
                    noise_pred, _ = output_sample.chunk(2, dim=1)
                else:
                    noise_pred = output_sample

                latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / wrapper.vae.scaling_factor
        latents = latents.to(compute_dtype)
        images = wrapper.vae.decode(latents).sample
        images = (images.to(torch.float32) / 2 + 0.5).clamp(0, 1)

        for i in range(current_batch_size):
            save_image(images[i], os.path.join(output_dir, f"gen_{image_counter:05d}.png"))
            image_counter += 1


def compute_fid(real_stats_name: str, generated_dir: str, device: torch.device, resolution: int = 256) -> float:
    score = fid.compute_fid(
        fdir1=generated_dir,
        dataset_name=real_stats_name,
        dataset_res=resolution,
        dataset_split="custom",
        device=device,
        num_workers=0  # to prevent multiprocessing crash
    )
    return score