import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from typing import Optional
from models import REPAWrapper


class DiffusionTrainer:
    def __init__(self, model_wrapper: REPAWrapper, learning_rate: float, use_repa: bool, lambda_repa: float):
        self.wrapper = model_wrapper
        self.use_repa = use_repa
        self.lambda_repa = lambda_repa
        self.device = self.wrapper.device

        # Noise Scheduler
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        # Optimizer: Train Student + Projection Head
        trainable_params = list(self.wrapper.student.parameters())
        if self.use_repa:
            trainable_params += list(self.wrapper.proj_head.parameters())

        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    def train_step(self, x_0: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> dict:
        self.optimizer.zero_grad()
        B = x_0.shape[0]

        if class_labels is None:
            # 1000 is the dedicated "null" class index for DiT models
            class_labels = torch.full((B,), 1000, device=self.device, dtype=torch.long)

        # Encode images to latent space
        with torch.no_grad():
            # VAE outputs a distribution, we sample from it
            latent_dist = self.wrapper.vae.encode(x_0).latent_dist
            latents_0 = latent_dist.sample()
            # Standard LDM scaling (prevents variance explosion in latents)
            latents_0 = latents_0 * self.wrapper.vae.config.scaling_factor

        # Sample noise and timesteps
        noise = torch.randn_like(latents_0)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
        x_t = self.noise_scheduler.add_noise(latents_0, noise, timesteps)
        
        loss_repa_val = 0.0

        if self.use_repa:
            # 1. Teacher Pass
            z_0 = self.wrapper.get_teacher_features(x_0)

            # 2. Student Forward Pass (Hooks populate hidden states)
            student_outputs = self.wrapper.student(x_t, timestep=timesteps, class_labels=class_labels)
            predicted_noise, predicted_variance = student_outputs.sample.chunk(2, dim=1)    # split the 8-channel output into 4-channel noise and 4-channel variance
            
            # 3. Extract and Align Features
            h_t = self.wrapper.hidden_states['h_t']
            z_hat, z_0_aligned = self.wrapper.align_features(h_t, z_0)

            # 4. REPA Loss
            loss_repa = - F.cosine_similarity(z_hat, z_0_aligned, dim=-1).mean()
            loss_repa_val = loss_repa.item()
        else:
            # Standard Pass
            student_outputs = self.wrapper.student(x_t, timestep=timesteps, class_labels=class_labels)
            predicted_noise, predicted_variance = student_outputs.sample.chunk(2, dim=1)
            loss_repa = 0.0

        # Standard Diffusion Loss (MSE)
        loss_diff = F.mse_loss(predicted_noise, noise)

        # Total Objective
        loss_total = loss_diff + (self.lambda_repa * loss_repa if self.use_repa else 0.0)

        loss_total.backward()
        self.optimizer.step()

        return {
            "loss_total": loss_total.item(),
            "loss_diff": loss_diff.item(),
            "loss_repa": loss_repa_val
        }