import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from models import REPAWrapper


class DiffusionTrainer:
    def __init__(self, model_wrapper: REPAWrapper, learning_rate: float, lambda_repa: float):
        self.wrapper = model_wrapper
        self.mode = self.wrapper.mode
        self.lambda_repa = lambda_repa
        self.device = self.wrapper.device
        self.dtype = self.wrapper.compute_dtype

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        trainable_params = list(self.wrapper.student.parameters())
        if self.mode in ["repa", "irepa", "dog"]:
            trainable_params += list(self.wrapper.proj_head.parameters())

        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.wrapper.use_scaler)

    def train_step(self, x_0: torch.Tensor) -> dict:
        self.optimizer.zero_grad(set_to_none=True)
        B = x_0.shape[0]
        class_labels = torch.full((B,), 1000, device=self.device, dtype=torch.long)

        # 1. Target Features
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
            latent_dist = self.wrapper.vae.encode(x_0.to(self.dtype)).latent_dist
            latents_0 = latent_dist.sample() * self.wrapper.vae.config.scaling_factor
            z_0 = self.wrapper.get_teacher_features(x_0)

        # 2. Add Noise
        noise = torch.randn_like(latents_0)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
        x_t = self.noise_scheduler.add_noise(latents_0, noise, timesteps)

        loss_repa_val = 0.0

        # 3. Execution Pass
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            student_outputs = self.wrapper.student(x_t, timestep=timesteps, class_labels=class_labels)

            # Safe Chunking: Handle variance channels only if the model predicts them
            output_sample = student_outputs.sample
            if output_sample.shape[1] == latents_0.shape[1] * 2:
                predicted_noise, _ = output_sample.chunk(2, dim=1)
            else:
                predicted_noise = output_sample

            if self.mode != "vanilla":
                h_t = self.wrapper.hidden_states['h_t'].contiguous()
                z_hat, z_target = self.wrapper.align_features(h_t, z_0)

                if self.mode == "repa":
                    # Token sequence cosine similarity: mean over tokens (dim=1), output is [B]
                    loss_repa_per_sample = - F.cosine_similarity(z_hat, z_target, dim=-1).mean(dim=1)
                elif self.mode in ["irepa", "dog"]:
                    # Spatial grid: mean over spatial dims H and W, output is [B]
                    loss_repa_per_sample = - F.cosine_similarity(z_hat, z_target, dim=1).mean(dim=[1, 2])
                
                # Timestep dependent weighting
                t_norm = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
                
                # Stronger alignment at low noise (t->0), weaker to high noise (t->1000)
                dynamic_lambda = self.lambda_repa * (1.0 - t_norm)

                loss_repa = (loss_repa_per_sample * dynamic_lambda).mean()
                loss_repa_val = loss_repa.item()
            else:
                loss_repa = 0.0
                dynamic_lambda = 0.0

            loss_diff = F.mse_loss(predicted_noise, noise)
            loss_total = loss_diff + loss_repa

        # 4. Optimize
        self.scaler.scale(loss_total).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss_total": loss_total.item(),
            "loss_diff": loss_diff.item(),
            "loss_repa": loss_repa_val
        }