import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

from repa.align.projection import build_projection_for_student
from repa.models import REPAWrapper



class DiffusionTrainer:
    def __init__(self, model_wrapper: REPAWrapper, learning_rate: float, lambda_repa: float):
        self.wrapper = model_wrapper
        self.mode = self.wrapper.mode
        self.lambda_repa = lambda_repa
        self.device = torch.device(self.wrapper.device)
        self.device_type = self.device.type
        self.dtype = self.wrapper.compute_dtype
        self.num_train_timesteps = 1000

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps)

        trainable_params = list(self.wrapper.student.parameters())
        if self.mode in ["repa", "irepa", "dog"] and self.wrapper.proj_head is not None:
            trainable_params += list(self.wrapper.proj_head.parameters())

        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        self.scaler = torch.amp.GradScaler(device=self.device_type, enabled=self.wrapper.use_scaler)

    def train_step(self, x_0: torch.Tensor) -> dict:
        self.optimizer.zero_grad(set_to_none=True)
        B = int(x_0.shape[0])
        class_labels = x_0.new_full((B,), 1000, dtype=torch.long)

        # 1. Target Features
        with torch.no_grad(), torch.autocast(device_type=self.device_type, dtype=self.dtype):
            latent_dist = self.wrapper.vae.encode(x_0.to(self.dtype)).latent_dist
            latents_0 = latent_dist.sample() * self.wrapper.vae.config.scaling_factor
            z_0 = self.wrapper.get_teacher_features(x_0)

        # 2. Add Noise
        noise = torch.randn_like(latents_0)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
        x_t = self.noise_scheduler.add_noise(latents_0, noise, timesteps)

        loss_repa_val = 0.0

        # 3. Execution Pass
        with torch.autocast(device_type=self.device_type, dtype=self.dtype):
            student_outputs = self.wrapper.forward_student(x_t, timesteps=timesteps, class_labels=class_labels)

            # Lazy init the projector on the very first step
            if self.mode != "vanilla" and self.wrapper.proj_head is None:
                self.wrapper.proj_head = build_projection_for_student(
                    self.wrapper.meta,
                    self.wrapper.extractor_fn(),
                    z_0,
                    str(self.mode),
                ).to(self.device)
                self.optimizer.add_param_group({'params': self.wrapper.proj_head.parameters()})

            # Safe Chunking: Handle variance channels only if the model predicts them
            output_sample = student_outputs.sample if hasattr(student_outputs, "sample") else student_outputs
            if output_sample.shape[1] == latents_0.shape[1] * 2:
                predicted_noise, _ = output_sample.chunk(2, dim=1)
            else:
                predicted_noise = output_sample

            if self.mode != "vanilla":
                z_hat, z_target = self.wrapper.align_features(z_0)

                if self.mode == "repa":
                    # Token sequence cosine similarity: mean over tokens (dim=1), output is [B]
                    loss_repa_per_sample = - F.cosine_similarity(z_hat, z_target, dim=-1).mean(dim=1)
                elif self.mode in ["irepa", "dog"]:
                    # Spatial grid: mean over spatial dims H and W, output is [B]
                    loss_repa_per_sample = - F.cosine_similarity(z_hat, z_target, dim=1).mean(dim=[1, 2])
                
                # Timestep dependent weighting
                t_norm = timesteps.float() / self.num_train_timesteps
                
                # Stronger alignment at high noise (t->1000), weaker to low noise (t->0)
                dynamic_lambda = self.lambda_repa * t_norm

                loss_repa = (loss_repa_per_sample * dynamic_lambda).mean()
                loss_repa_val = loss_repa_per_sample.mean().item()
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