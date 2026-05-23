from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKL
from torchvision.transforms.functional import gaussian_blur
from transformers import AutoModel

from repa.align.hooks import register_feature_hook
from repa.align.shape_utils import match_teacher_spatial_grid, tokens_to_spatial
from repa.models.factory import ModelMeta
from repa.align.projection import ConvProjector


class REPAWrapper(nn.Module):
    def __init__(
            self,
            student_model: nn.Module,
            meta: ModelMeta,
            model_type: str,
            teacher_model_id: str = "facebook/dinov2-base",
            vae_model_id: str = "stabilityai/sd-vae-ft-mse",
            mode: str = "vanilla"
    ):
        super().__init__()
        self.mode = mode.lower()
        self.model_type = model_type.lower()
        self.meta = meta
        assert self.mode in ["vanilla", "repa", "irepa", "dog"], f"Unknown mode: {mode}"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.compute_dtype = torch.bfloat16
            self.use_scaler = False
        elif self.device.type in ["mps", "cuda"]:
            self.compute_dtype = torch.float16
            self.use_scaler = True
        else:
            self.compute_dtype = torch.float32
            self.use_scaler = False

        # 1. Initialize Teacher & VAE (Frozen)
        self.teacher = AutoModel.from_pretrained(teacher_model_id, torch_dtype=self.compute_dtype).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=self.compute_dtype).to(self.device)

        self.teacher.eval()
        self.vae.eval()
        for param in self.teacher.parameters(): param.requires_grad = False
        for param in self.vae.parameters(): param.requires_grad = False

        # 2. Assign Injected Student
        self.student = student_model.to(self.device)
        if hasattr(self.student, "enable_xformers_memory_efficient_attention") and self.device.type == "cuda":
            try: self.student.enable_xformers_memory_efficient_attention()
            except Exception: pass

        # 3. Setup Standardized Hook
        self.extractor_fn, self.hook_handle = register_feature_hook(self.student, self.meta.hook_target_name)

        # 4. Lazy Projection Head (Built in train.py on step 1)
        self.proj_head = None if self.mode != "vanilla" else nn.Identity()

    def forward_student(self, x, timesteps, class_labels):
        """Architecture-agnostic forward pass."""
        if self.model_type == "sit":
            return self.student(x, timestep=timesteps, class_labels=class_labels)
        else:
            return self.student(x, t=timesteps)

    def get_teacher_features(self, x_0: torch.Tensor) -> torch.Tensor:
        if self.mode == "vanilla":
            return torch.empty(0)

        x_0_teacher = F.interpolate(x_0, size=(224, 224), mode='bilinear', align_corners=False).to(self.compute_dtype)
        x_0_teacher = (x_0_teacher + 1.0) / 2.0
        x_0_teacher = TF.normalize(x_0_teacher, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(self.compute_dtype)
        
        with torch.no_grad():
            outputs = self.teacher(pixel_values=x_0_teacher, output_hidden_states=True)
            z_0 = outputs.hidden_states[-2]
            if z_0.shape[1] > (x_0_teacher.shape[2] // 14) * (x_0_teacher.shape[3] // 14):
                z_0 = z_0[:, 1:, :]
        return z_0.detach()

    def align_features(self, z_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_t = self.extractor_fn().contiguous()
        
        if self.proj_head is None:
            raise RuntimeError("Projection head not initialized. Call build_projection_for_student first.")

        B, N_t, D_t = z_0.shape
        H_t = int(N_t ** 0.5)
        z_0_spatial = z_0.transpose(1, 2).view(B, D_t, H_t, H_t)

        # === U-REPA Project-Then-Interpolate Logic ===
        
        if self.meta.feature_shape_hint == "tokens":
            if isinstance(self.proj_head, ConvProjector):
                # Reshape FIRST so the 3x3 convolution works on spatial neighbours
                h_spatial = tokens_to_spatial(h_t)
                z_hat_spatial = self.proj_head(h_spatial)
            else:
                # MLP projection on tokens
                z_hat = self.proj_head(h_t)
                z_hat_spatial = tokens_to_spatial(z_hat)
            z_hat_spatial = match_teacher_spatial_grid(z_hat_spatial, z_0_spatial)
        else:
            # UNet is already spatial
            z_hat_spatial = self.proj_head(h_t)
            z_hat_spatial = match_teacher_spatial_grid(z_hat_spatial, z_0_spatial)

        # Apply target modifications based on mode
        if self.mode == "repa":
            return z_hat_spatial.flatten(2).transpose(1, 2), z_0

        elif self.mode == "irepa":
            spatial_mean = z_0_spatial.mean(dim=[-2, -1], keepdim=True)
            spatial_std = z_0_spatial.std(dim=[-2, -1], keepdim=True) + 1e-6
            z_0_target = (z_0_spatial - spatial_mean) / spatial_std
            return z_hat_spatial, z_0_target

        elif self.mode == "dog":
            blur1 = gaussian_blur(z_0_spatial, kernel_size=[3, 3], sigma=[0.6, 0.6])
            blur2 = gaussian_blur(z_0_spatial, kernel_size=[5, 5], sigma=[2.0, 2.0])
            z_0_target = blur1 - blur2
            return z_hat_spatial, z_0_target

        return z_hat_spatial, z_0_spatial
