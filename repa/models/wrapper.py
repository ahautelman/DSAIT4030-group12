from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import AutoModel
from torchvision.transforms.functional import gaussian_blur

from repa.align.hooks import register_feature_hook
from repa.align.shape_utils import match_teacher_spatial_grid, tokens_to_spatial
from repa.models.factory import ModelMeta
from repa.align.projection import ConvProjector
from repa.models.vae import DiffusersVAEWrapper, BaseVAEWrapper
from repa.config import ExperimentConfig


class REPAWrapper(nn.Module):
    def __init__(self, student_model: nn.Module, meta: ModelMeta, config: ExperimentConfig, custom_vae: BaseVAEWrapper = None):
        super().__init__()
        self.mode = config.mode.lower()
        self.model_type = config.model_type.lower()
        self.meta = meta

        self._setup_device_and_dtype()

        # 1. Initialize Teacher & VAE (Frozen)
        self.teacher = AutoModel.from_pretrained(config.teacher_model_id, torch_dtype=self.compute_dtype).to(
            self.device)
        
        if custom_vae is not None:
            self.vae = custom_vae.to(self.device)
        else:
            self.vae = DiffusersVAEWrapper(config.vae_model_id, self.compute_dtype).to(self.device)

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 2. Assign Injected Student
        self.student = student_model.to(self.device)
        self._enable_memory_efficient_attention()

        # 3. Setup Standardized Hook
        self.extractor_fn, self.hook_handle = register_feature_hook(self.student, self.meta.hook_target_name)

        # 4. Lazy Projection Head
        self.proj_head = None if self.mode != "vanilla" else nn.Identity()

    def _setup_device_and_dtype(self):
        """Infers optimal device and mixed precision settings."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

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

    def _enable_memory_efficient_attention(self):
        if hasattr(self.student, "enable_xformers_memory_efficient_attention") and self.device.type == "cuda":
            try:
                self.student.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    def forward_student(self, x, timesteps, class_labels):
        """Architecture-agnostic forward pass."""
        if self.model_type == "sit":
            return self.student(x, timestep=timesteps, class_labels=class_labels)
        if self.model_type == "sit_l_2":
            return self.student(x, timesteps, class_labels)
        return self.student(x, t=timesteps)

    def get_teacher_features(self, x_0: torch.Tensor) -> torch.Tensor:
        if self.mode == "vanilla":
            return torch.empty(0)

        # Prepare teacher input images
        x_0_teacher = F.interpolate(x_0, size=(224, 224), mode='bilinear', align_corners=False).to(self.compute_dtype)
        x_0_teacher = (x_0_teacher + 1.0) / 2.0
        x_0_teacher = TF.normalize(x_0_teacher, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(
            self.compute_dtype)     # normalize image for DinoV2 using ImageNet statistics 

        with torch.no_grad():
            outputs = self.teacher(pixel_values=x_0_teacher, output_hidden_states=True)
            z_0 = outputs.hidden_states[-2]

            # Remove CLS token if present
            if z_0.shape[1] > (x_0_teacher.shape[2] // 14) * (x_0_teacher.shape[3] // 14):
                z_0 = z_0[:, 1:, :]

        return z_0.detach()

    def align_features(self, z_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projects student features and computes target alignments (REPA, iREPA, DoG)."""
        h_t = self.extractor_fn().contiguous()
        if self.proj_head is None:
            raise RuntimeError("Projection head not initialized.")

        B, N_t, D_t = z_0.shape
        H_t = int(N_t ** 0.5)
        z_0_spatial = z_0.transpose(1, 2).view(B, D_t, H_t, H_t)

        z_hat_spatial = self._project_student_features(h_t, z_0_spatial)
        z_0_target = self._compute_alignment_target(z_0_spatial)

        # REPA expects flat sequences, spatial modes expect grids
        if self.mode == "repa":
            return z_hat_spatial.flatten(2).transpose(1, 2), z_0

        return z_hat_spatial, z_0_target

    def _project_student_features(self, h_t: torch.Tensor, z_0_spatial: torch.Tensor) -> torch.Tensor:
        """Handles projection logic depending on whether features are token or spatial."""
        if self.meta.feature_shape_hint == "tokens":
            if isinstance(self.proj_head, ConvProjector):
                h_spatial = tokens_to_spatial(h_t)
                z_hat = self.proj_head(h_spatial)
                return match_teacher_spatial_grid(z_hat, z_0_spatial)

            z_hat = self.proj_head(h_t)
            z_hat_spatial = tokens_to_spatial(z_hat)
            return match_teacher_spatial_grid(z_hat_spatial, z_0_spatial)

        # UNet is already spatial
        z_hat_spatial = self.proj_head(h_t)
        return match_teacher_spatial_grid(z_hat_spatial, z_0_spatial)

    def _compute_alignment_target(self, z_0_spatial: torch.Tensor) -> torch.Tensor:
        """Applies spatial target modifications based on the alignment method."""
        if self.mode == "irepa":
            spatial_mean = z_0_spatial.mean(dim=[-2, -1], keepdim=True)
            spatial_std = z_0_spatial.std(dim=[-2, -1], keepdim=True) + 1e-6
            return (z_0_spatial - spatial_mean) / spatial_std

        if self.mode == "dog":
            blur1 = gaussian_blur(z_0_spatial, kernel_size=[3, 3], sigma=[0.6, 0.6])
            blur2 = gaussian_blur(z_0_spatial, kernel_size=[5, 5], sigma=[2.0, 2.0])
            return blur1 - blur2

        return z_0_spatial