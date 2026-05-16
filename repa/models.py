import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from diffusers import DiTTransformer2DModel, AutoencoderKL
from typing import Tuple, Optional, Dict


class ProjectionHead(nn.Module):
    """
    Projects the student's hidden states to the teacher's latent dimension.
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(student_dim, student_dim),
            nn.GELU(),
            nn.Linear(student_dim, teacher_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class REPAWrapper(nn.Module):
    """
    Wraps the DiT student, DINOv2 teacher, and Projection Head.
    Handles intermediate feature extraction via forward hooks.
    """

    def __init__(
            self,
            student_model_id: str = "facebook/DiT-XL-2-256",
            teacher_model_id: str = "facebook/dinov2-base",
            vae_model_id: str = "stabilityai/sd-vae-ft-mse",
            target_layer_ratio: float = 0.4
    ):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

        # 1. Initialize Teacher (Frozen)
        self.teacher = AutoModel.from_pretrained(teacher_model_id).to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 2. Initialize VAE (Frozen) <-- NEW
        self.vae = AutoencoderKL.from_pretrained(vae_model_id).to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # 3. Initialize Student
        student_config = DiTTransformer2DModel.load_config(student_model_id, subfolder="transformer")
        self.student = DiTTransformer2DModel.from_config(student_config).to(self.device)

        # 4. Setup Hook
        self.hidden_states: Dict[str, torch.Tensor] = {}
        target_layer_idx = int(len(self.student.transformer_blocks) * target_layer_ratio)
        self._register_hook(target_layer_idx)

        # 5. Initialize Projection Head
        student_dim = self.student.config.num_attention_heads * self.student.config.attention_head_dim
        teacher_dim = self.teacher.config.hidden_size
        self.proj_head = ProjectionHead(student_dim, teacher_dim).to(self.device)

    def _register_hook(self, layer_idx: int) -> None:
        """Registers a forward hook to capture h_t."""

        def hook(model, input, output):
            self.hidden_states['h_t'] = output[0] if isinstance(output, tuple) else output

        self.student.transformer_blocks[layer_idx].register_forward_hook(hook)

    def get_teacher_features(self, x_0: torch.Tensor) -> torch.Tensor:
        """Extracts spatial semantic features from the clean image."""
        # Note: DINOv2 expects 224x224. DiT might use 256x256. 
        # Interpolate clean images for the teacher pass.
        x_0_teacher = F.interpolate(x_0, size=(224, 224), mode='bicubic', align_corners=False)

        with torch.no_grad():
            outputs = self.teacher(pixel_values=x_0_teacher)
            z_0 = outputs.last_hidden_state
            # Strip CLS token to isolate spatial patches
            if z_0.shape[1] > (x_0_teacher.shape[2] // 14) * (x_0_teacher.shape[3] // 14):
                z_0 = z_0[:, 1:, :]
        return z_0

    def align_features(self, h_t: torch.Tensor, z_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projects student features and spatially interpolates to match teacher tokens."""
        z_hat = self.proj_head(h_t)
        B, N_s, D = z_hat.shape
        B, N_t, _ = z_0.shape

        if N_s != N_t:
            H_s = int(N_s ** 0.5)
            H_t = int(N_t ** 0.5)
            z_hat_spatial = z_hat.transpose(1, 2).view(B, D, H_s, H_s)
            z_hat_spatial = F.interpolate(z_hat_spatial, size=(H_t, H_t), mode='bilinear', align_corners=False)
            z_hat = z_hat_spatial.flatten(2).transpose(1, 2)

        return z_hat, z_0