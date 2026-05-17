import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from diffusers import DiTTransformer2DModel, AutoencoderKL
from typing import Tuple, Dict


class ProjectionHead(nn.Module):
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

        # Determine optimal dtype for frozen models
        self.frozen_dtype = torch.bfloat16 if (torch.cuda.is_available() or torch.mps.is_available()) else torch.float32

        # 1. Initialize Teacher & VAE (Frozen, Half-Precision)
        self.teacher = AutoModel.from_pretrained(teacher_model_id, torch_dtype=self.frozen_dtype).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=self.frozen_dtype).to(self.device)

        self.teacher.eval()
        self.vae.eval()
        for param in self.teacher.parameters(): param.requires_grad = False
        for param in self.vae.parameters(): param.requires_grad = False

        # 2. Initialize Student (Kept in standard precision, autocast handles it during training)
        student_config = DiTTransformer2DModel.load_config(student_model_id, subfolder="transformer")
        self.student = DiTTransformer2DModel.from_config(student_config).to(self.device)

        if hasattr(self.student, "enable_xformers_memory_efficient_attention") and self.device.type == "cuda":
            self.student.enable_xformers_memory_efficient_attention()

        # 3. Setup Hook
        self.hidden_states: Dict[str, torch.Tensor] = {}
        target_layer_idx = int(len(self.student.transformer_blocks) * target_layer_ratio)
        self._register_hook(target_layer_idx)

        # 4. Initialize Projection Head
        student_dim = self.student.config.num_attention_heads * self.student.config.attention_head_dim
        teacher_dim = self.teacher.config.hidden_size
        self.proj_head = ProjectionHead(student_dim, teacher_dim).to(self.device)

    def _register_hook(self, layer_idx: int) -> None:
        def hook(model, input, output):
            self.hidden_states['h_t'] = output[0] if isinstance(output, tuple) else output

        self.student.transformer_blocks[layer_idx].register_forward_hook(hook)

    def get_teacher_features(self, x_0: torch.Tensor) -> torch.Tensor:
        # Cast input to frozen dtype
        x_0_teacher = F.interpolate(x_0, size=(224, 224), mode='bilinear', align_corners=False).to(self.frozen_dtype)
        with torch.no_grad():
            outputs = self.teacher(pixel_values=x_0_teacher)
            z_0 = outputs.last_hidden_state
            if z_0.shape[1] > (x_0_teacher.shape[2] // 14) * (x_0_teacher.shape[3] // 14):
                z_0 = z_0[:, 1:, :]
        return z_0