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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

        if self.device.type == "cuda":
            # Enable TF32 for Ampere/Ada/Hopper
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Determine precision based on hardware capability
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.compute_dtype = torch.bfloat16
            self.use_scaler = False
        elif self.device.type == "mps" or self.device.type == "cuda":
            self.compute_dtype = torch.float16
            self.use_scaler = True
        else:
            self.compute_dtype = torch.float32
            self.use_scaler = False

        # 1. Initialize Teacher & VAE (Frozen, Optimal Precision)
        self.teacher = AutoModel.from_pretrained(teacher_model_id, torch_dtype=self.compute_dtype).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=self.compute_dtype).to(self.device)

        self.teacher.eval()
        self.vae.eval()
        for param in self.teacher.parameters(): param.requires_grad = False
        for param in self.vae.parameters(): param.requires_grad = False

        # 2. Initialize Student (Maintained in FP32, Autocast handles training)
        student_config = DiTTransformer2DModel.load_config(student_model_id, subfolder="transformer")
        self.student = DiTTransformer2DModel.from_config(student_config).to(self.device)

        # PyTorch 2.0+ handles SDPA automatically, but xformers is an alternative fallback
        if hasattr(self.student, "enable_xformers_memory_efficient_attention") and self.device.type == "cuda":
            try:
                self.student.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

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
            # Only capture states during training to prevent eval memory spikes
            if self.training:
                self.hidden_states['h_t'] = output[0] if isinstance(output, tuple) else output

        self.student.transformer_blocks[layer_idx].register_forward_hook(hook)

    def get_teacher_features(self, x_0: torch.Tensor) -> torch.Tensor:
        x_0_teacher = F.interpolate(x_0, size=(224, 224), mode='bilinear', align_corners=False).to(self.compute_dtype)
        with torch.no_grad():
            outputs = self.teacher(pixel_values=x_0_teacher)
            z_0 = outputs.last_hidden_state
            if z_0.shape[1] > (x_0_teacher.shape[2] // 14) * (x_0_teacher.shape[3] // 14):
                z_0 = z_0[:, 1:, :]
        return z_0

    def align_features(self, h_t: torch.Tensor, z_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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