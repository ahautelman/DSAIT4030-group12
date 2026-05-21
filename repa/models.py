import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import AutoModel
from diffusers import Transformer2DModel, AutoencoderKL
from typing import Tuple, Dict
from torchvision.transforms.functional import gaussian_blur


class ProjectionHead(nn.Module):
    """Vanilla REPA token-wise MLP Projection Head."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(student_dim, student_dim),
            nn.GELU(),
            nn.Linear(student_dim, teacher_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class iREPAProjectionHead(nn.Module):
    """iREPA Spatially Preserving 3x3 Convolutional Projection Head (Also used for DoG)."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=student_dim,
            out_channels=teacher_dim,
            kernel_size=3,
            padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects shape: [B, student_dim, H, W]
        return self.conv(x)


class REPAWrapper(nn.Module):
    def __init__(
            self,
            student_model_id: str = "BiliSakura/SiT-diffusers",
            teacher_model_id: str = "facebook/dinov2-base",
            vae_model_id: str = "stabilityai/sd-vae-ft-mse",
            target_layer_ratio: float = 0.4,
            mode: str = "vanilla"
    ):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ["vanilla", "repa", "irepa", "dog"], f"Unknown mode: {mode}"

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.compute_dtype = torch.bfloat16
            self.use_scaler = False
        elif self.device.type == "mps" or self.device.type == "cuda":
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

        # 2. Initialize Student Architecture (Without Pretrained Weights)
        raw_config = Transformer2DModel.load_config(student_model_id, subfolder="SiT-S-2-256-diffusers/transformer")

        # Safely extract legacy architecture dimensions
        hidden_size = raw_config.get("hidden_size", 384)
        num_heads = raw_config.get("num_heads", 6)

        # Explicitly map and instantiate to bypass the broken diffusers legacy parser
        model_kwargs = {
            "sample_size": raw_config.get("input_size", 32),
            "patch_size": 2,
            "in_channels": 4,
            "out_channels": 8,  # SiT predicts both noise (4) and variance (4)
            "num_layers": raw_config.get("depth", 12),
            "num_attention_heads": num_heads,
            "attention_head_dim": hidden_size // num_heads,  # 384 // 6 = 64
            "norm_type": "ada_norm_zero",
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": raw_config.get("num_classes", 1000),
        }

        self.student = Transformer2DModel(**model_kwargs).to(self.device)
        
        if hasattr(self.student, "enable_xformers_memory_efficient_attention") and self.device.type == "cuda":
            try:
                self.student.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        # 3. Setup Layer Hooking
        self.hidden_states: Dict[str, torch.Tensor] = {}
        target_layer_idx = int(len(self.student.transformer_blocks) * target_layer_ratio)
        self._register_hook(target_layer_idx)

        # 4. Initialize Alignment Projection Heads dynamically
        if hasattr(self.student.config, "inner_dim"):
            student_dim = self.student.config.inner_dim
        else:
            student_dim = self.student.config.num_attention_heads * self.student.config.attention_head_dim

        # Use teacher's penultimate hidden layer
        teacher_dim = self.teacher.config.hidden_size   # for DinoV2 uses the same hidden size across al layers

        if self.mode == "repa":
            self.proj_head = ProjectionHead(student_dim, teacher_dim).to(self.device)
        elif self.mode in ["irepa", "dog"]:
            self.proj_head = iREPAProjectionHead(student_dim, teacher_dim).to(self.device)
        else:
            self.proj_head = nn.Identity()

    def _register_hook(self, layer_idx: int) -> None:
        def hook(model, input, output):
            if self.training and self.mode != "vanilla":
                self.hidden_states['h_t'] = output[0] if isinstance(output, tuple) else output

        self.student.transformer_blocks[layer_idx].register_forward_hook(hook)

    def get_teacher_features(self, x_0: torch.Tensor) -> torch.Tensor:
        if self.mode == "vanilla":
            return torch.empty(0)

        x_0_teacher = F.interpolate(x_0, size=(224, 224), mode='bilinear', align_corners=False).to(self.compute_dtype)
        
        # Convert from VAE format [-1, 1] back to [0, 1]
        x_0_teacher = (x_0_teacher + 1.0) / 2.0

        # Apply ImageNet normalization for DINOv2
        x_0_teacher = TF.normalize(
            x_0_teacher,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).to(self.compute_dtype)
        
        with torch.no_grad():
            outputs = self.teacher(pixel_values=x_0_teacher, output_hidden_states=True)
            # Extract penultimate layer (-2)
            z_0 = outputs.hidden_states[-2]
            
            # Clip off CLS token if present
            if z_0.shape[1] > (x_0_teacher.shape[2] // 14) * (x_0_teacher.shape[3] // 14):
                z_0 = z_0[:, 1:, :]
        return z_0.detach()     # detach to prevent accidental graph tracking

    def align_features(self, h_t: torch.Tensor, z_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_s, D_s = h_t.shape
        B, N_t, D_t = z_0.shape
        H_s = int(N_s ** 0.5)
        H_t = int(N_t ** 0.5)

        if self.mode == "repa":
            # Vanilla REPA: Process tokens sequentially with MLP head
            z_hat = self.proj_head(h_t)
            if N_s != N_t:
                z_hat_spatial = z_hat.transpose(1, 2).view(B, D_t, H_s, H_s)
                z_hat_spatial = F.interpolate(z_hat_spatial, size=(H_t, H_t), mode='bilinear', align_corners=False)
                z_hat = z_hat_spatial.flatten(2).transpose(1, 2)
            return z_hat, z_0

        elif self.mode in ["irepa", "dog"]:
            # Reshape representations into 2D spatial grids
            h_t_spatial = h_t.transpose(1, 2).view(B, D_s, H_s, H_s)
            z_hat_spatial = self.proj_head(h_t_spatial)

            z_0_spatial = z_0.transpose(1, 2).view(B, D_t, H_t, H_t)

            if H_s != H_t:
                z_hat_spatial = F.interpolate(z_hat_spatial, size=(H_t, H_t), mode='bilinear', align_corners=False)

            if self.mode == "irepa":
                # iREPA: Spatial Instance-like Normalization
                spatial_mean = z_0_spatial.mean(dim=[-2, -1], keepdim=True)
                spatial_std = z_0_spatial.std(dim=[-2, -1], keepdim=True) + 1e-6
                z_0_target = (z_0_spatial - spatial_mean) / spatial_std

            else:
                # DoG: Difference of Gaussians (Spectrum Matching Hypothesis)
                # Apply band-pass filtering to isolate mid-frequency directional energy
                blur1 = gaussian_blur(z_0_spatial, kernel_size=[3, 3], sigma=[0.6, 0.6])
                blur2 = gaussian_blur(z_0_spatial, kernel_size=[5, 5], sigma=[2.0, 2.0])
                z_0_target = blur1 - blur2

            return z_hat_spatial, z_0_target

        return h_t, z_0