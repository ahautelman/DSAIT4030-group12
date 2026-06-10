import os
import sys
from sys import platform

from dataclasses import dataclass
from typing import Literal, Tuple
from diffusers import Transformer2DModel

from diffuser.unet import DiffusionUNet
from diffuser.diffuser_example.unet_example_config import DiffuserConfig

ModelType = Literal["sit", "unet", "sit_l_2"]
FeatureShapeHint = Literal["tokens", "spatial"]


@dataclass(frozen=True)
class ModelMeta:
    hook_target_name: str
    feature_shape_hint: FeatureShapeHint
    example_input_shape: Tuple[int, ...]

def _build_sit(student_model_id: str, target_layer_ratio: float = 0.4):
    model_kwargs = {
        "sample_size": 32,
        "patch_size": 2,
        "in_channels": 4,
        "out_channels": 8,
        "num_layers": 12,
        "num_attention_heads": 6,
        "attention_head_dim": 64,
        "norm_type": "ada_norm_zero",
        "activation_fn": "gelu-approximate",
        "num_embeds_ada_norm": 1000,
    }

    student_model = Transformer2DModel(**model_kwargs)
    target_layer_idx = int(len(student_model.transformer_blocks) * target_layer_ratio)
    target_layer_idx = max(0, min(target_layer_idx, len(student_model.transformer_blocks) - 1))

    meta = ModelMeta(
        hook_target_name=f"transformer_blocks.{target_layer_idx}",
        feature_shape_hint="tokens",
        example_input_shape=(1, 4, 32, 32),
    )
    return student_model, meta

def _build_sit_l_2(student_model_id: str, target_layer_ratio: float = 0.4):
    
    if platform == "linux" or platform == "linux2":
        # We assume that the project folder is located in the home directory
        home_dir = os.path.expanduser("~")
        sys.path.insert(0, os.path.abspath(os.path.join(home_dir, 'DSAIT4030-group12')))
        # Requires you to clone https://github.com/willisma/SiT into your home folder
        sys.path.insert(0, os.path.abspath(os.path.join(home_dir, 'SiT')))

    from models import SiT_models

    student_model = SiT_models['SiT-L/2'](
                    input_size=32, 
                    in_channels=4
                    )

    num_blocks = len(student_model.blocks)
    target_layer_idx = int(num_blocks * target_layer_ratio)
    target_layer_idx = max(0, min(target_layer_idx, num_blocks - 1))

    meta = ModelMeta(
        hook_target_name=f"blocks.{target_layer_idx}",
        feature_shape_hint="tokens",
        example_input_shape=(1, 4, 32, 32),
    )
    return student_model, meta


def _build_unet():
    config = DiffuserConfig()
    student_model = DiffusionUNet(
        config=config,
        model_in_channels=4,
        model_out_channels=4,
    )
    meta = ModelMeta(
        hook_target_name="mid_block",
        feature_shape_hint="spatial",
        example_input_shape=(1, 4, 32, 32),
    )
    return student_model, meta


def build_student_model(
    model_type: ModelType,
    student_model_id: str = "BiliSakura/SiT-diffusers",
    target_layer_ratio: float = 0.4,
):
    model_type = model_type.lower()
    if model_type == "sit":
        return _build_sit(student_model_id=student_model_id, target_layer_ratio=target_layer_ratio)
    if model_type == "sit_l_2":
        return _build_sit_l_2(student_model_id=student_model_id, target_layer_ratio=target_layer_ratio)
    if model_type == "unet":
        return _build_unet()
    raise ValueError(f"Unsupported model_type '{model_type}'. Expected 'sit(_l_2)' or 'unet'.")


