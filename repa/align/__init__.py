from .hooks import register_feature_hook
from .projection import ConvProjector, build_projection_for_student
from .shape_utils import match_teacher_spatial_grid, spatial_to_flat, tokens_to_spatial

__all__ = [
    "register_feature_hook",
    "ConvProjector",
    "build_projection_for_student",
    "match_teacher_spatial_grid",
    "spatial_to_flat",
    "tokens_to_spatial",
]

