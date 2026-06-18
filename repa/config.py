import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ExperimentConfig:
    data_dir: str
    dataset_name: str
    output_dir: str
    max_steps: int
    batch_size: int | None
    lr: float
    model_type: Literal["sit", "sit_l_2", "unet"]
    mode: Literal["vanilla", "repa", "irepa", "dog"]
    lambda_repa: float
    num_evals: int
    num_eval_images: int
    teacher_model_id: str = "facebook/dinov2-base"
    vae_model_id: str = "stabilityai/sd-vae-ft-mse"

    @classmethod
    def _build_parser(cls) -> argparse.ArgumentParser:
        project_root = Path(__file__).resolve().parent.parent
        default_data_dir = str(project_root / "data")
        default_output_dir = str(project_root / "results")

        parser = argparse.ArgumentParser()
        parser.add_argument("--data_dir", type=str, default=default_data_dir)
        parser.add_argument("--dataset_name", type=str, default="celeba")
        parser.add_argument("--output_dir", type=str, default=default_output_dir)
        parser.add_argument("--max_steps", type=int, default=30_000)
        parser.add_argument("--batch_size", type=int, default=None)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--model_type", type=str, choices=["sit", "sit_l_2", "unet"], default="sit")
        parser.add_argument("--mode", type=str, choices=["vanilla", "repa", "irepa", "dog"], default="dog")
        parser.add_argument("--lambda_repa", type=float, default=1.0)
        parser.add_argument("--num_evals", type=int, default=40)
        parser.add_argument("--num_eval_images", type=int, default=2_000)
        return parser

    @classmethod
    def from_args(cls, args: list[str] | None = None) -> "ExperimentConfig":
        """Parses CLI arguments into a typed configuration object."""
        parser = cls._build_parser()
        parsed_args = parser.parse_args(args=args)

        if parsed_args.max_steps <= 1:
            parsed_args.num_evals = 1
            parsed_args.num_eval_images = min(parsed_args.num_eval_images, parsed_args.batch_size or 1)

        return cls(**vars(parsed_args))

    @classmethod
    def from_defaults(cls) -> "ExperimentConfig":
        """Creates a config populated only with the declared defaults."""
        return cls.from_args(args=[])

