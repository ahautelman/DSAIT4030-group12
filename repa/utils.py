import json
import os
from typing import Dict, List

import psutil
import torch


class ExperimentLogger:
    """Handles logging of telemetry, losses, and metrics to a JSON file."""

    def __init__(self, log_filepath: str):
        self.log_filepath = log_filepath
        self.history: List[Dict] = []
        self.process = psutil.Process(os.getpid())

    def log_step(self, step: int, losses: dict, fid_score: float = None):
        """Records a single step/evaluation event."""
        # Hardware Telemetry
        # ram_mb = self.process.memory_info().rss / (1024 * 1024)
        # if torch.cuda.is_available():
        #     gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        # else:
        #     gpu_mb = 0.0
        gpu_mb = 0.0

        record = {
            "global_step": step,
            "loss_diff": losses.get("loss_diff", 0.0),
            "loss_repa": losses.get("loss_repa", 0.0),
            "loss_total": losses.get("loss_total", 0.0),
            "fid_score": fid_score,
            # "ram_usage_mb": round(ram_mb, 2),
            "gpu_memory_peak_mb": round(gpu_mb, 2),
            "avg_step_time_secs": losses.get("avg_step_time_secs", 0.0),
            "throughput_imgs_sec": losses.get("throughput_imgs_sec", 0.0)
        }

        self.history.append(record)
        self._save_to_disk()

    def _save_to_disk(self):
        with open(self.log_filepath, 'w') as f:
            json.dump(self.history, f, indent=4)
