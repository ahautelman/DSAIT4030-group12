import os
import shutil
import gc
import time
from pathlib import Path

import cleanfid
from cleanfid import fid
import torch
from tqdm import tqdm

from repa.config import ExperimentConfig
from repa.dataset import get_celeba_dataloader
from repa.models.wrapper import REPAWrapper
from repa.models.factory import build_student_model
from repa.eval import generate_and_save_images, compute_fid
from repa.utils import ExperimentLogger
from repa.train import DiffusionTrainer


def find_max_batch_size(trainer: DiffusionTrainer, starting_batch: int = 256) -> int:
    """Empirically finds the maximum safe batch size for hardware."""
    current_batch = starting_batch
    print(f"Starting automatic batch size finder from {starting_batch}...")

    while current_batch > 0:
        try:
            dummy_images = torch.randn((current_batch, 3, 256, 256), device=trainer.device)
            trainer.optimizer.zero_grad(set_to_none=True)
            _ = trainer.train_step(dummy_images)

            trainer.optimizer.zero_grad(set_to_none=True)
            del dummy_images
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

            safe_batch = max(1, int(current_batch * 0.70))
            print(f"Success! Safe batch size set to: {safe_batch}")
            return safe_batch

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
                current_batch -= 32
                trainer.optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
            else:
                raise e
    raise RuntimeError("Could not find a viable batch size.")


def get_infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def run_evaluation_step(trainer, config, global_step, best_fid, optimal_batch_size, avg_step_time, running_losses,
                        logger):
    """Executes the validation generation, computes FID, and saves checkpoints."""
    print(f"\n--- Running Evaluation Step {global_step} ---")
    eval_dir = os.path.join(config.output_dir, "temp_eval_fast")

    generate_and_save_images(
        trainer.wrapper, config.num_eval_images, optimal_batch_size, trainer.device, eval_dir
    )

    try:
        current_fid = compute_fid(config.dataset_name, eval_dir, trainer.device)
        print(f"Step {global_step} | Fast FID: {current_fid:.4f}")
    except Exception as e:
        print(f"Evaluation Failed: {e}")
        current_fid = float('inf')

    metrics_to_log = {
        **running_losses,
        "avg_step_time_secs": avg_step_time,
        "throughput_imgs_sec": optimal_batch_size / avg_step_time
    }
    logger.log_step(global_step, metrics_to_log, current_fid)

    # State Checkpointing
    checkpoint_dict = {
        "step": global_step,
        "student_state": trainer.wrapper.student.state_dict(),
        "fid": current_fid,
        "mode": config.mode,
        "model_type": config.model_type,
    }
    if config.mode != "vanilla":
        checkpoint_dict["proj_head_state"] = trainer.wrapper.proj_head.state_dict()

    torch.save(checkpoint_dict,
               os.path.join(config.output_dir, f"checkpoint_last_{config.model_type}_{config.mode}.pt"))

    if current_fid < best_fid:
        best_fid = current_fid
        torch.save(checkpoint_dict,
                   os.path.join(config.output_dir, f"checkpoint_best_{config.model_type}_{config.mode}.pt"))
        print("🌟 Saved new optimal state checkpoint.")

    shutil.rmtree(eval_dir)
    trainer.wrapper.train()
    return best_fid


def cache_cleanfid_stats(config: ExperimentConfig, device: torch.device):
    """Ensures CleanFID statistics are pre-computed and cached locally."""
    stat_path_1 = os.path.join(os.path.dirname(cleanfid.__file__), "stats",
                               f"{config.dataset_name}_clean_custom_na.npz")
    stat_path_2 = os.path.expanduser(f"~/.cache/cleanfid/stats/{config.dataset_name}_clean_custom_na.npz")

    if not (os.path.exists(stat_path_1) or os.path.exists(stat_path_2)):
        image_dir = Path(config.data_dir) / config.dataset_name
        fid.make_custom_stats(name=config.dataset_name, fdir=str(image_dir), device=device, num_workers=0)


def main():
    config = ExperimentConfig.from_args()
    os.makedirs(config.output_dir, exist_ok=True)

    log_file = os.path.join(config.output_dir, f"experiment_log_{config.model_type}_{config.mode}.json")
    logger = ExperimentLogger(log_file)

    # 1. Initialize Student & Wrapper
    student_model, meta = build_student_model(config.model_type)
    wrapper = REPAWrapper(student_model=student_model, meta=meta, config=config)

    # 2. Init Trainer Execution Instance
    trainer = DiffusionTrainer(wrapper, config.lr, config.lambda_repa)

    # 3. Auto-Batch Optimization & Datasets
    optimal_batch_size = config.batch_size if config.batch_size else find_max_batch_size(trainer, starting_batch=64)
    eval_step_interval = max(1, config.max_steps // config.num_evals)

    dataloader = get_celeba_dataloader(config.data_dir, optimal_batch_size)
    data_iterator = get_infinite_dataloader(dataloader)

    cache_cleanfid_stats(config, trainer.device)

    # 4. Main Training Loop
    global_step = 1
    best_fid = float('inf')
    running_losses = {"loss_diff": 0.0, "loss_repa": 0.0, "loss_total": 0.0}
    step_times = []

    trainer.wrapper.train()
    progress_bar = tqdm(total=config.max_steps, desc=f"Training [{config.model_type.upper()}::{config.mode.upper()}]")

    while global_step <= config.max_steps:
        images, _ = next(data_iterator)
        images = images.to(trainer.device)

        if trainer.device.type == "cuda": torch.cuda.synchronize()
        start_time = time.perf_counter()

        losses = trainer.train_step(images)

        if trainer.device.type == "cuda": torch.cuda.synchronize()
        end_time = time.perf_counter()

        step_times.append(end_time - start_time)
        for k in running_losses.keys():
            running_losses[k] = losses[k]

        progress_bar.set_postfix({
            "Diff": f"{running_losses['loss_diff']:.4f}",
            "Align": f"{running_losses['loss_repa']:.4f}",
            "Step_ms": f"{(end_time - start_time) * 1000:.1f}ms"
        })
        progress_bar.update(1)

        # Evaluation Pipeline Phase
        if global_step % eval_step_interval == 0 or global_step == config.max_steps:
            avg_step_time = sum(step_times) / len(step_times)
            best_fid = run_evaluation_step(
                trainer, config, global_step, best_fid,
                optimal_batch_size, avg_step_time, running_losses, logger
            )
            step_times.clear()

        global_step += 1

    progress_bar.close()
    print("Execution complete.")


if __name__ == "__main__":
    main()