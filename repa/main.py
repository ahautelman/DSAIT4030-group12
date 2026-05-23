# main.py
import argparse
import os
import shutil
import gc
import time
from pathlib import Path

import cleanfid
from cleanfid import fid
import torch
from tqdm import tqdm
from repa.dataset import get_celeba_dataloader
from repa.models import REPAWrapper
from repa.models.factory import build_student_model
from repa.eval import generate_and_save_images, compute_fid
from repa.utils import ExperimentLogger
from repa.train import DiffusionTrainer


def find_max_batch_size(trainer, starting_batch=256):
    current_batch = starting_batch
    device = trainer.device
    print(f"Starting automatic batch size finder from {starting_batch}...")

    while current_batch > 0:
        try:
            dummy_images = torch.randn((current_batch, 3, 256, 256), device=device)
            trainer.optimizer.zero_grad(set_to_none=True)
            _ = trainer.train_step(dummy_images)

            trainer.optimizer.zero_grad(set_to_none=True)
            del dummy_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            safe_batch = max(1, int(current_batch * 0.70))
            print(f"Success! Safe batch size set to: {safe_batch}")
            return safe_batch

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
                current_batch -= 32
                trainer.optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                raise e
    raise RuntimeError("Could not find a viable batch size.")


def get_infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset_name", type=str, default="celeba")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--max_steps", type=int, default=30_000)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_type", type=str, choices=["sit", "unet"], default="sit")
    parser.add_argument("--mode", type=str, choices=["vanilla", "repa", "irepa", "dog"], default="dog",
                        help="Structural alignment methodology variant to implement")
    parser.add_argument("--lambda_repa", type=float, default=1.0,
                        help="Weighting factor for the alignment loss component. Recommended: use 0.4 for REPA, 1.0 for iREPA / DoG.")
    parser.add_argument("--num_evals", type=int, default=40,
                        help="Target number of evaluations during training")
    parser.add_argument("--num_eval_images", type=int, default=2_000)
    args = parser.parse_args()

    if args.max_steps <= 1:
        args.num_evals = 1
        args.num_eval_images = min(args.num_eval_images, args.batch_size or 1)

    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, f"experiment_log_{args.model_type}_{args.mode}.json")
    logger = ExperimentLogger(log_file)

    # 1. Initialize Wrapper Configuration
    student_model, meta = build_student_model(args.model_type)
    wrapper = REPAWrapper(student_model=student_model, meta=meta, model_type=args.model_type, mode=args.mode)

    # 2. Init Trainer Execution Instance
    trainer = DiffusionTrainer(wrapper, args.lr, args.lambda_repa)

    # 3. Auto-Batch Optimization
    optimal_batch_size = args.batch_size if args.batch_size else find_max_batch_size(trainer, starting_batch=64)

    # Calculate step interval to hit the target number of evaluations
    eval_step_interval = max(1, args.max_steps // args.num_evals)

    dataloader = get_celeba_dataloader(args.data_dir, optimal_batch_size)
    data_iterator = get_infinite_dataloader(dataloader)

    # Cache CleanFID Statistics
    stat_path_1 = os.path.join(os.path.dirname(cleanfid.__file__), "stats", f"{args.dataset_name}_clean_custom_na.npz")
    stat_path_2 = os.path.expanduser(f"~/.cache/cleanfid/stats/{args.dataset_name}_clean_custom_na.npz")

    if not (os.path.exists(stat_path_1) or os.path.exists(stat_path_2)):
        image_dir = Path(args.data_dir) / args.dataset_name
        fid.make_custom_stats(name=args.dataset_name, fdir=str(image_dir), device=trainer.device, num_workers=0)

    global_step = 1
    best_fid = float('inf')
    running_losses = {"loss_diff": 0.0, "loss_repa": 0.0, "loss_total": 0.0}

    trainer.wrapper.train()
    progress_bar = tqdm(total=args.max_steps, desc=f"Training [{args.model_type.upper()}::{args.mode.upper()}]")

    # Metrics for empirical tracking
    step_times = []

    while global_step <= args.max_steps:
        images, _ = next(data_iterator)
        images = images.to(trainer.device)

        # Precise Hardware Synchronization for Accurate Benchmarking
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        losses = trainer.train_step(images)

        if trainer.device.type == "cuda":
            torch.cuda.synchronize()
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

        # Evaluation Pipeline
        if global_step % eval_step_interval == 0 or global_step == args.max_steps:
            print(f"\n--- Running Evaluation Step {global_step} ---")
            eval_dir = os.path.join(args.output_dir, "temp_eval_fast")

            generate_and_save_images(trainer.wrapper, args.num_eval_images, optimal_batch_size, trainer.device,
                                     eval_dir)

            try:
                current_fid = compute_fid(args.dataset_name, eval_dir, trainer.device)
                print(f"Step {global_step} | Fast FID: {current_fid:.4f}")
            except Exception as e:
                print(f"Evaluation Failed: {e}")
                current_fid = float('inf')

            # Calculate and append run performance metrics
            avg_step_time = sum(step_times) / len(step_times)
            metrics_to_log = {**running_losses, "avg_step_time_secs": avg_step_time,
                              "throughput_imgs_sec": optimal_batch_size / avg_step_time}
            logger.log_step(global_step, metrics_to_log, current_fid)
            step_times.clear()  # Reset step timing accumulator for the next window

            checkpoint_dict = {
                "step": global_step,
                "student_state": trainer.wrapper.student.state_dict(),
                "fid": current_fid,
                "mode": args.mode,
                "model_type": args.model_type,
            }
            if args.mode in ["repa", "irepa", "dog"]:
                checkpoint_dict["proj_head_state"] = trainer.wrapper.proj_head.state_dict()

            torch.save(checkpoint_dict, os.path.join(args.output_dir, f"checkpoint_last_{args.model_type}_{args.mode}.pt"))

            if current_fid < best_fid:
                best_fid = current_fid
                torch.save(checkpoint_dict, os.path.join(args.output_dir, f"checkpoint_best_{args.model_type}_{args.mode}.pt"))
                print("🌟 Saved new optimal state checkpoint.")

            shutil.rmtree(eval_dir)
            trainer.wrapper.train()

        global_step += 1

    progress_bar.close()
    print("Execution complete.")


if __name__ == "__main__":
    main()