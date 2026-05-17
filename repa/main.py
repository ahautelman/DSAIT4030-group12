import argparse
import os
import shutil
import gc
from pathlib import Path

import cleanfid
from cleanfid import fid
import torch
from tqdm import tqdm
from dataset import get_celeba_dataloader
from models import REPAWrapper
from eval import generate_and_save_images, compute_fid
from utils import ExperimentLogger
from train import DiffusionTrainer


def find_max_batch_size(trainer, starting_batch=64):
    """Finds optimal batch size by probing with real image dimensions."""
    current_batch = starting_batch
    device = trainer.device
    print(f"Starting automatic batch size finder from {starting_batch}...")

    while current_batch > 0:
        try:
            dummy_images = torch.randn((current_batch, 3, 256, 256), device=device)
            trainer.optimizer.zero_grad(set_to_none=True)
            _ = trainer.train_step(dummy_images)

            print(f"Success! Max viable batch size found: {current_batch}")

            trainer.optimizer.zero_grad(set_to_none=True)
            del dummy_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

            safe_batch = max(1, int(current_batch * 0.70))
            print(f"Applying safety margin. Using batch size: {safe_batch}")
            return safe_batch

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
                current_batch -= 8
                trainer.optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
            else:
                raise e

    raise RuntimeError("Could not find a viable batch size.")


def get_infinite_dataloader(dataloader):
    """Yields batches indefinitely for step-based training."""
    while True:
        for batch in dataloader:
            yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset_name", type=str, default="celeba",
                        help="Name of the dataset for CleanFID statistics")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--max_steps", type=int, default=8_000, help="Total number of training steps")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_repa", action="store_true")
    parser.add_argument("--lambda_repa", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=2000,
                        help="Number of training images between evaluations")
    parser.add_argument("--num_eval_images", type=int, default=250, help="Number of images to generate for Fast FID")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize JSON logger
    log_file = os.path.join(args.output_dir, "experiment_log.json")
    logger = ExperimentLogger(log_file)

    # 1. Initialize Wrapper
    wrapper = REPAWrapper()

    # 2. Compile Student Model (Strictly for CUDA, PyTorch 2.0+)
    if torch.__version__ >= "2.0.0" and torch.cuda.is_available():
        print("Compiling Student Model via torch.compile for maximum CUDA throughput...")
        wrapper.student = torch.compile(wrapper.student)

    # 3. Initialize Trainer
    trainer = DiffusionTrainer(wrapper, args.lr, args.use_repa, args.lambda_repa)

    # 4. Auto Batch Sizing
    if args.batch_size:
        optimal_batch_size = args.batch_size
    else:
        optimal_batch_size = find_max_batch_size(trainer, starting_batch=64)

    # Convert Image Interval to Step Interval
    eval_step_interval = max(1, args.eval_interval // optimal_batch_size)
    print(f"Evaluation interval: ~{args.eval_interval} images -> Every {eval_step_interval} steps.")

    # 5. Initialize DataLoader & Infinite Iterator
    dataloader = get_celeba_dataloader(args.data_dir, optimal_batch_size)
    data_iterator = get_infinite_dataloader(dataloader)

    # CleanFID Real Statistics Caching ---
    print(f"Checking CleanFID statistics for '{args.dataset_name}'...")

    # cleanfid saves stats in one of these two locations depending on OS/environment
    stat_path_1 = os.path.join(os.path.dirname(cleanfid.__file__), "stats", f"{args.dataset_name}_clean_custom_na.npz")
    stat_path_2 = os.path.expanduser(f"~/.cache/cleanfid/stats/{args.dataset_name}_clean_custom_na.npz")

    if not (os.path.exists(stat_path_1) or os.path.exists(stat_path_2)):
        print(f"Custom statistics not found. Computing real image statistics...")
        print("This may take a few minutes for 200k images, but it only happens once!")

        # IMPORTANT: cleanfid does NOT search subfolders recursively.
        # It needs the direct path to the folder containing the actual .jpg/.png files.
        image_dir = Path(args.data_dir) / args.dataset_name

        fid.make_custom_stats(
            name=args.dataset_name,
            fdir=str(image_dir),
            device=trainer.device,
            num_workers=0   # to prevent multiprocessing crash
        )
        print("Statistics cached successfully!")
    else:
        print("Real image statistics found locally!")

    # Training State
    global_step = 1
    best_fid = float('inf')
    running_losses = {"loss_diff": 0.0, "loss_repa": 0.0, "loss_total": 0.0}

    trainer.wrapper.train()
    progress_bar = tqdm(total=args.max_steps, desc="Training (Steps)")

    while global_step <= args.max_steps:
        # Fetch next batch
        images, _ = next(data_iterator)
        images = images.to(trainer.device)

        # Forward & Backward Pass
        losses = trainer.train_step(images)

        for k in running_losses.keys():
            running_losses[k] = losses[k]

        progress_bar.set_postfix(
            {"Diff": f"{running_losses['loss_diff']:.4f}", "REPA": f"{running_losses['loss_repa']:.4f}"})
        progress_bar.update(1)

        # Evaluation & Checkpointing Phase
        if global_step % eval_step_interval == 0 or global_step == args.max_steps:
            print(f"\n--- Running Evaluation at Step {global_step} ---")
            eval_dir = os.path.join(args.output_dir, "temp_eval_fast")

            # Generate images
            generate_and_save_images(
                trainer.wrapper,
                args.num_eval_images,
                optimal_batch_size,
                trainer.device,
                eval_dir
            )

            # Compute Fast FID
            try:
                current_fid = compute_fid(args.dataset_name, eval_dir, trainer.device)
                print(f"Step {global_step} | Fast FID ({args.num_eval_images} imgs): {current_fid:.4f}")
            except Exception as e:
                print(f"FID computation failed (ensure real stats are cached): {e}")
                current_fid = float('inf')

            # Log metrics to JSON
            logger.log_step(global_step, running_losses, current_fid)

            # Package Weights
            checkpoint_dict = {
                "step": global_step,
                "student_state": trainer.wrapper.student.state_dict(),
                "fid": current_fid
            }
            if args.use_repa:
                checkpoint_dict["proj_head_state"] = trainer.wrapper.proj_head.state_dict()

            # Save 'Last' Checkpoint
            torch.save(checkpoint_dict, os.path.join(args.output_dir, "checkpoint_last.pt"))

            # Save 'Best' Checkpoint
            if current_fid < best_fid:
                best_fid = current_fid
                torch.save(checkpoint_dict, os.path.join(args.output_dir, "checkpoint_best.pt"))
                print(f"🌟 New best FID! Checkpoint saved.")

            # Cleanup and revert to train mode
            shutil.rmtree(eval_dir)
            trainer.wrapper.train()

        global_step += 1

    progress_bar.close()
    print("Training Complete.")


if __name__ == "__main__":
    main()