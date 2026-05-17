import argparse
import os
import shutil
import gc
import torch
from tqdm import tqdm
from dataset import get_celeba_dataloader
from models import REPAWrapper
from eval import generate_and_save_images, compute_fid
from utils import ExperimentLogger
from train import DiffusionTrainer
from cleanfid import fid


def find_max_batch_size(trainer, starting_batch=64):
    """Finds optimal batch size by probing with real image dimensions."""
    current_batch = starting_batch
    device = trainer.device
    print(f"Starting automatic batch size finder from {starting_batch}...")

    while current_batch > 0:
        try:
            # Full 256x256 pixel images to test the entire pipeline
            dummy_images = torch.randn((current_batch, 3, 256, 256), device=device)

            trainer.optimizer.zero_grad()
            _ = trainer.train_step(dummy_images)

            print(f"Success! Max viable batch size found: {current_batch}")

            trainer.optimizer.zero_grad()
            del dummy_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

            # Apply safety margin (65-70% of max capacity)
            safe_batch = max(1, int(current_batch * 0.70))
            print(f"Applying safety margin. Using batch size: {safe_batch}")
            return safe_batch

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
                current_batch -= 2  # Step down slowly
                trainer.optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
            else:
                raise e

    raise RuntimeError("Could not find a viable batch size.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_repa", action="store_true")
    parser.add_argument("--lambda_repa", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=2500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize Wrapper
    wrapper = REPAWrapper()

    # 2. Compile Student Model (If on PyTorch 2.0+ and CUDA)
    if torch.__version__ >= "2.0.0" and torch.cuda.is_available():
        print("Compiling Student Model via torch.compile...")
        wrapper.student = torch.compile(wrapper.student)

    # 3. Initialize Trainer
    trainer = DiffusionTrainer(wrapper, args.lr, args.use_repa, args.lambda_repa)

    # 4. Auto Batch Sizing
    optimal_batch_size = find_max_batch_size(trainer, starting_batch=64)

    # 5. Initialize DataLoader
    dataloader = get_celeba_dataloader(args.data_dir, optimal_batch_size)

    global_step = 0
    running_losses = {"loss_diff": 0.0, "loss_repa": 0.0, "loss_total": 0.0}

    # Training Loop
    for epoch in range(args.epochs):
        trainer.wrapper.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(trainer.device)

            losses = trainer.train_step(images)

            for k in running_losses.keys():
                running_losses[k] = losses[k]

            progress_bar.set_postfix({"Step": global_step, "Diff": f"{running_losses['loss_diff']:.4f}"})
            global_step += 1

            if global_step % args.eval_interval == 0:
                eval_dir = os.path.join(args.output_dir, "temp_eval_fast")

                # Dropped to 1,000 images for speed via DPMSolver
                generate_and_save_images(trainer.wrapper, 1000, optimal_batch_size, trainer.device, eval_dir)

                # Compute FID and clean up...
                # fid_score = compute_fid(...)
                shutil.rmtree(eval_dir)
                trainer.wrapper.train()


if __name__ == "__main__":
    main()
    