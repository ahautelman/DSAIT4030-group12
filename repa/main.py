import argparse
import os
import shutil
import time

import torch
import logging
from tqdm import tqdm
from dataset import get_celeba_dataloader
from models import REPAWrapper
from eval import generate_and_save_images, compute_fid
from utils import ExperimentLogger
from train import DiffusionTrainer
from cleanfid import fid

def prepare_real_fid_stats(data_dir: str, stats_name: str = "celeba_real_256"):
    """
    Checks if real image statistics are already cached. If not, calculates and 
    saves them to disk. This is done once per dataset/resolution.
    """
    # clean-fid stores custom stats natively in its installation directory or a local cache.
    if not fid.test_stats_exists(stats_name, mode="clean"):
        print(f"Real dataset statistics '{stats_name}' not found. Computing now...")
        # Note: clean-fid expects a folder containing images directly. If your dataset uses
        # ImageFolder (subfolders for classes), clean-fid handles it recursively.
        device = ("cuda" if torch.cuda.is_available() 
                  else "mps" if torch.mps.is_available()
                  else "cpu")
        fid.make_custom_stats(stats_name, data_dir, mode="clean", device=device)
        print("Real dataset statistics computed and cached successfully.")
    else:
        print(f"Found cached real dataset statistics for '{stats_name}'.")

def main():
    parser = argparse.ArgumentParser(description="REPA vs Vanilla Diffusion Experiment on CelebA")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to CelebA data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save logs and images")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_repa", action="store_true", help="Enable REPA auxiliary loss")
    parser.add_argument("--lambda_repa", type=float, default=0.1, help="Weight for REPA loss")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Steps between fast FID evaluations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exp_name = "repa" if args.use_repa else "vanilla"
    log_file = os.path.join(args.output_dir, f"{exp_name}_training_log.json")
    logger = ExperimentLogger(log_file)

    print(f"Starting {exp_name.upper()} experiment. Logs will save to {log_file}.")

    # Pre-compute and cache Real Image Distribution for FID
    stats_name = "celeba_real_256"
    prepare_real_fid_stats(args.data_dir, stats_name)

    # Initialize Data and Models
    try:
        dataloader = get_celeba_dataloader(args.data_dir, args.batch_size)
    except FileNotFoundError as e:
        print(e)
        return

    # Initialize Model Wrapper
    wrapper = REPAWrapper()

    # Initialize Trainer
    trainer = DiffusionTrainer(
        model_wrapper=wrapper,
        learning_rate=args.lr,
        use_repa=args.use_repa,
        lambda_repa=args.lambda_repa
    )
    global_step = 0
    start_time = time.time()
    
    # Smoothing dictionary for moving average of losses
    running_losses = {"loss_diff": 0.0, "loss_repa": 0.0, "loss_total": 0.0}

    # Training Loop
    for epoch in range(args.epochs):
        trainer.wrapper.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")                                                                                                                                                
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(trainer.device)
            # Optionally move labels to device if conditioning DiT on classes
            # labels = labels.to(trainer.device) 

            losses = trainer.train_step(images, class_labels=None)

            for k in running_losses.keys():
                running_losses[k] = losses[k]

            progress_bar.set_postfix({
                "Step": global_step,
                "Diff": f"{running_losses['loss_diff']:.4f}",
                "REPA": f"{running_losses['loss_repa']:.4f}" if args.use_repa else "N/A"
            })
            
            global_step += 1
            
            if global_step % args.eval_interval == 0:
                print(f"--- Fast evaluation at Step {global_step} ---")
                eval_dir = os.path.join(args.output_dir, "temp_eval_fast")
                
                # Generate 2k images using DDIM
                generate_and_save_images(
                    wrapper=trainer.wrapper,
                    num_images=2000,
                    batch_size=args.batch_size,
                    device=trainer.device,
                    output_dir=eval_dir,
                )
                
                # Compute FID against cached real stats
                fid_score = compute_fid(stats_name, eval_dir)
                print(f"Step {global_step} | Fast FID (2k): {fid_score:.4f}")
                
                # Log to JSON
                logger.log_step(global_step, running_losses, fid_score)
                
                # Clean up temp image
                shutil.rmtree(eval_dir)
                
                # Save Checkpoint
                ckpt_path = os.path.join(args.output_dir, f"model_{exp_name}_step_{global_step}.pt")
                
                torch.save(trainer.wrapper.student.state_dict(), ckpt_path)
                
                # IMPORTANT: model goes back to training mode
                trainer.wrapper.train()
                
    print("Training completed! Starting final 50k FID evaluation...")
    final_eval_dir = os.path.join(args.output_dir, "temp_eval_final")
    
    generate_and_save_images(
        wrapper=trainer.wrapper,
        num_images=50000,
        batch_size=args.batch_size,
        device=trainer.device,
        output_dir=final_eval_dir,
    )
    
    final_fid = compute_fid(stats_name, final_eval_dir)
    total_time = time.time() - start_time
    
    print(f"======== EXPERIMENT COMPLETED! ========")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Final FID (50k): {final_fid:.4f}")
    
    # Save final logs
    logger.save_final_summary(total_time_sec=total_time, final_fid=final_fid)
    shutil.rmtree(final_eval_dir)                
                

if __name__ == "__main__":
    main()