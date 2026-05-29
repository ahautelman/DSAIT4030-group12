import os
import sys

import torchvision
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from vae import VAE

from dataset_loader import load_dataset
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = "celeba"

resume_path = "/home/remcohuijsen/DSAIT4030-group12/vae/step_100000.pt"
vae = VAE(mode="kl").to(DEVICE)
checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False)

vae.load_state_dict(checkpoint["vae"], strict=False)
print("VAE loaded successfully.")

NUM_WORKERS = 0
BATCH_SIZE = 4


train_set = load_dataset(DATASET, split="train")

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
    persistent_workers=True if NUM_WORKERS > 0 else False,
)

for batch_idx, batch in enumerate(train_loader):

    x = batch["images"].to(DEVICE, non_blocking=True)

    x_hat = vae.decode(vae.encode(x))

    torchvision.utils.save_image(x, f"original_batch_{batch_idx}.png")
    torchvision.utils.save_image(x_hat, f"reconstructed_batch_{batch_idx}.png")

    print(f"Batch {batch_idx}:")
    print(f"  x:     {x.shape}")       # (B, 3, 256, 256)
    print(f"  x_hat: {x_hat.shape}")   # (B, 3, 256, 256)

    break # just one batch for testing
    

