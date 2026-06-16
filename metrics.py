# =====================================================================
# 1. ENVIRONMENT SETUP
# =====================================================================
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/vae" )

# =====================================================================
# 2. IMPORTS
# =====================================================================
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

# CHANGED: Since /kaggle/working/vae is directly in your path, 
# you can import the file 'vae.py' directly as a module!
import vae
from vae import VAE

from dataset_loader import load_dataset

# =====================================================================
# 3. CONFIGURATION
# =====================================================================
CHECKPOINTS = {
    "kl":  "vae/runs/run_name/step_200000.pt",
    "esm": "vae/runs/run_name/step_200000.pt",
    "dsm": "vae/runs/run_name/step_100000.pt",
    "esm": "vae/runs/celeba_esm_20260606_105940/checkpoints/step_200000.pt" # Inverse Spectrum
}
BATCH_SIZE  = 16
NUM_WORKERS = 0
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# =====================================================================
# 4. HELPER FUNCTIONS
# =====================================================================
def load_vae(checkpoint_path, mode):
    vae_instance = VAE(mode=mode).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    vae_instance.load_state_dict(ckpt["vae"])
    vae_instance.eval()
    print(f"Loaded {mode.upper()} VAE from {checkpoint_path} (step {ckpt['step']})")
    return vae_instance

def evaluate_one(vae_model, val_loader, mode):
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(DEVICE)
    fid_metric  = FrechetInceptionDistance(normalize=True).to(DEVICE)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating {mode.upper()}"):
            real = batch["images"].to(DEVICE)
            z = vae_model.encode(real)
            recon = vae_model.decode(z).clamp(-1, 1)
            
            # Metrics calculation
            psnr_metric.update(recon, real)
            ssim_metric.update(recon, real)
            
            # Rescale from [-1, 1] to [0, 1] as float for torchmetrics FID
            real_normalized = (real + 1) / 2.0
            recon_normalized = (recon + 1) / 2.0
            
            fid_metric.update(real_normalized, real=True)
            fid_metric.update(recon_normalized, real=False)

    return {
        "rFID": fid_metric.compute().item(),
        "PSNR": psnr_metric.compute().item(),
        "SSIM": ssim_metric.compute().item(),
    }

# =====================================================================
# 5. EXECUTION PIPELINE
# =====================================================================
val_set = load_dataset("celeba", split="validation")
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
print(f"Validation set size: {len(val_set)}")

all_results = {}
for mode, ckpt_path in CHECKPOINTS.items():
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path} — skipping")
        continue
    vae_model = load_vae(ckpt_path, mode)
    all_results[mode] = evaluate_one(vae_model, val_loader, mode)

# Print Final Results Table
print(f"\n{'='*50}")
print(f"{'Mode':<8} {'rFID':>8} {'PSNR':>8} {'SSIM':>8}")
print(f"{'-'*50}")
for mode, metrics in all_results.items():
    print(f"{mode.upper():<8} {metrics['rFID']:>8.2f} {metrics['PSNR']:>8.2f} {metrics['SSIM']:>8.3f}")
print(f"{'='*50}")
