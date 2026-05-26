# REPA: Representation-based Projection Alignment for Diffusion Models

## Project Overview
This project implements a quantitative comparative study of diffusion model training under different structural alignment methodologies. We train lightweight student diffusion models—exploring both **Transformer (SiT)** and **UNet** architectures—using four distinct training objectives:

1. **Vanilla**: Standard diffusion objective (MSE loss baseline)
2. **REPA**: Token-alignment representation learning (MLP projection)
3. **iREPA**: Spatial-normalization structural alignment (Convolutional projection + Instance-like normalization)
4. **DoG**: Difference-of-Gaussians spectrum matching (Band-pass frequency alignment)

**Primary Research Goal**: Quantitatively measure and compare the trade-offs between training complexity, generation quality (FID), memory consumption, and runtime across these methodologies and architectures on a consumer hardware setup.

---

## Alignment Methods Overview

All non-vanilla methods use a dynamic timestep weighting mechanism: `λ_dynamic = λ * (1.0 - t_norm)`, ensuring stronger alignment at high noise levels (t=1000) and weaker alignment at low noise (t=0).

| Method | Mechanism | Projection Head | Alignment Target (z_0) | Recommended λ |
|--------|-----------|-----------------|------------------------|---------------|
| **Vanilla** | Pure diffusion (Baseline) | None | N/A | N/A |
| **REPA** | Token-wise semantic alignment | 2-Layer MLP (Tokens) | DINOv2 raw tokens | 0.4 |
| **iREPA** | Spatial grid alignment | 3×3 Conv (Spatial) | Spatially normalized DINOv2 | 1.0 |
| **DoG** | Mid-frequency spectrum matching | 3×3 Conv (Spatial) | Band-pass filtered DINOv2 (σ₁-σ₂) | 1.0 |

*Note on DoG Kernel Selection*: To effectively isolate mid-frequency directional energy, tune your Gaussian blur σ values relative to the grid width. The Gaussian kernel parameters heavily depend on image size and composition. Low frequencies (σ ≈ 15-25% of width) and high frequencies (σ ≈ 0.5-1.0) should be subtracted out.

---

## Project Structure
The codebase is heavily modularized to separate training logic, model abstractions, and experiment configuration.

```
repa/
├── __init__.py
├── config.py                  # Centralized ExperimentConfig and CLI arguments
├── main.py                    # Entry point: orchestrates training loop & evaluation
├── train.py                   # DiffusionTrainer: step execution, loss calculation
├── eval.py                    # Inference, DPM-Solver generation, and CleanFID
├── dataset.py                 # CelebA dataloader with automated VAE normalization
├── utils.py                   # Hardware telemetry and ExperimentLogger
├── align/                     # Alignment math and hooking logic
│   ├── hooks.py               # Feature extraction hooks (train-mode only)
│   ├── projection.py          # MLP and Conv projection head definitions
│   └── shape_utils.py         # Token <-> Spatial reshaping helpers
├── models/                    # Architecture abstractions
│   ├── wrapper.py             # REPAWrapper: Manages student, teacher, VAE, and alignment
│   ├── factory.py             # Initialization logic for SiT and UNet models
│   └── vae.py                 # BaseVAE abstraction for future custom VAE integration
├── results/                   # Auto-generated checkpoints, logs, and metrics
└── README.md                  # This documentation
```

---

## Running Experiments

### Running a Single Experiment
To train a single model using a specific alignment method, run the `main.py` script directly. For example, to train a SiT model using the DoG alignment method:

```bash
python main.py \
  --model_type sit \
  --mode dog \
  --lambda_repa 1.0 \
  --output_dir ./results/sit_dog \
  --batch_size 16 \
  --max_steps 30000
```

### Running All 8 Experiments (SiT & UNet)
To execute the full comparative study (SiT and UNet across all 4 modes), you can run the following bash loop. By default, the script automatically finds the optimal hardware batch size.

```
#!/bin/bash
ARCHS=("sit" "unet")
MODES=("vanilla" "repa" "irepa" "dog")

for arch in "${ARCHS[@]}"; do
  for mode in "${MODES[@]}"; do
    
    # Set appropriate lambda weight
    if [ "$mode" == "repa" ]; then
      LAMBDA=0.4
    elif [ "$mode" == "vanilla" ]; then
      LAMBDA=0.0
    else
      LAMBDA=1.0
    fi
    
    echo "Starting Experiment: Model=$arch | Mode=$mode"
    python main.py \
      --model_type $arch \
      --mode $mode \
      --lambda_repa $LAMBDA \
      --output_dir ./results/${arch}_${mode}
      
  done
done
```

### Key Command-Line Arguments
All hyperparameters are managed centrally in `config.py`. Key arguments include:

* `--model_type`: Generative architecture (`sit` or `unet`). Default: `sit`
* `--mode`: Training objective (`vanilla`, `repa`, `irepa`, `dog`). Default: `dog`
* `--lambda_repa`: Alignment loss weight. Default: `1.0`
* `--max_steps`: Total training steps. Default: `30000`
* `--batch_size`: Forces a specific batch size (omitting this triggers auto-detection).
* `--lr`: AdamW learning rate. Default: `1e-4`
* `--num_evals`: Target number of evaluation checkpoints. Default: `40`
* `--num_eval_images`: Images generated per FID calculation. Default: `2000`

---

## Technical Architecture

### 1. Teacher Model (Frozen)
* **Architecture**: DINOv2-Base (`facebook/dinov2-base`)
* **Purpose**: Provides high-quality semantic/perceptual feature references without task-specific fine-tuning.
* **Extraction**: Hidden states from the penultimate layer, CLS token removed.

### 2. Student Models (Trained)
* **SiT (Scalable Integrity Transformer)**: Lightweight `SiT-S/2` configuration (12 blocks, 384 hidden size). Fast convergence, extracted at layer `0.4 * depth`. Features operate in *Token* space.
* **UNet**: Standard diffusers UNet implementation. Extracted at the `mid_block`. Features operate in *Spatial* space.

### 3. Latent Space (Frozen)
* **Architecture**: Stability AI VAE (`stabilityai/sd-vae-ft-mse`)
* **Role**: Compresses 256×256 RGB images into 32×32 4-channel latents, massively reducing VRAM requirements and compute time. Abstraction allows custom VAEs to be plugged in easily via `models/vae.py`.

### 4. Hardware Optimization & Reproducibility
* **Auto-Batching**: The script dry-runs dummy tensors to empirically find the maximum safe batch size for your specific VRAM capacity.
* **Mixed Precision**: Uses `bfloat16` automatically on compatible CUDA GPUs; falls back to `float16` with a GradScaler on older GPUs or Apple Silicon (MPS).
* **TF32**: Enabled natively for faster matrix multiplications on Ampere+ architectures.
* **Memory Management**: Feature extraction hooks automatically disable themselves during inference to prevent memory leaks.

---

## Metrics & Checkpointing

Evaluations run periodically (determined by `max_steps / num_evals`). Outputs are saved to your specified `--output_dir`.

### Logged Data (`experiment_log_{model}_{mode}.json`)
Telemetry and experiment metrics are tracked per evaluation step:
* `loss_diff`, `loss_repa`, `loss_total`: Objective values.
* `fid_score`: Fréchet Inception Distance computed against custom CelebA stats (via CleanFID).
* `avg_step_time_secs`, `throughput_imgs_sec`: Speed benchmarks.
* `ram_usage_mb`, `gpu_memory_peak_mb`: Memory profiling.

### Checkpoints (`.pt`)
The system saves two variants: `checkpoint_last_{...}.pt` and `checkpoint_best_{...}.pt` (lowest FID). They contain:
* `student_state`: Generative model weights.
* `proj_head_state`: Projection head weights (skipped for vanilla).
* `fid` and `step`: Current progress markers.

---

## Next Steps & Future Work

1. **Model & Dataset Scaling**: Upgrade to SiT-L/XL or UNet-Large; expand to multi-domain datasets (ImageNet, CelebA-HQ) to verify alignment benefits generalize across complexities.
2. **Spectrum Matching VAE**: Substitute the default VAE with a custom model trained using spectrum matching to test if matched feature spaces compound quality improvements.
3. **Expanded Metrics**: Implement LPIPS for human-aligned perceptual quality, and track computational efficiency (e.g., `FID improvement / extra FLOPs`).
4. **Alternative Teachers**: Compare DINOv2 against CLIP-ViT (semantic) or SAM (spatial) as feature targets.
5. **Frequency Domain Analysis**: Decompose predictions using FFT/wavelets to quantitatively validate the DoG mid-frequency alignment hypothesis.