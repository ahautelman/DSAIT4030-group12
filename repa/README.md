# REPA: Representation-based Projection Alignment for Diffusion Models

## Project Overview
This project implements a quantitative comparative study of diffusion model training under different structural alignment methodologies. We train a lightweight student diffusion model using four distinct training objectives:
1. **Vanilla**: Standard diffusion model training (MSE loss baseline)
2. **REPA**: Token-alignment representation learning (token-wise MLP projection)
3. **iREPA**: Spatial-normalization structural alignment (spatial convolutional projection with instance normalization)
4. **DoG**: Difference-of-Gaussians spectrum matching (band-pass filtering for frequency alignment)

**Primary Research Goal**: Quantitatively measure and compare the trade-offs between training complexity, generation quality (FID), memory consumption, and runtime across these four methodologies on a consumer hardware setup.

---
## Alignment Methods Overview

### 1. **Vanilla (Baseline)**
- **Mechanism**: Pure diffusion objective with MSE loss between predicted and actual noise
- **Equation**: `L_diff = ||ε_pred - ε_true||²`
- **Advantages**: Fast, simple, no teacher model required
- **Use Case**: Baseline for comparison

### 2. **REPA (Representation Extraction and Projection Alignment)**
- **Mechanism**: Token-wise alignment using an MLP projection head with dynamic loss weighting
- **Process**:
  1. Extract intermediate hidden states from student model at layer `0.4 * depth`
  2. Project student features to teacher feature space via 2-layer MLP with GELU activation
  3. Align projected student tokens with teacher tokens (DINOv2, penultimate layer) via cosine similarity
  4. Scale alignment loss dynamically with timestep (stronger at low noise, weaker at high noise)
- **Equation**: `L_repa = -CosineSimilarity(Proj_MLP(h_t), z_0)` (mean across token dimension)
- **Alignment Weighting**: `λ_dynamic = λ * (1.0 - t_norm)` where `t_norm = t / num_timesteps`
- **Advantages**: Lightweight, token-level semantic alignment
- **Recommended λ**: `0.2` (aligns with original paper)
- **Computational Cost**: Minimal (MLP projection only)

### 3. **iREPA (Improved REPA with Spatial Normalization)**
- **Mechanism**: Spatial grid alignment with instance-like normalization and convolutional projection
- **Process**:
  1. Reshape token sequences into 2D spatial grids (H×W format)
  2. Apply 3×3 convolution-based projection head for spatial-aware alignment
  3. Normalize teacher features spatially (mean/std per spatial location)
  4. Align via channel-wise cosine similarity with dynamic timestep weighting
- **Equation**: `L_irepa = -CosineSimilarity(Conv(h_t_spatial), Normalize(z_0_spatial))` (mean across spatial dims)
- **Alignment Weighting**: `λ_dynamic = λ * (1.0 - t_norm)` where `t_norm = t / num_timesteps`
- **Recommended λ**: `0.5` (stronger gradient for convolutional projection head)
- **Advantages**: Preserves spatial structure, reduces token-level noise
- **Computational Cost**: Slightly higher (convolution + normalization)

### 4. **DoG (Difference-of-Gaussians Spectrum Matching)**
- **Mechanism**: Band-pass filtering to align middle-frequency components with dynamic weighting
- **Process**:
  1. Reshape features into 2D spatial grids
  2. Apply two Gaussian blurs with different sigmas (σ₁, σ₂) to isolate mid-frequencies
  3. Compute difference-of-Gaussians: `DoG = Blur(σ₁) - Blur(σ₂)`
  4. Project student features via 3×3 convolution
  5. Align DoG-filtered teacher features with projected student features
  6. Scale alignment loss dynamically with timestep
- **Equation**: `L_dog = -CosineSimilarity(Conv(h_t_spatial), DoG(z_0_spatial))`
- **Alignment Weighting**: `λ_dynamic = λ * (1.0 - t_norm)` where `t_norm = t / num_timesteps`
- **Kernel Selection Guide**: For your teacher model, tune σ values to:
  - **Low frequencies** (global signal): σ ≈ 15-25% of grid width
  - **High frequencies** (noise): σ ≈ 0.5-1.0 (adjacent token blending)
  - Verify PSD: DC component should be zeroed, high-frequency energy declining, mid-frequency energy preserved
- **Recommended λ**: `0.5` (stronger gradient for convolutional projection)
- **Theory**: DoG filter isolates mid-frequency directional energy; improving alignment on these frequencies should enhance perceptual quality
- **Advantages**: Theoretically grounded in frequency domain; potentially better for high-frequency details
- **Computational Cost**: Moderate (two Gaussian blurs per alignment step)
---

## Project Structure & File Organization
```
repa/
├── main.py                    # Entry point: orchestrates training loop & evaluation
├── models.py                  # Core model definitions (REPAWrapper, projection heads)
├── train.py                   # DiffusionTrainer class: implements training step logic
├── eval.py                    # Inference & FID computation for evaluation
├── dataset.py                 # CelebA dataloader with ImageNet normalization
├── utils.py                   # ExperimentLogger for telemetry tracking
├── results/                   # Training outputs (checkpoints, logs, metrics)
├── playbooks/                 # Jupyter notebooks for analysis & visualization
└── README.md                  # This file
../data/
├── celeba/                    # CelebA training images (256×256 crops)
└── celeba.zip                 # Original compressed dataset
```

### Key Components

| File | Responsibility |
|------|-----------------|
| `models.py::REPAWrapper` | Manages teacher (DINOv2), student (SiT), VAE, and projection heads; implements alignment logic |
| `models.py::ProjectionHead` | 2-layer MLP for vanilla REPA token alignment |
| `models.py::iREPAProjectionHead` | 3×3 Conv for spatial alignment (used in iREPA & DoG) |
| `train.py::DiffusionTrainer` | Handles training steps, loss computation, gradient scaling for mixed precision |
| `eval.py::generate_and_save_images` | Generates images via denoising loop using DPM-Solver++ scheduler |
| `eval.py::compute_fid` | Computes FID scores against CelebA statistics via CleanFID |
| `main.py::find_max_batch_size` | Automatic GPU memory profiling to find optimal batch size |
| `main.py::main` | Training loop with periodic evaluation, checkpointing, and metrics logging |
| `utils.py::ExperimentLogger` | JSON-based logging of losses, FID, memory, and runtime per step |

---

## Running Experiments

### Quick Start

```bash
cd repa
# 1. Train vanilla baseline (MSE only)
python main.py --mode vanilla --output_dir ./results/vanilla

# 2. Train with REPA alignment
python main.py --mode repa --lambda_repa 0.4 --output_dir ./results/repa

# 3. Train with iREPA (spatial normalized) alignment
python main.py --mode irepa --output_dir ./results/irepa

# 4. Train with DoG (spectrum matching) alignment
python main.py --mode dog --output_dir ./results/dog
```

### Command-Line Arguments

| Argument | Default     | Description                                                                                                                                             |
|----------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--mode` | `dog`       | Training objective: `vanilla`, `repa`, `irepa`, `dog`                                                                                                   |
| `--output_dir` | `./output`  | Directory to save checkpoints, logs, and metrics                                                                                                        |
| `--max_steps` | `30000`     | Total training steps                                                                                                                                    |
| `--batch_size` | Auto-detect | Batch size (auto-computed if not specified)                                                                                                             |
| `--lr` | `1e-4`      | Learning rate for AdamW optimizer                                                                                                                       |
| `--lambda_repa` | `1.0`       | Alignment loss weight: `L_total = L_diff + λ·L_repa`. Use `0.4` for REPA and `1.0` for iREPA & DoG. Internally weighted by noise level during training. |
| `--num_evals` | `60`        | Target number of evaluation checkpoints (determines eval interval)                                                                                      |
| `--num_eval_images` | `250`       | Number of images generated per evaluation                                                                                                               |
| `--data_dir` | `../data`   | Path to CelebA dataset                                                                                                                                  |
| `--dataset_name` | `celeba`    | Name for CleanFID statistics reference                                                                                                                  |

### Example with Custom Hyperparameters

```bash
python main.py \
  --mode repa \
  --output_dir ./results/repa_exp1 \
  --max_steps 50000 \
  --batch_size 32 \
  --lr 2e-4 \
  --lambda_repa 0.3 \
  --num_evals 80
```

### Monitoring Training
Training outputs are saved to `{output_dir}`:
- `experiment_log_{mode}.json` - Detailed metrics per evaluation step (losses, FID, memory, throughput)
- `checkpoint_last_{mode}.pt` - Latest model checkpoint
- `checkpoint_best_{mode}.pt` - Best checkpoint (lowest FID)
Checkpoints include:
- `student_state`: Student model weights
- `proj_head_state`: Projection head weights (for REPA/iREPA/DoG)
- `fid`: FID score at this checkpoint
- `step`: Training step number

---

## Technical Details

### Teacher Model (Feature Extraction)
- **Architecture**: DINOv2-Base (`facebook/dinov2-base`)
- **Input**: RGB images (3 channels), resized to 224×224
- **Output**: Token features of shape `[B, 256, 768]` (256 patches after removing CLS token, 768-dim embeddings)
- **Purpose**: Provides high-quality semantic/perceptual feature references
- **Training**: Frozen (no gradients)
- **Why DINOv2**: Self-supervised, strong semantic understanding of visual content without task-specific fine-tuning

### Student Model (Generative)
- **Architecture**: SiT-S/2 (Scalable Integrity Transformer) from `BiliSakura/SiT-diffusers`
- **Dimensions**:
  - Hidden size: 384
  - Attention heads: 6
  - Head dimension: 64 (384 ÷ 6)
  - Patch size: 2
  - Depth: 12 transformer blocks
  - Input channels: 4 (VAE latent)
  - Output channels: 8 (noise + variance prediction)
- **Input**: Noisy latents from VAE encoder, timestep embedding, class labels
- **Training**: Full gradients via AdamW optimizer
- **Why SiT**: Lightweight yet capable architecture, suitable for training on consumer hardware; fast convergence compared to UNet-based approaches

### VAE (Latent Space Encoder/Decoder)
- **Architecture**: Stability AI VAE (`stabilityai/sd-vae-ft-mse`)
- **Input**: RGB images (256×256)
- **Latent Output**: 4-channel latents (256×256 → 32×32)
- **Scaling Factor**: Applied to latents for stable training
- **Purpose**: Compresses images to latent space; reduces memory/compute during training
- **Training**: Frozen (no gradients)

### Alignment Architecture

#### **REPA Projection Head**

```python
StudentFeatures (B, N, 384) 
  → Linear(384 → 384) + GELU 
  → Linear(384 → 768) 
  → ProjectedFeatures (B, N, 768)
```

Maps token features from student space to teacher space via 2 fully-connected layers.

#### **iREPA & DoG Projection Head**

```python
StudentFeatures (B, 384, H, W) 
  → Conv2d(384 → 768, kernel=3, padding=1) 
  → ProjectedFeatures (B, 768, H, W)
```
Spatially-aware projection via convolutional layer, preserving local spatial structure.

### Training Pipeline
1. **Data Loading**:
   - CelebA images resized to 256×256 and center-cropped
   - Normalized to [-1, 1] for VAE (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
   - Batched with automatic memory-based batch size optimization
2. **Noise Scheduling** (DDPM):
   - 1000 timesteps from clean to pure noise
   - Linear noise schedule
   - Random timestep sampling per batch
3. **Forward Process**:
   - Encode images to VAE latents: `z_0 = VAE.encode(x_0)`
   - Extract teacher features: `z_teacher = DINOv2(x_0)` (frozen)
   - Add Gaussian noise at random timestep: `z_t = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε`
   - Student predicts noise: `ε̂ = Student(z_t, t, class_label)`
4. **Loss Computation**:
   ```
   L_diff = MSE(ε̂, ε)                           # Diffusion objective
   L_align = -CosineSimilarity(Proj(h_t), z_ref)  # Alignment objective (mode-dependent)
   L_total = L_diff + λ · L_align
   ```
5. **Optimization**:
   - Optimizer: AdamW
   - Learning rate: 1e-4 (default)
   - Gradient accumulation: None (direct step)
   - Mixed precision: 
     - **CUDA with bfloat16 support**: Automatic mixed precision with bfloat16, no scaler
     - **CUDA without bfloat16**: float16 with gradient scaling
     - **MPS**: float16 with gradient scaling
     - **CPU**: float32 (no mixed precision)
6. **Inference & Evaluation**:
   - Scheduler: DPM-Solver++ multistep (20 steps)
   - Generation: 2k images per evaluation
   - Metric: FID score against CelebA reference statistics

### FID Measurement Method
- **Tool**: CleanFID (`cleanfid` library)
- **Reference**: Pre-computed CelebA 256×256 statistics
- **Computation**: 
  1. Generate 2000 images from trained student model
  2. Compute Inception-v3 features for generated images
  3. Compute FID distance between generated and reference distributions
  4. Auto-downloads CelebA stats if not cached
- **Timing**: ~2-3 minutes per evaluation (varies by device)

### Hardware Optimizations

#### **CUDA Optimizations**
- `torch.backends.cuda.matmul.allow_tf32 = True`: Faster matrix multiplications using Tensor Float 32
- `torch.backends.cudnn.allow_tf32 = True`: Faster cuDNN operations
- `torch.cuda.synchronize()`: Accurate timing for benchmarking
- xFormers memory-efficient attention (if available): Reduces memory for multi-head attention
- Gradient scaling (float16): Prevents underflow in half-precision training

#### **MPS (Apple Silicon) Optimizations**
- float16 computation with gradient scaling

#### **Automatic Batch Size Optimization**
```python
find_max_batch_size(trainer, starting_batch=64)
```

Finds maximum batch size that fits in VRAM:
- Starts with a large batch size (e.g., 64)
- On out-of-memory error, reduces batch size by 32 and retries
- Returns 70% of max size for safety margin

---

## Metrics & Logging

### Per-Step Metrics (Logged Every Evaluation)

| Metric | Description | Unit |
|--------|-------------|------|
| `loss_diff` | MSE noise prediction loss | - |
| `loss_repa` | Alignment loss (mode-specific) | - |
| `loss_total` | Combined loss | - |
| `fid_score` | Fréchet Inception Distance | lower is better |
| `avg_step_time_secs` | Average training step duration | seconds |
| `throughput_imgs_sec` | Images processed per second | imgs/s |
| `ram_usage_mb` | CPU RAM consumption | MB |
| `gpu_memory_peak_mb` | Peak GPU VRAM usage | MB |

### Output Format
Metrics are saved in JSON format (`experiment_log_{mode}.json`):

```json
[
  {
    "global_step": 500,
    "loss_diff": 0.0245,
    "loss_repa": -0.8934,
    "loss_total": 0.1135,
    "fid_score": 45.23,
    "ram_usage_mb": 8234.56,
    "gpu_memory_peak_mb": 12456.78,
    "avg_step_time_secs": 0.512,
    "throughput_imgs_sec": 62.5
  },
  ...
]
```

---

## Next Steps & Improvements

### 1. **Model & Dataset Scaling**
- **Current Compromise**: SiT-S (small, fast) trained on CelebA for rapid iteration
- **Improvement**: 
  - Compare against larger models (SiT-M, SiT-L) to verify alignment benefits generalize
  - Extend to multi-domain datasets (ImageNet 64×64, CelebA-HQ)
  - Measure if FID improvements persist across different model capacities

### 2. **Spectrum Matching Integration**
- **Concept**: Use a teacher/VAE trained with spectrum matching techniques
- **Hypothesis**: Spectrum-matched teacher features should provide better high-frequency alignment targets
- **Implementation**:
  - Train VAE with spectrum matching loss instead of MSE
  - Explore frequency-aware projection heads (DCT-based, wavelets)
  - Measure FID improvement over baseline spectrum matching

### 3. **Expanded Metrics**
- **Current**: FID only
- **Proposed**:
  - **LPIPS (Learned Perceptual Image Patch Similarity)**: Human-aligned perceptual quality
  - **SSIM/PSNR**: Low-level pixel-level similarity

### 4. **Computational Cost Analysis**
- **Current**: Runtime measurements only
- **Proposed**:
  - **FLOPs Counting**: Measure FLOPs per forward/backward pass for each mode
  - **Cost-Benefit Analysis**: Plot FID improvement vs. additional compute/memory
  - **Cost Formula**: `Efficiency = FID_improvement / (Extra_FLOPs + Extra_Memory + Extra_Time)`

### 5. **Teacher Model Exploration**
- Compare alternative teachers:
  - **CLIP-ViT**: Larger (more compute) but strong semantic understanding
  - **SAM (Segment Anything)**: Spatial awareness for structure alignment
  - **Spectral-matched teacher**: Custom teacher optimized for spectrum alignment
- Measure impact of teacher quality on student learning

### 6. **Frequency Domain Analysis**
- Decompose predictions using FFT/wavelets
- Analyze which frequency bands improve most per method
- Visualize frequency alignment via heatmaps
- Validate DoG hypothesis quantitatively

---
## Reproducibility & System Requirements

### Recommended Hardware
- **GPU**: NVIDIA with CUDA 11.8+ (bfloat16 support recommended)
  - **_TODO_**: Tested on RTX 3090/4090 (24GB VRAM)
- **RAM**: 16GB+ system memory
- **Disk**: 50GB for dataset + results
- **Time**: TODO estimate training time on CUDA

### Reproducibility Notes
1. Hardware differences (CPU/GPU/MPS) may cause minor metric variations
2. Batch size affects gradient estimates; note batch size in comparisons
3. CleanFID stats should be cached consistently across runs

---

## References & Theory (TODO: add correct links)

### Alignment Methodology
- **REPA**: Token-level alignment for knowledge distillation
- **iREPA**: Spatial normalization based on batch statistics
- **DoG (Spectrum Matching)**: Frequency-domain alignment hypothesis
  - Theory: Different image components exist at different frequency scales
  - Hypothesis: Mid-frequency alignment (DoG) improves perceptual quality
  - Validation: FID improvement would support this hypothesis

### Models Used
- **SiT (Scalable Integrity Transformer)**: Ma et al., "Integrity Transformer"
- **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision"
- **Diffusers Library**: Multi-method diffusion model implementations

### Relevant Techniques
- **Diffusion Models**: Ho et al., "Denoising Diffusion Probabilistic Models"
- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network"
- **Feature Alignment**: Learnable projection heads for feature space alignment
