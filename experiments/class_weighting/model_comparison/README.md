# Model Comparison — Class-Weighted 2D Architectures

Compare **UNet**, **ResNet**, **Swin Transformer**, and **ViT-V-Net** (all 2D)
using the best loss configuration from the class weighting experiment:

> **Balanced Softmax Tversky τ=1.0** (α=0.6, β=0.4)

## Quick Start

### On Blanca Biokem (SLURM)

```bash
# Submit all 4 models (each gets 2× A100)
./launch_all_blanca.sh

# Or submit individually
./run.sh submit unet
./run.sh submit swin

# Check status
./run.sh status
```

### Locally (Rocinante)

```bash
# Run one model
./run.sh local unet

# Run all 4 sequentially
./run.sh local-all
```

### After Training

```bash
# View comparison table
./run.sh summary

# Generate analysis plots
python analyze_results.py

# TensorBoard
./run.sh tensorboard
```

## Experiment Design

| Parameter | Value |
|-----------|-------|
| **Loss** | Balanced Softmax Tversky (τ=1.0, α=0.6, β=0.4) |
| **Classes** | 5: nuc, mito_mem, er_mem, pm, golgi_mem |
| **Batch size** | 24 (same for all models) |
| **Input** | (1, 256, 256) at 8nm isotropic |
| **Epochs** | 100 |
| **Iter/epoch** | 200 |
| **Optimizer** | AdamW (weight_decay=1e-4) |
| **Scheduler** | OneCycleLR (stepped per batch) |
| **AMP** | Enabled (TF32 on A100) |
| **DDP** | 2× A100 per model on Blanca |

### Model-Specific Settings

| Model | Type | LR | Notes |
|-------|------|----|-------|
| UNet 2D | CNN | 1e-4 | Baseline, InstanceNorm |
| ResNet 2D | CNN | 1e-4 | 6 blocks, 2 downsampling |
| Swin 2D | Transformer | 5e-5 | [2,2,6,2] depths, 96 embed |
| ViT-V-Net 2D | Transformer | 5e-5 | 12 layers, 768 hidden, patch=16 |

## Metrics Tracked

- **Per-class**: Dice, IoU, Precision, Recall
- **Aggregate**: Mean Dice, Mean IoU
- **Training**: Loss curves, LR schedule, gradient norms
- **Visualization**: Fixed samples (shared across models), feature maps

## File Structure

```
model_comparison/
├── config.py                     # All hyperparameters
├── train.py                      # Unified training script
├── analyze_results.py            # Generate plots
├── run.sh                        # Launcher
├── launch_all_blanca.sh          # Submit all 4 to SLURM
├── train_unet_blanca.sbatch      # SBATCH: UNet
├── train_resnet_blanca.sbatch    # SBATCH: ResNet
├── train_swin_blanca.sbatch      # SBATCH: Swin
├── train_vit_blanca.sbatch       # SBATCH: ViT
├── README.md                     # This file
├── checkpoints/                  # Best + periodic checkpoints
├── results/                      # JSON results per model
│   └── plots/                    # Analysis plots
├── runs/                         # TensorBoard event files
├── visualizations/               # Prediction + feature maps
├── metrics/                      # MetricsTracker outputs
└── features/                     # Extracted feature maps
```

## Why This Loss?

The **Balanced Softmax Tversky τ=1.0** was the winner from the
[class weighting experiment](../EXPERIMENT_DETAILS.md) (Mean Dice = 0.5711),
beating 14 other strategies including inverse-frequency, Class-Balanced,
and Seesaw variants. It works by shifting logits based on class frequency
priors, giving rare classes (golgi_mem, pm) a detection advantage without
the instability of high static weights.
