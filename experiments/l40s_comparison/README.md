# L40S Model Comparison Experiment

**Branch**: `feature/l40s-model-comparison`  
**Cluster**: UNC Longleaf — `l40-gpu` partition (L40S 48GB)  
**Status**: Ready to submit

## Overview

Head-to-head comparison of **8 segmentation models** (4×2D + 4×3D) on the
CellMap FIB-SEM **14-class** organelle segmentation task, using a unified loss
function that combines the best findings from three prior experiments.

### Models

| Model | Dim | Batch/GPU | GPUs | Grad Accum | Eff. Batch |
|-------|-----|-----------|------|------------|------------|
| UNet | 2D | 26 | 2 | 4 | 208 |
| ResNet | 2D | 28 | 2 | 4 | 224 |
| Swin | 2D | 18 | 2 | 4 | 144 |
| ViT | 2D | 10 | 2 | 4 | 80 |
| UNet | 3D | 5 | 2 | 4 | 40 |
| ResNet | 3D | 5 | 2 | 4 | 40 |
| Swin | 3D | 2 | 4 | 6 | 48 |
| ViT | 3D | 1 | 2 | 8 | 16 |

### Loss Function: Balanced Softmax Partial Tversky

Combines findings from three experiments:

1. **Class Weighting** (Rocinante): `balanced_softmax_tau_1.0` was the winner
   (0.5711 mean Dice) across 15 configurations
2. **Loss Optimization** (Shenron): Per-class Tversky with α=0.6, β=0.4
   outperforms Dice+BCE and Focal for membrane/lumen class imbalance
3. **Auto3DSeg** (Longleaf): Partial annotation masking required for 14-class
   training — each crop only has annotations for a subset of classes

The loss applies:
- **Balanced Softmax logit adjustment**: `logit_c -= τ * (log(n_c) - mean(log(n)))`
  with τ=1.0, boosting rare class logits before loss computation
- **Per-class Tversky**: α=0.6 (FP weight), β=0.4 (FN weight) — penalizes
  false positives slightly more
- **Partial annotation masking**: Channels with NaN targets are excluded from loss

### Hardware: L40S 48GB

From MONAI Auto3DSeg profiling (Job 30343731):
- 48GB VRAM (45GB usable), Ada Lovelace architecture
- 512GB RAM required (200GB insufficient for 14-class label expansion)
- 11-day max wall time
- TF32 + BFloat16 AMP enabled

## Quick Start

```bash
# Submit all 8 models
bash experiments/l40s_comparison/slurm/launch_all.sh

# Submit only 2D models
bash experiments/l40s_comparison/slurm/launch_all.sh 2d

# Submit only 3D models
bash experiments/l40s_comparison/slurm/launch_all.sh 3d

# Submit a single model
sbatch experiments/l40s_comparison/slurm/train_unet_2d.sbatch

# Check job status
squeue -u $USER
```

## Directory Structure

```
experiments/l40s_comparison/
├── config.py          # L40S-optimized configuration
├── losses.py          # BalancedSoftmaxPartialTverskyLoss
├── train.py           # Training script (DDP, AMP, early stopping)
├── README.md
├── checkpoints/       # Best + periodic model weights
├── tensorboard/       # TensorBoard logs (per model)
├── visualizations/    # Input/GT/prediction comparison figures
├── metrics/           # Per-class Dice, loss curves
├── results/           # Final summary
└── slurm/
    ├── launch_all.sh          # Submit all 8 jobs
    ├── train_unet_2d.sbatch
    ├── train_resnet_2d.sbatch
    ├── train_swin_2d.sbatch
    ├── train_vit_2d.sbatch
    ├── train_unet_3d.sbatch
    ├── train_resnet_3d.sbatch
    ├── train_swin_3d.sbatch
    ├── train_vit_3d.sbatch
    └── logs/
```

## Key Features

- **Early stopping**: patience=20 epochs on validation Dice
- **Checkpoint resumption**: `--resume path/to/checkpoint.pth`
- **Online class frequency tracking**: Logged to TensorBoard every 50 batches
- **Per-class Dice logging**: All 14 classes tracked per epoch
- **Logit adjustment visualization**: TensorBoard shows how much each class is
  boosted/suppressed by the balanced softmax

## Monitoring

```bash
# TensorBoard (on a login node with port forwarding)
tensorboard --logdir experiments/l40s_comparison/tensorboard --port 6007

# Check a specific job's output
tail -f experiments/l40s_comparison/slurm/logs/unet_2d_<JOBID>.out

# GPU utilization on the running node
ssh <node> nvidia-smi
```

## Training Parameters

| Parameter | 2D | 3D |
|-----------|----|----|
| Input shape | (1, 256, 256) | (32, 256, 256) |
| Max epochs | 200 | 200 |
| Iters/epoch | 1000 | 500 |
| num_workers | 8 | 0 |
| AMP dtype | BFloat16 | BFloat16 |
| Scheduler | OneCycleLR (cosine) | OneCycleLR (cosine) |
| Optimizer | AdamW (wd=1e-4) | AdamW (wd=1e-4) |
| Grad clip | 1.0 | 1.0 |

## Related Experiments

- `experiments/class_weighting/` — Balanced Softmax winner identification
- `experiments/loss_optimization/` — Tversky α/β optimization
- `auto3dseg/` — MONAI Auto3DSeg (Job 30343731, RUNNING)
- `experiments/model_comparison/` — Original H100-targeted comparison
