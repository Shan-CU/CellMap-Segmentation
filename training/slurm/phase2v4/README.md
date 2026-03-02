# Phase 2 v4: Architecture Comparison

**9 models × 1 configuration each** — fresh restart with researched optimal hyperparameters.

## Quick Start

```bash
# On Longleaf login node:
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# 1. Archive old v2/v3 runs & kill their jobs (run once)
bash training/slurm/phase2v4/archive_v2v3.sh

# 2. Launch all 9 jobs
bash training/slurm/phase2v4/launch_all.sh

# 3. Monitor
squeue -u $USER
tensorboard --logdir runs/ablation --bind_all
```

## Models

| Script | GPU | Model | Params | Optimizer | LR | WD | Notes |
|--------|-----|-------|--------|-----------|----|----|-------|
| `resnet_2d.sbatch` | L40S | ResNet 2D | ~8M | RAdam | 1e-4 | 0 | InstanceNorm default |
| `unet_2d.sbatch` | L40S | UNet 2D | ~31M | RAdam | 1e-4 | 0 | +InstanceNorm +dropout=0.1 |
| `swin_2d.sbatch` | L40S | Swin V2 2D | ~36M | AdamW | 1e-4 | 0.05 | Swin V2 paper config |
| `vit_2d.sbatch` | L40S | ViT-V-Net 2D | ~105M | AdamW | 1e-4 | 0.01 | DeiT-style regularization |
| `resnet_3d.sbatch` | L40S | ResNet 3D | ~8M | RAdam | 4e-4 | 0 | LR scaled (base 1e-4@bs=2) |
| `unet_3d.sbatch` | L40S | UNet 3D | ~31M | RAdam | 4e-4 | 0 | +InstanceNorm +dropout=0.1 |
| `segresnet_3d.sbatch` | A100 | SegResNetDS | ~87M | AdamW | 2e-4 | 1e-5 | MONAI template + DeepSup, bs=4 |
| `swinunetr_3d.sbatch` | A100 | SwinUNETR | ~62M | AdamW | 4e-4 | 1e-5 | MONAI template + GradCheckpoint |
| `vitnet_3d.sbatch` | A100 | ViT-V-Net 3D | ~28M | RAdam | 4e-4 | 0 | Paper config (wd=0) |

## Common Settings (all 9 jobs)

- **Loss**: `dice_bce`
- **EMA**: 0.999
- **Foreground mask**: ON
- **Weighted sampler**: ON
- **AMP**: ON
- **bias_init**: -3.0 (RetinaNet focal prior — prevents BCE collapse)
- **Gradient clipping**: max_norm=1.0
- **Scheduler**: Cosine with warmup
- **Validation**: Every 5 epochs, per-class Dice (48 classes), 600s time limit

### 2D Training
- Patches: 256×256, Batch: 8, Epochs: 100, Iters/epoch: 1000
- LR: 1e-4 (no scaling), Warmup: 10 epochs

### 3D Training
- Crops: 96³ @ 8nm, Batch: 8 (or 4 for SegResNetDS), Epochs: 1000, Iters/epoch: 300
- LR: Scaled linearly (base 1e-4 @ batch=2), Warmup: 34 epochs

## Key Fixes from v2/v3

1. **bias_init=-3.0**: Prevents BCE collapse (3/4 2D models had zero dice in v2)
2. **SegResNetDS reconfigured**: 5-level encoder, deep supervision, InstanceNorm (was 4-level, no DS, BatchNorm → only 20M params instead of 87M)
3. **UNet InstanceNorm**: Switched from BatchNorm (unstable at small batch sizes)
4. **Architecture-specific optimizers**: AdamW+wd for transformers, RAdam for CNNs
5. **SwinUNETR gradient checkpointing**: Enables batch=8 on A100

## Research References

- **SegResNetDS**: MONAI Auto3DSeg template (`hyper_parameters.yaml`)
- **SwinUNETR**: MONAI Auto3DSeg SwinUNETR template
- **Swin V2**: Liu et al. CVPR 2022
- **ViT-V-Net**: Chen et al. MIDL 2021
- **bias_init**: Lin et al. "Focal Loss" (RetinaNet), ICCV 2017
