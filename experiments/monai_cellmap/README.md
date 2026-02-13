# MONAI 3D CellMap Segmentation Experiment

> **Branch:** `experiment/monai-3d-l40s`  
> **Date:** February 13, 2026  
> **Node:** g181003 (8× NVIDIA L40S 48 GB, 1 TB RAM)  
> **Slurm Job:** 30530860 — reservation `gsgeorge_9034`

## Overview

Custom MONAI training pipeline for the [CellMap Segmentation Challenge](https://cellmapsegmentationchallenge.janelia.org/). Three 3D architectures are trained **simultaneously** on 6 GPUs (2 GPUs each) using DDP, with per-class Tversky loss and Balanced Softmax online class weighting.

The pipeline handles **partial annotations** (not every voxel is labeled for every class) and uses a crop-first data strategy to keep RAM under control with large 3D electron microscopy volumes.

## Models

| Model | Config | Patch Size | Batch/GPU | GPUs | LR | Key Feature |
|-------|--------|-----------|-----------|------|-----|-------------|
| **SegResNet-DS** | `cfg_segresnet.py` | 128³ | 2 | 0,1 | 2e-4 | Deep supervision (4 scales) |
| **FlexibleUNet-ResNet34** | `cfg_flexunet_resnet.py` | 96³ | 4 | 2,3 | 1e-3 | Mixup augmentation (CryoET winner style) |
| **SwinUNETR v2** | `cfg_swinunetr.py` | 96³ | 2 | 4,5 | 1e-4 | Shifted-window self-attention encoder |

All models train for **600 epochs** with validation every **5 epochs**.

## Task

- **Input:** Single-channel 3D EM volumes (FIB-SEM / cryo-ET)
- **Output:** 14-class multi-label segmentation (sigmoid per channel)
- **Classes:** `ecs`, `pm`, `mito_mem`, `mito_lum`, `mito_ribo`, `golgi_mem`, `golgi_lum`, `ves_mem`, `ves_lum`, `endo_mem`, `endo_lum`, `er_mem`, `er_lum`, `nuc`
- **Dataset:** 22 CellMap volumes (train/val split via `datasplit.csv`)

## Loss Function

**Balanced Softmax Tversky Loss** (`losses/partial_annotation.py`)

- Per-channel Tversky with α=0.6 (FP penalty), β=0.4 (FN penalty)
- Online frequency estimation for logit-adjusted class balancing (τ=1.0)
- Partial annotation masking — only backpropagates through annotated channels
- Deep supervision wrapper for SegResNet (multi-scale loss with weights [1.0, 0.5, 0.25, 0.125])

## Architecture Decisions

1. **Crop-first data loading** (`data/ds_cellmap.py`): Pre-crops volumes into ROI-sized chunks at dataset init, then caches only those crops (≤500 MB each). This keeps RAM at ~200-360 GB instead of 900+ GB from loading full volumes.

2. **BFloat16 training**: Native L40S bf16 support for faster throughput without precision loss.

3. **Per-model DDP isolation**: Each model gets its own `torchrun` process with separate `CUDA_VISIBLE_DEVICES` and `MASTER_PORT` to avoid cross-model interference.

4. **CosineAnnealingLR with warmup**: 5% warmup then cosine decay to 0.

## Early Results (Epoch 5)

| Model | Train Loss | Val Dice |
|-------|-----------|----------|
| SegResNet-DS | 0.75 | 0.1162 |
| FlexibleUNet | 0.88 | 0.1254 |
| SwinUNETR v2 | 0.87 | 0.1005 |

Loss trending downward for all models. Dice scores expected to improve significantly over 600 epochs.

## Resource Usage

| Metric | Value |
|--------|-------|
| GPUs allocated | 6 / 8 (2 per model) |
| VRAM per GPU | 13-41 GB / 48 GB |
| GPU utilization | ~100% |
| RAM usage | 200-360 GB / 1007 GB |
| Wall time requested | 11 days |
| Estimated time/epoch | SegResNet ~7.4 min, FlexUNet ~5.5 min, SwinUNETR ~7.1 min |

## File Structure

```
experiments/monai_cellmap/
├── README.md                  # This file
├── IMPLEMENTATION_SPEC.md     # Detailed design document
├── train.py                   # Main training loop (DDP, mixed precision, checkpointing)
├── utils.py                   # Utilities (Config loading, logging, etc.)
├── configs/
│   ├── common_config.py       # Shared base config (all hyperparams)
│   ├── cfg_segresnet.py       # SegResNet-DS config
│   ├── cfg_flexunet_resnet.py # FlexibleUNet + ResNet34 config
│   └── cfg_swinunetr.py       # SwinUNETR v2 config
├── configs_2d/
│   ├── train_2d_unet.py       # 2D UNet config (CSC framework)
│   └── train_2d_swin.py       # 2D Swin Transformer config (CSC framework)
├── data/
│   └── ds_cellmap.py          # CellMap dataset with crop-first strategy
├── losses/
│   └── partial_annotation.py  # Tversky + Balanced Softmax + partial annotation
├── models/
│   └── mdl_cellmap.py         # Model factory (SegResNet, FlexUNet, SwinUNETR)
└── slurm/
    ├── train_reserved.sbatch  # 3D training (3 models × 2 GPUs)
    └── train_2d_reserved.sbatch # 2D training (pending)
```

## Reproduction

```bash
# From the repo root on a Slurm cluster with L40S GPUs:
cd CellMap-Segmentation
sbatch experiments/monai_cellmap/slurm/train_reserved.sbatch
```

Requires: MONAI 1.5.2, PyTorch 2.10.0+cu128, conda env `csc`.

## Monitoring

```bash
# Check job status and latest training output
squeue -j <JOB_ID>
for m in segresnet flexunet swinunetr; do
    echo "=== $m ==="
    tail -10 logs/${m}_<JOB_ID>.out
done

# RAM monitor (embedded in sbatch)
grep MONITOR logs/monai_cellmap_<JOB_ID>.out | tail -5
```

## Next Steps

- [ ] Monitor 3D training through weekend (target: 600 epochs each)
- [ ] Launch 2D models (UNet + SwinTransformer) — coordinate with HPC admin
- [ ] Run inference / prediction on validation set
- [ ] Compare architectures and select best for final submission
