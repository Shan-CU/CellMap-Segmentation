# Phase 2 v5: Optimized Training Configurations

## Summary of Changes from v4 → v5

All v5 configs are based on comprehensive research of:
- **CSC official source code** (training pipeline defaults)
- **Lauenburg fork** (CellMap competition winner's approach)
- **eminorhan fork** (competition participant)
- **MONAI Auto3DSeg** (SegResNetDS paper defaults)
- **Original architecture papers** (Swin V2, UNet, etc.)

### Universal Changes (all models)

| Parameter | v4 | v5 | Source |
|---|---|---|---|
| Optimizer | RAdam wd=0 | AdamW wd=1e-4 | Lauenburg, all baselines |
| eta_min | 0 (LR→0) | 1e-6 | Lauenburg CosineAnnealing |
| Intensity aug | none | `--intensity_aug` | Lauenburg: GaussBlur p=0.2 |
| Loss bce_weight | 0.5 | 0.4 | Lauenburg: 60% dice / 40% BCE |
| Loss smooth | 1e-6 | 1e-3 | Lauenburg |
| Grad clip | 1.0 | 0.5 | Lauenburg |
| Val frequency | every 5 | every 10 | Longer runs, save val time |

### Per-Model Changes

#### ResNet 2D (`resnet_2d.sbatch`) — L40
- **Epochs**: 100 → 300 (300K total steps)
- **Grad accum**: 1→2 (eff batch 16)
- **Warmup**: 10→15 epochs (5%)

#### UNet 2D (`unet_2d.sbatch`) — L40
- Same as ResNet 2D
- Added `model_kwargs: use_instancenorm=true, dropout=0.1`

#### UNet 3D (`unet_3d.sbatch`) — 2× A100 DDP
- **LR**: 4e-4 → 1e-4 (CSC default)
- **Iters/epoch**: 300 → 500 (500K total steps)
- **Warmup**: 34 → 50 epochs (5%)
- ROI 128³, DDP batch=4/GPU (eff 8), port 29501

#### ResNet 3D (`resnet_3d.sbatch`) — 1× A100
- **LR**: 4e-4 → 1e-4 (CSC default)
- **Iters/epoch**: 300 → 500 (500K total steps)
- **Grad accum**: 1→2 (eff batch 16)
- **Warmup**: 34 → 50 epochs (5%)
- ROI 128³, single GPU batch=8

#### SegResNetDS 3D (`segresnet_3d.sbatch`) — 2× A100 DDP
- v4 was already well-configured (matched Auto3DSeg)
- Added: eta_min=1e-6, loss tuning, iters 300→500
- Kept: AdamW lr=2e-4 wd=1e-5, deep_supervision, ROI 224³

### Retired Models (not included in v5)
- **Swin 2D**: Dice declining at 0.012 after 100K steps. Without ImageNet-pretrained encoder weights, Swin V2 from scratch cannot compete. Would need custom weight loading code.
- **ViT 2D**: Killed in v4 (0.00 dice, collapsed).
- **ViTVNet 3D**: Dead (0.0002 dice).
- **SwinUNETR 3D**: Not relaunched (would need pretrained weights).

## GPU Allocation

| Model | Partition | GPUs | Memory |
|---|---|---|---|
| ResNet 2D | l40-gpu | 1× L40 | 128G |
| UNet 2D | l40-gpu | 1× L40 | 128G |
| SegResNetDS 3D | a100-multi-gpu | 2× A100 | 200G |
| UNet 3D | a100-multi-gpu | 2× A100 | 200G |
| ResNet 3D | a100-multi-gpu | 1× A100 | 128G |
| **Total** | | **2 L40 + 5 A100** | |

## Launch

```bash
bash training/slurm/phase2v5/launch_all.sh
```
