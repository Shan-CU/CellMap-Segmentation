# Phase 2 v1 Archive (Feb 27-28, 2026)

These runs are from the first Phase 2 launch attempt. They were killed and
relaunched as "v2" due to critical configuration bugs:

## Bugs in v1

1. **EMA was DISABLED** on all jobs — `--ema` flag was missing from the
   sbatch files (only `--ema_decay 0.999` was passed). Phase 1 showed
   EMA gives 4× better val_loss.

2. **SegResNet used BST loss** instead of dice_bce + deep_supervision.
   BST was a workaround; deep supervision is the proper fix.

3. **ViT 2D crashed** with `cudaErrorInvalidConfiguration` at batch=8.
   Fixed in v2 by reducing to batch=4.

4. **Validation too infrequent** — 2D every 10ep, 3D every 30ep.
   Changed to every 5ep for consistent cross-dimensional comparison.

## Contents

- `runs/arch_*` — Early launch attempt (via launch_phase2.sh, "arch_" naming)
- `runs/canary_*` — Memory leak debugging canary tests (7 iterations)
- `runs/p2_*` — v1 Phase 2 runs (via manual launch, "p2_" naming)
- `runs/tb_phase2` — SLURM-based TensorBoard job
- `logs/` — All SLURM .out/.err files from above

## v1 Results (14h of training, for reference)

| Run | Epochs | Train Loss | Val Dice | Notes |
|-----|--------|------------|----------|-------|
| p2_resnet_2d | 24/100 | 0.385 | **0.271** | Best 2D, 39/48 classes active |
| p2_unet_2d | 19/100 | 0.419 | 0.000 | BatchNorm eval stats broken |
| p2_swin_2d | 24/100 | 0.840 | 0.039 | Known slow Swin convergence |
| p2_vit_2d | 0 | — | — | CUDA crash (batch=8 + AMP) |
| p2_resnet_3d | 7/1000 | 0.478 | — | No val yet |
| p2_unet_3d | 7/1000 | 0.597 | — | No val yet |
| p2_vitnet_3d | 7/1000 | 0.534 | — | No val yet |
| p2_swinunetr_3d | 7/1000 | 0.570 | — | No val yet |
| p2_segresnet_3d | 6.5/1000 | 0.891 | — | BST loss (wrong, should be dice_bce) |
