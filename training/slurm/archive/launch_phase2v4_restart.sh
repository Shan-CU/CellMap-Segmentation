#!/bin/bash
# ============================================================================
# Phase 2 v4: FRESH RESTART — All architectures optimally configured
# ============================================================================
# RUN THIS SCRIPT ON LONGLEAF (not through SSH):
#   ssh longleaf.unc.edu
#   cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
#   bash training/slurm/launch_phase2v4_restart.sh
#
# RESEARCH SUMMARY — OPTIMAL TRAINING CONFIGS PER ARCHITECTURE:
#
# ═══════════════════════════════════════════════════════════════════════════
# CSC IN-HOUSE ARCHITECTURES:
# ═══════════════════════════════════════════════════════════════════════════
#
# ResNet 2D/3D (~7.8M params):
#   - Already uses InstanceNorm by default (Johnson-style ResNet generator)
#   - 6 ResBlocks, ngf=64, 2 downsampling stages
#   - Was BEST 2D (0.234 dice) and BEST 3D (0.102 dice) in v2
#   - Fix: bias_init=-3.0 (RetinaNet-style focal prior, prevents BCE collapse)
#   - Optimizer: RAdam, wd=0 (simple convnet, worked well in v2)
#
# UNet 2D/3D (~31M params):
#   - v2 used BatchNorm → SWITCH to InstanceNorm (standard for medical seg;
#     used by SegResNetDS, nnU-Net, V-Net — stable with batch sizes 1-8)
#   - Add dropout=0.1 (model supports it; light regularization)
#   - Fix: bias_init=-3.0
#   - Optimizer: RAdam, wd=0
#
# SwinTransformer V2 2D (~36M params):
#   - Swin V2 paper (Liu et al. CVPR 2022): AdamW, lr=1e-4, wd=0.05,
#     stochastic_depth=0.2, cosine schedule with warmup
#   - Our config: embed_dim=96, depths=[2,2,6,2], window_size=8
#   - Fix: bias_init=-3.0 + AdamW + weight_decay=0.05
#   - v2/v3 all collapsed to 0.000 dice → bias_init is critical for this
#
# ViTVNet 2D (~105M params):
#   - ViT-V-Net paper (Chen et al. MIDL 2021): Adam, lr=1e-4, wd=0
#   - Large model (105M) needs transformer-style regularization
#   - Fix: bias_init=-3.0 + AdamW + wd=0.01 (DeiT-style ViT training)
#
# ViTVNet 3D (~28M params):
#   - Paper: Adam, lr=1e-4, wd=0, amsgrad=True
#   - RegistrationHead has near-zero init → needs bias_init=-3.0
#   - img_size auto-set from --input_shape 96 96 96 in train.py
#   - Optimizer: RAdam, wd=0 (small model, match paper's wd=0)
#
# ═══════════════════════════════════════════════════════════════════════════
# MONAI ARCHITECTURES (from MONAI Auto3DSeg official templates):
# ═══════════════════════════════════════════════════════════════════════════
#
# SegResNetDS 3D (~87M params) — FULLY RECONFIGURED from v2:
#   v2 (WRONG):  blocks_down=(1,2,2,4), dsdepth=1, BatchNorm, 20M params
#   v4 (CORRECT): blocks_down=(1,2,2,4,4), dsdepth=4, InstanceNorm, 87M params
#   - Deep supervision: 4 outputs with exponential weights [1.0, 0.5, 0.25, ...]
#   - Optimizer: AdamW, base_lr=2e-4 (at batch=2, linear scaled), wd=1e-5
#   - Source: research-contributions/auto3dseg/algorithm_templates/segresnet/
#
# SwinUNETR 3D (~62M params):
#   - MONAI template: AdamW, lr=4e-4, wd=1e-5
#   - use_checkpoint=True for gradient checkpointing (saves ~30% VRAM)
#   - Fix: bias_init=-3.0 (v2 had 0.000 dice after 2+ hours)
#
# ═══════════════════════════════════════════════════════════════════════════
# COMMON SETTINGS (Phase 1 winning recipe):
# ═══════════════════════════════════════════════════════════════════════════
#   Loss: dice_bce | EMA: 0.999 | fg_mask: ON | weighted_sampler: ON
#   AMP: ON | bias_init: -3.0 | val_every_n_epochs: 5
#   validation_time_limit: 600 | best_metric: val_dice
#
# GPU PLAN: 3 A100 + 6 L40S = 9 jobs total
# ============================================================================

set -euo pipefail
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

# Parse args
DO_KILL=true
DO_ARCHIVE=true
DO_LAUNCH=true
for arg in "$@"; do
    case $arg in
        --kill-only)    DO_ARCHIVE=false; DO_LAUNCH=false ;;
        --archive-only) DO_KILL=false; DO_LAUNCH=false ;;
        --launch-only)  DO_KILL=false; DO_ARCHIVE=false ;;
    esac
done

# ============================================================================
# STEP 1: Kill ALL running Phase 2 jobs
# ============================================================================
if $DO_KILL; then
    echo "=== Killing ALL Phase 2 v2/v3 jobs ==="
    echo ""

    JOB_LIST=$(squeue -u gsgeorge -h -o '%i %j' 2>/dev/null | grep -E 'p2' || true)

    if [ -n "$JOB_LIST" ]; then
        echo "$JOB_LIST" | while IFS=' ' read -r jid jname; do
            echo "  Cancelling $jid ($jname)..."
            scancel "$jid" 2>/dev/null || true
        done
        N_KILLED=$(echo "$JOB_LIST" | wc -l)
        echo ""
        echo "  Killed $N_KILLED jobs. Waiting 15s for GPU release..."
        sleep 15
    else
        echo "  No Phase 2 jobs found."
    fi
    echo ""
fi

# ============================================================================
# STEP 2: Archive old run directories
# ============================================================================
if $DO_ARCHIVE; then
    echo "=== Archiving Phase 2 v2/v3 runs ==="
    ARCHIVE_DIR="runs/ablation/phase2_v2v3_archive"
    mkdir -p "$ARCHIVE_DIR"

    for run_dir in runs/ablation/p2_* runs/ablation/p2v3_*; do
        if [ -d "$run_dir" ]; then
            run_name=$(basename "$run_dir")
            echo "  Moving $run_name → $ARCHIVE_DIR/"
            mv "$run_dir" "$ARCHIVE_DIR/" 2>/dev/null || true
        fi
    done

    echo ""
    echo "  Archived to: $ARCHIVE_DIR/"
    ls -1 "$ARCHIVE_DIR/" 2>/dev/null || echo "  (empty)"
    echo ""
fi

# ============================================================================
# STEP 3: Launch ALL 9 optimized v4 jobs
# ============================================================================
if $DO_LAUNCH; then

echo ""
echo "============================================"
echo "Phase 2 v4: FRESH RESTART — All 9 architectures"
echo "Recipe: dice_bce + EMA(0.999) + fg_mask + bias_init=-3.0"
echo "============================================"
echo ""

LOSS="dice_bce"

# -----------------------------------------------------------------------
# 2D MODELS on L40S (4 jobs)
# 256×256 patches, batch=8, 100 epochs × 1000 iters, val every 5 ep
# -----------------------------------------------------------------------
echo "--- 2D Models on L40S (4 jobs) ---"

# 1. ResNet 2D — Our best performer, RAdam, bias_init
echo "  [L40S] p2v4_resnet_2d: RAdam, lr=1e-4, wd=0, bias_init=-3.0"
EXPERIMENT_NAME=p2v4_resnet_2d \
MODEL_NAME=resnet_2d \
LOSS_NAME=${LOSS} \
BATCH_SIZE=8 \
EPOCHS=100 \
ITERS=1000 \
EXTRA_ARGS="--bias_init -3.0" \
sbatch -J p2v4_resnet_2d training/slurm/phase2_2d_l40s.sbatch

# 2. UNet 2D — InstanceNorm + dropout, bias_init
echo "  [L40S] p2v4_unet_2d: RAdam, lr=1e-4, wd=0, instancenorm, dropout=0.1"
EXPERIMENT_NAME=p2v4_unet_2d \
MODEL_NAME=unet_2d \
LOSS_NAME=${LOSS} \
BATCH_SIZE=8 \
EPOCHS=100 \
ITERS=1000 \
EXTRA_ARGS='--bias_init -3.0 --model_kwargs '"'"'{"use_instancenorm": true, "dropout": 0.1}'"'"'' \
sbatch -J p2v4_unet_2d training/slurm/phase2_2d_l40s.sbatch

# 3. SwinTransformer 2D — AdamW + wd=0.05, bias_init (Swin V2 paper config)
echo "  [L40S] p2v4_swin_2d: AdamW, lr=1e-4, wd=0.05, bias_init=-3.0"
EXPERIMENT_NAME=p2v4_swin_2d \
MODEL_NAME=swin_2d \
LOSS_NAME=${LOSS} \
BATCH_SIZE=8 \
EPOCHS=100 \
ITERS=1000 \
EXTRA_ARGS="--bias_init -3.0 --optimizer adamw --weight_decay 0.05" \
sbatch -J p2v4_swin_2d training/slurm/phase2_2d_l40s.sbatch

# 4. ViTVNet 2D — AdamW + wd=0.01 (DeiT-style), bias_init
echo "  [L40S] p2v4_vit_2d: AdamW, lr=1e-4, wd=0.01, bias_init=-3.0"
EXPERIMENT_NAME=p2v4_vit_2d \
MODEL_NAME=vit_2d \
LOSS_NAME=${LOSS} \
BATCH_SIZE=8 \
EPOCHS=100 \
ITERS=1000 \
EXTRA_ARGS="--bias_init -3.0 --optimizer adamw --weight_decay 0.01" \
sbatch -J p2v4_vit_2d training/slurm/phase2_2d_l40s.sbatch

echo ""

# -----------------------------------------------------------------------
# 3D CNN MODELS on L40S (2 jobs: resnet_3d, unet_3d — fit in 48GB)
# 96³ crops, batch=8, 1000 epochs × 300 iters, val every 5 ep
# LR scaled linearly: base_lr=1e-4 @ batch=2 → 4e-4 @ batch=8
# -----------------------------------------------------------------------
echo "--- 3D CNN Models on L40S (2 jobs) ---"

# 5. ResNet 3D — bias_init, best 3D model in v2 (0.102 dice)
echo "  [L40S] p2v4_resnet_3d: RAdam, lr=4e-4 (scaled), wd=0, bias_init=-3.0"
EXPERIMENT_NAME=p2v4_resnet_3d \
MODEL_NAME=resnet_3d \
LOSS_NAME=${LOSS} \
BATCH_SIZE=8 \
EPOCHS=1000 \
ITERS=300 \
EXTRA_ARGS="--bias_init -3.0" \
sbatch -J p2v4_resnet_3d training/slurm/phase2_3d_l40s.sbatch

# 6. UNet 3D — InstanceNorm + dropout, bias_init
echo "  [L40S] p2v4_unet_3d: RAdam, lr=4e-4 (scaled), wd=0, instancenorm, dropout=0.1"
EXPERIMENT_NAME=p2v4_unet_3d \
MODEL_NAME=unet_3d \
LOSS_NAME=${LOSS} \
BATCH_SIZE=8 \
EPOCHS=1000 \
ITERS=300 \
EXTRA_ARGS='--bias_init -3.0 --model_kwargs '"'"'{"use_instancenorm": true, "dropout": 0.1}'"'"'' \
sbatch -J p2v4_unet_3d training/slurm/phase2_3d_l40s.sbatch

echo ""

# -----------------------------------------------------------------------
# 3D MODELS on A100 80GB (3 jobs: MONAI architectures + ViTVNet)
# 96³ crops, val every 5 ep
# LR scaled linearly: base_lr=1e-4 @ batch=2
# -----------------------------------------------------------------------
echo "--- 3D MONAI + ViT Models on A100 80GB (3 jobs) ---"

# 7. SegResNetDS — FULL MONAI config: 87M params, DS, AdamW
#    batch=4: LR = 1e-4 * 4/2 = 2e-4 (matches MONAI official lr=2e-4)
echo "  [A100] p2v4_segresnet_3d: AdamW, lr=2e-4 (scaled), wd=1e-5, deep_supervision"
EXPERIMENT_NAME=p2v4_segresnet_3d \
MODEL_NAME=segresnet_3d \
LOSS_NAME=${LOSS} \
BATCH_SIZE=4 \
EPOCHS=1000 \
ITERS=300 \
EXTRA_ARGS="--bias_init -3.0 --optimizer adamw --weight_decay 1e-5 --deep_supervision" \
sbatch -J p2v4_segresnet_3d training/slurm/phase2_3d_a100.sbatch

# 8. SwinUNETR — gradient checkpointing, AdamW
#    batch=8: LR = 1e-4 * 8/2 = 4e-4 (matches MONAI SwinUNETR template)
echo "  [A100] p2v4_swinunetr_3d: AdamW, lr=4e-4 (scaled), wd=1e-5, use_checkpoint"
EXPERIMENT_NAME=p2v4_swinunetr_3d \
MODEL_NAME=swinunetr_3d \
LOSS_NAME=${LOSS} \
BATCH_SIZE=8 \
EPOCHS=1000 \
ITERS=300 \
EXTRA_ARGS='--bias_init -3.0 --optimizer adamw --weight_decay 1e-5 --model_kwargs '"'"'{"use_checkpoint": true}'"'"'' \
sbatch -J p2v4_swinunetr_3d training/slurm/phase2_3d_a100.sbatch

# 9. ViTVNet 3D — bias_init critical (RegistrationHead init ≈ 0)
#    batch=8: LR = 1e-4 * 8/2 = 4e-4, RAdam, wd=0 (paper: Adam wd=0)
echo "  [A100] p2v4_vitnet_3d: RAdam, lr=4e-4 (scaled), wd=0, bias_init=-3.0"
EXPERIMENT_NAME=p2v4_vitnet_3d \
MODEL_NAME=vitnet_3d \
LOSS_NAME=${LOSS} \
BATCH_SIZE=8 \
EPOCHS=1000 \
ITERS=300 \
EXTRA_ARGS="--bias_init -3.0" \
sbatch -J p2v4_vitnet_3d training/slurm/phase2_3d_a100.sbatch

echo ""
echo "============================================"
echo "Launched 9 jobs: 4 L40S (2D) + 2 L40S (3D) + 3 A100 (3D)"
echo "============================================"
echo ""
echo "Monitor:"
echo "  squeue -u gsgeorge"
echo "  tensorboard --logdir runs/ablation --bind_all"
echo ""
echo "Job summary:"
echo "  p2v4_resnet_2d    [L40S] ResNet 2D    — RAdam lr=1e-4 wd=0"
echo "  p2v4_unet_2d      [L40S] UNet 2D      — RAdam lr=1e-4 wd=0 + InstanceNorm + dropout"
echo "  p2v4_swin_2d      [L40S] Swin V2 2D   — AdamW lr=1e-4 wd=0.05"
echo "  p2v4_vit_2d       [L40S] ViTVNet 2D   — AdamW lr=1e-4 wd=0.01"
echo "  p2v4_resnet_3d    [L40S] ResNet 3D    — RAdam lr=4e-4 wd=0"
echo "  p2v4_unet_3d      [L40S] UNet 3D      — RAdam lr=4e-4 wd=0 + InstanceNorm + dropout"
echo "  p2v4_segresnet_3d [A100] SegResNetDS  — AdamW lr=2e-4 wd=1e-5 + DeepSup"
echo "  p2v4_swinunetr_3d [A100] SwinUNETR    — AdamW lr=4e-4 wd=1e-5 + GradCheckpoint"
echo "  p2v4_vitnet_3d    [A100] ViTVNet 3D   — RAdam lr=4e-4 wd=0"
echo "  ALL: bias_init=-3.0 + EMA(0.999) + dice_bce + fg_mask + val@5ep"
echo ""

fi  # end DO_LAUNCH
