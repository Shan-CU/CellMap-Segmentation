#!/bin/bash
# ============================================================================
# Phase 2 v4: Optimized MONAI architectures + bias_init for all models
# ============================================================================
# RESEARCH FINDINGS (from MONAI Auto3DSeg official templates):
#
# SegResNetDS was severely misconfigured in v2:
#   - blocks_down was (1,2,2,4) → should be (1,2,2,4,4) [+1 encoder level]
#   - dsdepth was 1 (no deep supervision) → should be 4 (that's what "DS" means!)
#   - norm was batch → should be INSTANCE (standard for small-batch medical seg)
#   - Result: 87M params (vs 20M before), much more capacity + gradient signal
#
# SwinUNETR: enable gradient checkpointing (use_checkpoint=True) to save VRAM
#   and allow larger effective batch on A100 80GB.
#
# Optimizer: MONAI uses AdamW + weight_decay=1e-5 for all architectures.
#   We add --optimizer adamw --weight_decay 1e-5 for MONAI models.
#   CSC models (unet, resnet, vit) keep RAdam + wd=0 for fair comparison
#   (they were designed/tested with different optimizers).
#
# vitnet_3d: bias_init=-3.0 fixes collapse (RegistrationHead init ≈ zero logits)
#
# GPU ALLOCATION (A100 node g180701: 8× A100 SXM4 80GB):
#   debman:    4 GPUs (2 jobs × 2 GPUs each) — can't touch these
#   us (keep): 1 GPU  — p2v3_swin_2d_enhanced (healthy, ~1h in)
#   us (kill): 2 GPUs — p2_swinunetr_3d (0.000 dice @ 2h), p2v3_vitnet_3d_enhanced (I/O bottleneck)
#   free:      1 GPU
#   → After killing: 3 A100 GPUs available for new jobs
#
# L40S partition:
#   us (keep): 4 GPUs — p2_resnet_2d (0.234 dice), p2v3_unet_2d, p2v3_swin_2d, p2v3_vit_2d
#   us (kill): 4 GPUs — p2_segresnet_3d (0.009), p2_vitnet_3d (0.014),
#                        p2_unet_3d (0.021), p2_resnet_3d (0.102)
#   → After killing: 4 L40S GPUs available for new jobs
#
# NEW JOB PLAN (7 jobs: 3 A100 + 4 L40S):
#
#   A100 (3 GPUs):
#     1. segresnet_3d_v4    — full MONAI config: 5-level, DS, instance norm, AdamW
#     2. swinunetr_3d_v4    — gradient checkpointing, AdamW, bias_init
#     3. vitnet_3d_v4       — bias_init, batch=8 (not 16), fixed img_size
#
#   L40S (4 GPUs):
#     4. unet_3d_v4         — bias_init (was 0.021 dice → likely init problem)
#     5. resnet_3d_v4       — bias_init (resnet was 0.102, may improve further)
#     6. segresnet_3d_v4_l40— same as #1 but on L40S (smaller batch if needed)
#     7. swinunetr_3d_v4_l40— same as #2 but on L40S (with use_checkpoint for VRAM)
#
# Usage:
#   cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
#   bash training/slurm/launch_phase2v4_optimized.sh
#   bash training/slurm/launch_phase2v4_optimized.sh --kill-only
#   bash training/slurm/launch_phase2v4_optimized.sh --launch-only
# ============================================================================

set -euo pipefail
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

LOSS="dice_bce"
BIAS_INIT="--bias_init -3.0"

# Parse args
DO_KILL=true
DO_LAUNCH=true
for arg in "$@"; do
    case $arg in
        --kill-only)   DO_LAUNCH=false ;;
        --launch-only) DO_KILL=false ;;
    esac
done

# ============================================================================
# STEP 1: Kill underperforming jobs
# ============================================================================
if $DO_KILL; then
    echo "=== Killing underperforming v2 jobs ==="
    echo ""

    # A100 jobs to kill
    echo "  Cancelling p2_swinunetr_3d (33902468) — 0.000 dice after 2h..."
    scancel 33902468 2>/dev/null || true
    echo "  Cancelling p2v3_vitnet_3d_enhanced (33935984) — I/O bottleneck, 0.000 dice..."
    scancel 33935984 2>/dev/null || true

    # L40S jobs to kill
    echo "  Cancelling p2_segresnet_3d (33568017) — 0.009 dice, suboptimal config..."
    scancel 33568017 2>/dev/null || true
    echo "  Cancelling p2_vitnet_3d (33568021) — 0.014 dice, collapsing..."
    scancel 33568021 2>/dev/null || true
    echo "  Cancelling p2_unet_3d (33568019) — 0.021 dice..."
    scancel 33568019 2>/dev/null || true
    echo "  Cancelling p2_resnet_3d (33568020) — 0.102 dice, relaunch with bias_init..."
    scancel 33568020 2>/dev/null || true

    echo ""
    echo "  Killed 6 jobs. Waiting 10s for GPU release..."
    sleep 10
fi

# ============================================================================
# STEP 2: Launch optimized v4 jobs
# ============================================================================
if $DO_LAUNCH; then

echo ""
echo "============================================"
echo "Phase 2 v4: Optimized architecture comparison"
echo "Recipe: dice_bce + EMA(0.999) + fg_mask + bias_init=-3.0"
echo "MONAI models: AdamW + wd=1e-5 + architecture-specific optimizations"
echo "============================================"
echo ""

# -----------------------------------------------------------------------
# A100 JOBS (3 GPUs available after killing 2 + 1 free)
# -----------------------------------------------------------------------
echo "--- A100 80GB Jobs (3 GPUs) ---"

# 1. SegResNetDS — FULLY OPTIMIZED MONAI CONFIG
#    5-level encoder (1,2,2,4,4), deep supervision (dsdepth=4), instance norm
#    AdamW optimizer with weight_decay=1e-5 (MONAI default)
#    87M params — needs A100 80GB for batch=4 with 96^3 crops
JOB="p2v4_segresnet_3d"
echo "  [A100] ${JOB}: SegResNetDS optimized (87M params, DS, instance norm, AdamW)"
EXPERIMENT_NAME="${JOB}" \
MODEL_NAME="segresnet_3d" \
LOSS_NAME="${LOSS}" \
USE_FG_MASK="true" \
BATCH_SIZE=4 \
EPOCHS=1000 \
ITERS=300 \
EXTRA_ARGS="--ema --ema_decay 0.999 ${BIAS_INIT} --optimizer adamw --weight_decay 1e-5 --deep_supervision" \
sbatch --job-name="${JOB}" training/slurm/phase2_3d_a100.sbatch

# 2. SwinUNETR — with gradient checkpointing + AdamW
#    use_checkpoint=True saves ~30% VRAM, allows batch=8 on A100 80GB
#    AdamW + wd=1e-5 (MONAI default)
JOB="p2v4_swinunetr_3d"
echo "  [A100] ${JOB}: SwinUNETR optimized (gradient checkpointing, AdamW)"
EXPERIMENT_NAME="${JOB}" \
MODEL_NAME="swinunetr_3d" \
LOSS_NAME="${LOSS}" \
USE_FG_MASK="true" \
BATCH_SIZE=8 \
EPOCHS=1000 \
ITERS=300 \
EXTRA_ARGS="--ema --ema_decay 0.999 ${BIAS_INIT} --optimizer adamw --weight_decay 1e-5 --model_kwargs '{\"use_checkpoint\": true}'" \
sbatch --job-name="${JOB}" training/slurm/phase2_3d_a100.sbatch

# 3. ViT-V-Net 3D — bias_init fix + reasonable batch on A100
#    batch=8 (not 16 — that caused I/O bottleneck)
#    img_size auto-set from --input_shape 96 96 96
JOB="p2v4_vitnet_3d"
echo "  [A100] ${JOB}: ViTVNet3D (bias_init, batch=8)"
EXPERIMENT_NAME="${JOB}" \
MODEL_NAME="vitnet_3d" \
LOSS_NAME="${LOSS}" \
USE_FG_MASK="true" \
BATCH_SIZE=8 \
EPOCHS=1000 \
ITERS=300 \
EXTRA_ARGS="--ema --ema_decay 0.999 ${BIAS_INIT} --weight_decay 0.05" \
sbatch --job-name="${JOB}" training/slurm/phase2_3d_a100.sbatch

echo ""

# -----------------------------------------------------------------------
# L40S JOBS (4 GPUs available after killing 4 v2 jobs)
# -----------------------------------------------------------------------
echo "--- L40S 48GB Jobs (4 GPUs) ---"

# 4. UNet 3D — bias_init fix (was 0.021 dice, likely init problem)
JOB="p2v4_unet_3d"
echo "  [L40S] ${JOB}: UNet3D (bias_init)"
EXPERIMENT_NAME="${JOB}" \
MODEL_NAME="unet_3d" \
LOSS_NAME="${LOSS}" \
USE_FG_MASK="true" \
BATCH_SIZE=8 \
EPOCHS=1000 \
ITERS=300 \
EXTRA_ARGS="--ema --ema_decay 0.999 ${BIAS_INIT}" \
sbatch --job-name="${JOB}" training/slurm/phase2_3d_l40s.sbatch

# 5. ResNet 3D — bias_init (was best at 0.102, may improve further)
JOB="p2v4_resnet_3d"
echo "  [L40S] ${JOB}: ResNet3D (bias_init)"
EXPERIMENT_NAME="${JOB}" \
MODEL_NAME="resnet_3d" \
LOSS_NAME="${LOSS}" \
USE_FG_MASK="true" \
BATCH_SIZE=8 \
EPOCHS=1000 \
ITERS=300 \
EXTRA_ARGS="--ema --ema_decay 0.999 ${BIAS_INIT}" \
sbatch --job-name="${JOB}" training/slurm/phase2_3d_l40s.sbatch

# 6. SegResNetDS on L40S — same optimal config, but batch=2 to fit in 48GB
#    87M params with 96^3 crops, batch=2 should be ~20GB
#    This gives us a second SegResNet run for robustness
JOB="p2v4_segresnet_3d_l40"
echo "  [L40S] ${JOB}: SegResNetDS optimized (batch=2, 48GB VRAM limit)"
EXPERIMENT_NAME="${JOB}" \
MODEL_NAME="segresnet_3d" \
LOSS_NAME="${LOSS}" \
USE_FG_MASK="true" \
BATCH_SIZE=2 \
EPOCHS=1000 \
ITERS=300 \
EXTRA_ARGS="--ema --ema_decay 0.999 ${BIAS_INIT} --optimizer adamw --weight_decay 1e-5 --deep_supervision" \
sbatch --job-name="${JOB}" training/slurm/phase2_3d_l40s.sbatch

# 7. SwinUNETR on L40S — gradient checkpointing essential for 48GB
#    With use_checkpoint=True, batch=4 should fit in 48GB
JOB="p2v4_swinunetr_3d_l40"
echo "  [L40S] ${JOB}: SwinUNETR (gradient checkpointing for L40S, batch=4)"
EXPERIMENT_NAME="${JOB}" \
MODEL_NAME="swinunetr_3d" \
LOSS_NAME="${LOSS}" \
USE_FG_MASK="true" \
BATCH_SIZE=4 \
EPOCHS=1000 \
ITERS=300 \
EXTRA_ARGS="--ema --ema_decay 0.999 ${BIAS_INIT} --optimizer adamw --weight_decay 1e-5 --model_kwargs '{\"use_checkpoint\": true}'" \
sbatch --job-name="${JOB}" training/slurm/phase2_3d_l40s.sbatch

echo ""
echo "============================================"
echo "v4 Launch Summary"
echo "============================================"
echo ""
echo "  KILLED (6 jobs):"
echo "    A100: p2_swinunetr_3d, p2v3_vitnet_3d_enhanced"
echo "    L40S: p2_segresnet_3d, p2_vitnet_3d, p2_unet_3d, p2_resnet_3d"
echo ""
echo "  KEPT RUNNING (4 L40S + 1 A100 = 5 jobs):"
echo "    A100: p2v3_swin_2d_enhanced (batch=16, ~1h in)"
echo "    L40S: p2_resnet_2d, p2v3_unet_2d, p2v3_swin_2d, p2v3_vit_2d"
echo ""
echo "  NEW v4 JOBS (3 A100 + 4 L40S = 7 jobs):"
echo "    A100: p2v4_segresnet_3d    (87M, DS, instance norm, AdamW, batch=4)"
echo "    A100: p2v4_swinunetr_3d    (62M, grad ckpt, AdamW, batch=8)"
echo "    A100: p2v4_vitnet_3d       (28M, bias_init, batch=8)"
echo "    L40S: p2v4_unet_3d         (bias_init, batch=8)"
echo "    L40S: p2v4_resnet_3d       (bias_init, batch=8)"
echo "    L40S: p2v4_segresnet_3d_l40 (87M, DS, AdamW, batch=2)"
echo "    L40S: p2v4_swinunetr_3d_l40 (62M, grad ckpt, AdamW, batch=4)"
echo ""
echo "  TOTAL GPU USAGE: 4 A100 (ours) + 8 L40S (ours) = 12 GPUs"
echo ""
echo "  Monitor: squeue -u gsgeorge"
echo "============================================"

fi
