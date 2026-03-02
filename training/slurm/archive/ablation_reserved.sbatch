#!/bin/bash
#SBATCH --job-name=ablation_reserved
#SBATCH --reservation=gsgeorge_9034
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --output=runs/ablation/logs/%x_%j.out
#SBATCH --error=runs/ablation/logs/%x_%j.err

# ============================================================================
# Run MULTIPLE ablation experiments sequentially on the reserved node (8× L40S)
#
# This script runs experiments on separate GPUs in parallel using
# CUDA_VISIBLE_DEVICES. 8 experiments run at once, then the next batch.
#
# Usage:
#   sbatch training/slurm/ablation_reserved.sbatch
# ============================================================================

set -euo pipefail

echo "============================================"
echo "Reserved node ablation runner"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "============================================"

eval "$(micromamba shell hook -s bash)"
micromamba activate csc
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

# Function to run an experiment on a specific GPU
run_on_gpu() {
    local gpu_id="$1"
    local experiment_name="$2"
    local model="$3"
    local loss="$4"
    local fg_mask="${5:-true}"
    local extra_args="${6:-}"

    echo "[GPU ${gpu_id}] Starting: ${experiment_name}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python -m training.train \
        --experiment_name "${experiment_name}" \
        --run_dir runs/ablation \
        --model "${model}" \
        --loss "${loss}" \
        --batch_size 8 \
        --input_shape 1 256 256 \
        --input_scale 8 8 8 \
        --epochs 50 \
        --iterations_per_epoch 500 \
        --learning_rate 1e-4 \
        --max_grad_norm 1.0 \
        --scheduler cosine \
        --warmup_epochs 5 \
        --validation_time_limit 120 \
        --val_every_n_epochs 5 \
        --amp \
        $([ "${fg_mask}" = "false" ] && echo "--no_foreground_mask") \
        ${extra_args} \
        2>&1 | tee "runs/ablation/logs/${experiment_name}.log" &

    echo "[GPU ${gpu_id}] PID: $!"
}

# ============================================================================
# BATCH 1: Loss function sweep (2D) — 5 experiments on GPUs 0-4
# + Tversky α/β sweep — 3 experiments on GPUs 5-7
# ============================================================================
echo ""
echo "=== BATCH 1: Loss sweep + Tversky sweep ==="
run_on_gpu 0 "loss_2d_bce"                       resnet_2d bce
run_on_gpu 1 "loss_2d_focal"                      resnet_2d focal
run_on_gpu 2 "loss_2d_dice_bce"                   resnet_2d dice_bce
run_on_gpu 3 "loss_2d_tversky"                    resnet_2d tversky
run_on_gpu 4 "loss_2d_balanced_softmax_tversky"   resnet_2d balanced_softmax_tversky
run_on_gpu 5 "tversky_2d_balanced"                resnet_2d tversky_balanced
run_on_gpu 6 "tversky_2d_precision_07_03"         resnet_2d tversky_precision
run_on_gpu 7 "tversky_2d_recall"                  resnet_2d tversky_recall

echo "Waiting for Batch 1..."
wait
echo "Batch 1 complete!"

# ============================================================================
# BATCH 2: Weighting sweep (4) + Masking sweep first 4
# ============================================================================
echo ""
echo "=== BATCH 2: Weighting sweep + Masking sweep (part 1) ==="
run_on_gpu 0 "tau_2d_0"                          resnet_2d bst_tau0
run_on_gpu 1 "tau_2d_05"                          resnet_2d bst_tau05
run_on_gpu 2 "tau_2d_10"                          resnet_2d balanced_softmax_tversky
run_on_gpu 3 "tau_2d_15"                          resnet_2d bst_tau15
run_on_gpu 4 "mask_2d_none"                       resnet_2d bst_no_bbox     false
run_on_gpu 5 "mask_2d_fg_only"                    resnet_2d bst_no_bbox     true
run_on_gpu 6 "mask_2d_bbox_only"                  resnet_2d balanced_softmax_tversky  false
run_on_gpu 7 "mask_2d_bbox_fg"                    resnet_2d balanced_softmax_tversky  true

echo "Waiting for Batch 2..."
wait
echo "Batch 2 complete!"

# ============================================================================
# BATCH 3: Remaining masking experiments (3) + start 3D loss sweep (5)
# ============================================================================
echo ""
echo "=== BATCH 3: Masking sweep (part 2) + 3D loss sweep ==="
run_on_gpu 0 "mask_2d_bbox_loose"                resnet_2d bst_bbox_loose  true
run_on_gpu 1 "mask_2d_masksup03"                 resnet_2d bst_masksup03   true
run_on_gpu 2 "mask_2d_masksup03_no_bbox"         resnet_2d bst_masksup03_no_bbox true

# 3D experiments (batch_size=2, 128³ input, fewer iters)
# Override via extra_args
run_on_gpu 3 "loss_3d_bce"                       segresnet_3d bce          true "--batch_size 2 --input_shape 128 128 128 --iterations_per_epoch 250 --validation_time_limit 180"
run_on_gpu 4 "loss_3d_focal"                      segresnet_3d focal        true "--batch_size 2 --input_shape 128 128 128 --iterations_per_epoch 250 --validation_time_limit 180"
run_on_gpu 5 "loss_3d_dice_bce"                   segresnet_3d dice_bce     true "--batch_size 2 --input_shape 128 128 128 --iterations_per_epoch 250 --validation_time_limit 180"
run_on_gpu 6 "loss_3d_tversky"                    segresnet_3d tversky      true "--batch_size 2 --input_shape 128 128 128 --iterations_per_epoch 250 --validation_time_limit 180"
run_on_gpu 7 "loss_3d_balanced_softmax_tversky"   segresnet_3d balanced_softmax_tversky true "--batch_size 2 --input_shape 128 128 128 --iterations_per_epoch 250 --validation_time_limit 180"

echo "Waiting for Batch 3..."
wait
echo "Batch 3 complete!"

echo ""
echo "============================================"
echo "All ablation experiments complete!"
echo "Run analysis: python -m training.analyze_ablations --run_dir runs/ablation --plot"
echo "============================================"
