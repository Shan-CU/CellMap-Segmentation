#!/bin/bash
#SBATCH --job-name=p2v4_vitnet_3d
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/p2v4_vitnet_3d_%j.out
#SBATCH --error=runs/ablation/logs/p2v4_vitnet_3d_%j.err

# ============================================================================
# Phase 2 v4: ViT-V-Net 3D on A100 80GB
# ============================================================================
# Architecture: ViT-V-Net (vitnet.py, Chen et al. MIDL 2021) — ~28M params
#   - hidden_size=252, 12 layers, 12 heads, patch_size=8
#   - CNN encoder (16,32,32), decoder (96,48,32,32,16)
#   - RegistrationHead has near-zero init (Normal(0, 1e-5))
#   - img_size auto-set from --input_shape in train.py
#
# Optimizer: RAdam (default), lr=4e-4, wd=0
#   - Paper: Adam, lr=1e-4, wd=0, amsgrad=True
#   - LR: base 1e-4 @ batch=2, scaled to 4e-4 @ batch=8
#   - wd=0 matches paper
#
# Key fix: bias_init=-3.0 (RegistrationHead init ≈ 0 → needs strong prior)
#
# Recipe: dice_bce + EMA(0.999) + fg_mask + weighted_sampler
# Training: 1000 epochs × 300 iters, batch=8, 96³ crops @ 8nm
# Validation: every 5 epochs, per-class Dice (48 classes), 600s time limit
# ============================================================================

set -euo pipefail

# --- Environment ---
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

# --- Memory leak mitigations ---
export CELLMAP_TENSORSTORE_CACHE_BYTES=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=65536
export MALLOC_MMAP_THRESHOLD_=65536
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "Phase 2 v4: ViT-V-Net 3D (RAdam, bias_init=-3.0)"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "============================================"

python -m training.train \
    --experiment_name p2v4_vitnet_3d \
    --run_dir runs/ablation \
    --model vitnet_3d \
    --loss dice_bce \
    --batch_size 8 \
    --input_shape 96 96 96 \
    --input_scale 8 8 8 \
    --epochs 1000 \
    --iterations_per_epoch 300 \
    --learning_rate 4e-4 \
    --max_grad_norm 1.0 \
    --scheduler cosine \
    --warmup_epochs 34 \
    --validation_time_limit 600 \
    --val_every_n_epochs 5 \
    --best_metric val_dice \
    --num_workers 4 \
    --amp \
    --ema --ema_decay 0.999 \
    --bias_init_mode per_class \
    --optimizer radam \
    --weight_decay 0.0 \
    --random_seed 42 \
    --use_foreground_mask \
    --datasplit_path datasplit.csv \
    --filter_by_scale
