#!/bin/bash
#SBATCH --job-name=swin_3d_l40
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=512g
#SBATCH --time=11-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/model_comparison/slurm/longleaf/logs/swin_3d_l40s_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/model_comparison/slurm/longleaf/logs/swin_3d_l40s_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# SwinUNETR 3D Training - UNC Longleaf L40S (48GB)
# ============================================================
# Test job: Most memory-intensive model on L40S partition
# - 4x L40S 48GB = 192GB total GPU memory
# - 11-day wall time (vs 6-day on A100)
# - num_workers=0 REQUIRED for 3D
# - attention_dropout=0.1 for regularization
# ============================================================

MODEL="swin"
DIM="3d"

EPOCHS=200
BATCH_SIZE=2                     # L40S 48GB has headroom for batch=2
ITERATIONS_PER_EPOCH=500
LEARNING_RATE=5e-5
GRADIENT_ACCUMULATION=6          # Effective batch = 2*6*4GPUs = 48
NUM_WORKERS=0                    # CRITICAL for 3D

echo "=============================================="
echo "CellMap Model Comparison - SwinUNETR 3D"
echo "=============================================="
echo "Cluster: UNC Longleaf (L40S 48GB)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4x L40S 48GB"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=============================================="
echo "Model: $MODEL"
echo "Dimension: $DIM"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE (per GPU) - minimal for 3D transformer"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION * 4))"
echo "Num Workers: $NUM_WORKERS (0 to prevent memory leak)"
echo "=============================================="

PROJECT_ROOT=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
cd $PROJECT_ROOT
echo "Project root: $PROJECT_ROOT"

mkdir -p experiments/model_comparison/checkpoints
mkdir -p experiments/model_comparison/tensorboard
mkdir -p experiments/model_comparison/visualizations
mkdir -p experiments/model_comparison/slurm/longleaf/logs

module purge
module load cuda/12.2

# Use absolute paths for non-interactive batch environment
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'

echo "Python: $($MAMBA_EXE run -n csc which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Memory optimization for 3D transformer
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.7"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export NVIDIA_TF32_OVERRIDE=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

N_GPUS=4
echo "Using $N_GPUS GPUs with DDP"

echo ""
echo "Starting training: $MODEL ($DIM)..."
echo "WARNING: This is the most memory-intensive model"
echo ""

$MAMBA_EXE run -n csc torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model $MODEL \
    --dim $DIM \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --iterations_per_epoch $ITERATIONS_PER_EPOCH \
    --lr $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --amp \
    --compile \
    --save_features \
    --vis_every 10

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training Complete - Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================="

exit $EXIT_CODE
