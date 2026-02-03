#!/bin/bash
#SBATCH --job-name=unet_3d_ll
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:3
#SBATCH --mem=256g
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/unet_3d_longleaf_%j.out
#SBATCH --error=logs/unet_3d_longleaf_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# UNet 3D Training - Optimized for UNC Longleaf (A100 40GB)
# ============================================================
# CRITICAL lessons from Sycamore 3D OOM issues:
# - num_workers=0 REQUIRED (prevents memory leak in 3D)
# - Smaller batch size for volumetric data
# - InstanceNorm prevents collapse
# - 3 GPUs for parallel 3D processing
# ============================================================

MODEL="unet"
DIM="3d"

EPOCHS=200
BATCH_SIZE=2                     # Small for 3D volumes on 40GB
ITERATIONS_PER_EPOCH=500         # 3D iterations take longer
LEARNING_RATE=5e-5
GRADIENT_ACCUMULATION=8          # Effective batch = 2*8*3 = 48
NUM_WORKERS=0                    # CRITICAL: Prevents 3D memory leak!

echo "=============================================="
echo "CellMap Model Comparison - UNet 3D"
echo "=============================================="
echo "Cluster: UNC Longleaf (A100 40GB)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 3x A100 40GB"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=============================================="
echo "Model: $MODEL"
echo "Dimension: $DIM"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE (per GPU)"
echo "Iterations/Epoch: $ITERATIONS_PER_EPOCH"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION * 3))"
echo "Num Workers: $NUM_WORKERS (0 to prevent 3D memory leak)"
echo "=============================================="

cd $SLURM_SUBMIT_DIR/../../..
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

mkdir -p logs
mkdir -p experiments/model_comparison/checkpoints
mkdir -p experiments/model_comparison/tensorboard
mkdir -p experiments/model_comparison/visualizations
mkdir -p experiments/model_comparison/slurm/longleaf/logs

module purge
module load cuda/12.2

export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Memory optimizations critical for 3D
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export NVIDIA_TF32_OVERRIDE=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

N_GPUS=3
echo "Using $N_GPUS GPUs with DDP"

echo ""
echo "Starting training: $MODEL ($DIM)..."
echo "NOTE: num_workers=0 to prevent 3D DataLoader memory leak"
echo ""

torchrun --standalone --nproc_per_node=$N_GPUS \
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
