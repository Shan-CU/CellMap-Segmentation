#!/bin/bash
#SBATCH --job-name=swin_2d_ll
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=128g
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/swin_2d_longleaf_%j.out
#SBATCH --error=logs/swin_2d_longleaf_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# SwinUNETR 2D Training - Optimized for UNC Longleaf (A100 40GB)
# ============================================================
# Lessons from Sycamore:
# - attention_dropout=0.1 for transformers
# - Focal Loss + Dice works well
# - 2 GPUs sufficient for 2D
# ============================================================

MODEL="swin"
DIM="2d"

EPOCHS=200
BATCH_SIZE=16                    # Transformers need more VRAM
ITERATIONS_PER_EPOCH=1000
LEARNING_RATE=5e-5
GRADIENT_ACCUMULATION=4          # Effective batch = 16*4*2 = 128
NUM_WORKERS=8

echo "=============================================="
echo "CellMap Model Comparison - SwinUNETR 2D"
echo "=============================================="
echo "Cluster: UNC Longleaf (A100 40GB)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2x A100 40GB"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=============================================="
echo "Model: $MODEL"
echo "Dimension: $DIM"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE (per GPU)"
echo "Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION * 2))"
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

# Add micromamba to PATH
export PATH="$HOME/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export NVIDIA_TF32_OVERRIDE=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

N_GPUS=2
echo "Using $N_GPUS GPUs with DDP"

echo ""
echo "Starting training: $MODEL ($DIM)..."
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
    --pin_memory \
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
