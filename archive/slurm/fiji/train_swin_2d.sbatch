#!/bin/bash
#SBATCH --job-name=swin_2d_fiji
#SBATCH --partition=nvidia-a100
#SBATCH --account=reguser
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/swin_2d_fiji_%j.out
#SBATCH --error=logs/swin_2d_fiji_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=george.stephenson@colorado.edu

# ============================================================
# Swin Transformer 2D Training - Optimized for Fiji (4x A100 80GB)
# ============================================================
# Architecture: Swin Transformer 2D (Windowed Attention)
# Hardware: Fiji nvidia-a100 partition
# GPUs: 2x A100 80GB
# Expected training time: ~28-36 hours for 200 epochs
# Memory usage: ~20-30 GB per GPU
# All Sycamore optimizations applied
# ============================================================

MODEL="swin"
DIM="2d"
EPOCHS=200
BATCH_SIZE=24
ITERATIONS_PER_EPOCH=1000
LEARNING_RATE=3e-5
GRADIENT_ACCUMULATION=8
NUM_WORKERS=0

echo "=============================================="
echo "CellMap Model Comparison - Swin 2D (Fiji)"
echo "=============================================="
echo "Cluster: Fiji (A100)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 1x A100 80GB"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=============================================="
echo "Model: $MODEL | Dimension: $DIM"
echo "Epochs: $EPOCHS | Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Effective Batch: $((BATCH_SIZE * GRADIENT_ACCUMULATION * 1))"
echo "=============================================="

PROJECT_ROOT=/scratch/Users/gest9386/CellMap-Segmentation
cd $PROJECT_ROOT
echo "Project root: $PROJECT_ROOT"

mkdir -p logs experiments/model_comparison/{checkpoints,tensorboard,visualizations,metrics}

source ~/miniconda3/bin/activate cellmap

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# A100 Optimizations (OMP threads limited to prevent thread exhaustion)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.9"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_MODULE_LOADING=LAZY
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

N_GPUS=1
echo "Using $N_GPUS GPUs with DDP"

TB_PORT=$((6006 + SLURM_JOB_ID % 1000))
tensorboard --logdir=experiments/model_comparison/tensorboard --port=$TB_PORT --bind_all &
TB_PID=$!
echo "TensorBoard: http://$SLURM_NODELIST:$TB_PORT"

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
    --amp \
    --no_compile \
    --save_features \
    --vis_every 10

EXIT_CODE=$?
kill $TB_PID 2>/dev/null || true
# Backup checkpoints and logs to backed-up home storage
echo ""
echo "Backing up results to home storage (backed up nightly)..."
HOME_BACKUP=/Users/gest9386/CellMap-Segmentation-backups
mkdir -p $HOME_BACKUP/checkpoints $HOME_BACKUP/tensorboard $HOME_BACKUP/metrics
rsync -avz experiments/model_comparison/checkpoints/ $HOME_BACKUP/checkpoints/ 2>&1 | tail -5
rsync -avz experiments/model_comparison/tensorboard/ $HOME_BACKUP/tensorboard/ 2>&1 | tail -5
rsync -avz experiments/model_comparison/metrics/ $HOME_BACKUP/metrics/ 2>&1 | tail -5
echo "Backup complete: $HOME_BACKUP"
echo ""
echo "=============================================="
echo "Training Complete | Exit: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================="

exit $EXIT_CODE
