#!/bin/bash
#SBATCH --job-name=swin_2d
#SBATCH --partition=blanca-biokem
#SBATCH --account=blanca-biokem
#SBATCH --qos=blanca-biokem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=384G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/swin_2d_%j.out
#SBATCH --error=logs/swin_2d_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@colorado.edu

# ============================================================
# Swin Transformer 2D Training - Optimized for blanca-biokem (2x A100 80GB)
# ============================================================
# Architecture: Swin Transformer 2D (Transformer - High Memory)
# Expected training time: ~36-44 hours for 200 epochs
# Memory usage: ~25-35 GB per GPU (attention is memory intensive)
# ============================================================

MODEL="swin"
DIM="2d"

# Optimized hyperparameters for 2D Swin Transformer on 2x A100 80GB with DDP
# Transformers need smaller batches due to attention memory requirements
# DDP overhead: ~25% extra memory for transformer gradients
EPOCHS=200
BATCH_SIZE=16                    # Per-GPU batch (smaller for transformer + DDP)
ITERATIONS_PER_EPOCH=500         # Shared across all 2D models
LEARNING_RATE=5e-5               # Transformers benefit from lower LR
GRADIENT_ACCUMULATION=8          # Effective batch = 16*8*2GPUs = 256
NUM_WORKERS=8                    # DataLoader workers per GPU

echo "=============================================="
echo "CellMap Model Comparison - Swin Transformer 2D"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2x A100 80GB"
echo "Start time: $(date)"
echo "=============================================="
echo "Model: $MODEL"
echo "Dimension: $DIM"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Iterations/Epoch: $ITERATIONS_PER_EPOCH"
echo "Learning Rate: $LEARNING_RATE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "=============================================="

# Navigate to project root
cd $SLURM_SUBMIT_DIR/../../..
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# Create directories
mkdir -p logs
mkdir -p experiments/model_comparison/checkpoints
mkdir -p experiments/model_comparison/tensorboard
mkdir -p experiments/model_comparison/visualizations
mkdir -p experiments/model_comparison/metrics

# Load Blanca-specific Slurm module
module purge
module load slurm/blanca

# Activate conda environment
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

conda activate cellmap 2>/dev/null || \
    conda activate /projects/$USER/software/anaconda/envs/cellmap 2>/dev/null || \
    source activate cellmap 2>/dev/null || {
        echo "ERROR: Could not activate cellmap environment"
        exit 1
    }

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Environment settings for optimal A100 performance with transformers
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Enable TF32 for faster transformer training on A100
export NVIDIA_TF32_OVERRIDE=1

if [ -n "$SLURM_SCRATCH" ]; then
    export TMPDIR=$SLURM_SCRATCH
    echo "Using local scratch: $SLURM_SCRATCH"
fi

N_GPUS=2
echo "Using $N_GPUS GPUs with DDP"

# Start TensorBoard in background
TB_PORT=$((6006 + SLURM_JOB_ID % 1000))
tensorboard --logdir=experiments/model_comparison/tensorboard --port=$TB_PORT --bind_all &
TB_PID=$!
echo "TensorBoard started on port $TB_PORT"

# Run training with optimized settings
echo ""
echo "Starting training: $MODEL ($DIM)..."
echo "Estimated completion: ~36-44 hours"
echo ""

torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model $MODEL \
    --dim $DIM \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --save_features \
    --vis_every 5

EXIT_CODE=$?

# Cleanup
kill $TB_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "Training Complete"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================="

exit $EXIT_CODE
