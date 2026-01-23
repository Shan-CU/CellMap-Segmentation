#!/bin/bash
#SBATCH --job-name=swin_3d
#SBATCH --partition=blanca-biokem
#SBATCH --account=blanca-biokem
#SBATCH --qos=blanca-biokem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/swin_3d_%j.out
#SBATCH --error=logs/swin_3d_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@colorado.edu

# ============================================================
# Swin Transformer 3D Training - Optimized for blanca-biokem (2x A100 80GB)
# ============================================================
# Architecture: Swin Transformer 3D (Transformer - Very High Memory)
# Expected training time: ~44-48 hours for 150 epochs
# Memory usage: ~50-65 GB per GPU (3D windowed attention)
# Input shape: (32, 256, 256) - 32 slices of 256x256
# ============================================================

MODEL="swin"
DIM="3d"

# Optimized hyperparameters for 3D Swin Transformer on 2x A100 80GB with DDP
# 3D Swin has windowed attention but still very memory intensive
# DDP overhead: ~30% extra memory per GPU for transformer gradients
EPOCHS=150
BATCH_SIZE=2                     # Per-GPU batch size (minimum viable)
ITERATIONS_PER_EPOCH=250         # Shared across all 3D models
LEARNING_RATE=3e-5               # Lower LR for stable 3D transformer training
GRADIENT_ACCUMULATION=16         # Effective batch = 2*16*2GPUs = 64
NUM_WORKERS=4                    # DataLoader workers per GPU

echo "=============================================="
echo "CellMap Model Comparison - Swin Transformer 3D"
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

# Load Blanca-specific modules
module purge
# Example: module load miniconda/23-11  # adjust to Blanca's module name
module load anaconda 2>/dev/null || true

# If the module system doesn't expose Anaconda on this node, fall back to a user install.
if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/software/anaconda/etc/profile.d/conda.sh" ]; then
        source "$HOME/software/anaconda/etc/profile.d/conda.sh"
    elif [ -f "$HOME/software/anaconda/bin/activate" ]; then
        source "$HOME/software/anaconda/bin/activate"
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Load a conda module or install conda under ~/software/anaconda." >&2
    exit 2
fi

# Activate conda environment. Update the path if you install elsewhere.
conda activate /projects/$USER/software/anaconda/envs/cellmap || {
    echo "WARNING: Failed to activate /projects/$USER/software/anaconda/envs/cellmap"
    echo "Attempting to activate by name: cellmap"
    conda activate cellmap || true
}

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Environment settings for optimal A100 performance with 3D transformers
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Enable TF32 for faster transformer training on A100
export NVIDIA_TF32_OVERRIDE=1

# Enable gradient checkpointing for 3D transformers (saves memory)
export GRADIENT_CHECKPOINTING=1

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
echo "Estimated completion: ~44-48 hours"
echo "NOTE: Memory intensive - using gradient accumulation"
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
