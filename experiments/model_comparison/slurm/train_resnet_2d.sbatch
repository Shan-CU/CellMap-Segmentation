#!/bin/bash
#SBATCH --job-name=resnet_2d
#SBATCH --partition=blanca-biokem
#SBATCH --account=blanca-biokem
#SBATCH --qos=blanca-biokem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=320G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/resnet_2d_%j.out
#SBATCH --error=logs/resnet_2d_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@colorado.edu

# ============================================================
# ResNet 2D Training - Optimized for blanca-biokem (2x A100 80GB)
# ============================================================
# Architecture: ResNet 2D (CNN - Moderate Memory)
# Expected training time: ~28-40 hours for 200 epochs
# Memory usage: ~12-18 GB per GPU
# ============================================================

MODEL="resnet"
DIM="2d"

# Optimized hyperparameters for 2D ResNet on 2x A100 80GB with DDP
# ResNet uses more memory than UNet due to residual connections
# DDP overhead: ~20% extra memory for gradient synchronization
EPOCHS=200
BATCH_SIZE=32                    # Per-GPU batch (conservative for DDP)
ITERATIONS_PER_EPOCH=500         # Shared across all 2D models
LEARNING_RATE=1e-4
GRADIENT_ACCUMULATION=4          # Effective batch = 32*4*2GPUs = 256
NUM_WORKERS=8                    # DataLoader workers per GPU

echo "=============================================="
echo "CellMap Model Comparison - ResNet 2D"
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
module load slurm/blanca
module load anaconda

# Initialize conda if not already done (required on Blanca compute nodes)
if ! command -v conda &> /dev/null; then
    if [ -f "/curc/sw/anaconda3/latest/etc/profile.d/conda.sh" ]; then
        source "/curc/sw/anaconda3/latest/etc/profile.d/conda.sh"
    elif [ -f "$HOME/software/anaconda/etc/profile.d/conda.sh" ]; then
        source "$HOME/software/anaconda/etc/profile.d/conda.sh"
    fi
fi

# Activate conda environment - try user environment first, then project location
conda activate cellmap 2>/dev/null || \
    conda activate /projects/$USER/software/anaconda/envs/cellmap 2>/dev/null || \
    conda activate /projects/shku2971/software/anaconda/envs/cellmap 2>/dev/null || {
        echo "ERROR: Could not activate cellmap environment"
        echo "Please ensure the environment exists. Create it with:"
        echo "  conda create -n cellmap python=3.10 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia"
        exit 1
    }

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Environment settings for optimal A100 performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

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
echo "Estimated completion: ~28-40 hours"
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
