#!/bin/bash
#SBATCH --job-name=cellmap_single_model
#SBATCH --partition=aa100
#SBATCH --account=ucb-general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/single_model_%j.out
#SBATCH --error=logs/single_model_%j.err

# ============================================================
# Single Model Training Script (Template)
# ============================================================
# Customize the MODEL and DIM variables below to train a specific model
#
# Usage:
#   sbatch --export=MODEL=unet,DIM=2d run_single_model.sbatch
#   sbatch --export=MODEL=swin,DIM=2d run_single_model.sbatch
#   sbatch --export=MODEL=vit,DIM=3d run_single_model.sbatch
# ============================================================

# Default values if not provided via --export
MODEL=${MODEL:-unet}
DIM=${DIM:-2d}
EPOCHS=${EPOCHS:-100}

echo "=============================================="
echo "CellMap Single Model Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL"
echo "Dimension: $DIM"
echo "Epochs: $EPOCHS"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
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

# Load modules
module purge
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true

# Activate conda environment
source ~/.bashrc
conda activate cellmap 2>/dev/null || {
    echo "Trying alternative activation..."
    source /projects/$USER/software/anaconda/bin/activate cellmap
}

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

# Environment settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

N_GPUS=$(nvidia-smi -L | wc -l)
echo "Using $N_GPUS GPUs"

# Start TensorBoard
TB_PORT=$((6006 + SLURM_JOB_ID % 1000))
tensorboard --logdir=experiments/model_comparison/tensorboard --port=$TB_PORT --bind_all &
TB_PID=$!

# Run training
echo ""
echo "Starting training: $MODEL ($DIM)..."
echo ""

torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model $MODEL \
    --dim $DIM \
    --epochs $EPOCHS \
    --save_features \
    --vis_every 10

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
