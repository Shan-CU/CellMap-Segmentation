#!/bin/bash

#SBATCH --job-name=unet_run2
#SBATCH --partition=nvidia-a100
#SBATCH --account=reguser
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/unet_run2_%j.out
#SBATCH --error=logs/unet_run2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=george.stephenson@colorado.edu

# ============================================================
# CellMap UNet Training Run 2 on Fiji
# Separate output directory for parallel experiments
# ============================================================

RUN_NAME="unet_run2"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Run Name: $RUN_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Go to project directory
cd $SLURM_SUBMIT_DIR

# Initialize and activate conda (fiji uses local miniconda3)
source ~/miniconda3/bin/activate cellmap || {
    echo "ERROR: Failed to activate conda environment"
    echo "Checking miniconda3 location..."
    ls -la ~/miniconda3/envs/
    exit 1
}

# Verify GPU access
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")')"

# Create run-specific directories
mkdir -p logs checkpoints/$RUN_NAME tensorboard/$RUN_NAME

# ============================================================
# Start TensorBoard in background
# ============================================================
TB_PORT=$((7006 + SLURM_JOB_ID % 1000))
TB_LOGDIR="$SLURM_SUBMIT_DIR/tensorboard/$RUN_NAME"

echo "=========================================="
echo "TENSORBOARD CONNECTION INFO"
echo "=========================================="
echo "TensorBoard starting on port: $TB_PORT"
echo "Log directory: $TB_LOGDIR"
echo "Node: $SLURM_NODELIST"
echo ""
echo "To connect, run this SSH tunnel from your LOCAL machine:"
echo "  ssh -L $TB_PORT:$SLURM_NODELIST:$TB_PORT gest9386@fiji.colorado.edu"
echo ""
echo "Then open in browser: http://localhost:$TB_PORT"
echo "=========================================="

# Start TensorBoard in background
tensorboard --logdir=$TB_LOGDIR --port=$TB_PORT --bind_all &
TB_PID=$!
echo "TensorBoard PID: $TB_PID"

# Set environment variables for run-specific paths
export CELLMAP_RUN_NAME=$RUN_NAME
export CELLMAP_CHECKPOINT_DIR="checkpoints/$RUN_NAME"
export CELLMAP_TENSORBOARD_DIR="tensorboard/$RUN_NAME"

# Run training
echo "Starting UNet training (Run 2)..."
python examples/train_unet_baseline.py
TRAIN_EXIT_CODE=$?

# Cleanup TensorBoard
echo "Stopping TensorBoard..."
kill $TB_PID 2>/dev/null

echo "=========================================="
echo "End time: $(date)"
echo "Training exit code: $TRAIN_EXIT_CODE"
echo "Checkpoints saved to: checkpoints/$RUN_NAME"
echo "TensorBoard logs: tensorboard/$RUN_NAME"
echo "=========================================="

exit $TRAIN_EXIT_CODE
