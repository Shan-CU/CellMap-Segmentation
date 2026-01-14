#!/bin/bash

#SBATCH --job-name=unet_baseline
#SBATCH --partition=nvidia-a100
#SBATCH --account=reguser
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/unet_%j.out
#SBATCH --error=logs/unet_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=george.stephenson@colorado.edu

# ============================================================
# CellMap UNet Baseline Training on Fiji
# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Go to project directory
cd $SLURM_SUBMIT_DIR

# Load modules (adjust based on fiji's available modules)
module purge
module load anaconda

# Activate your conda environment
# Adjust path based on where your env is on fiji
conda activate cellmap

# Verify GPU access
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")')"

# Create logs directory if it doesn't exist
mkdir -p logs tensorboard

# ============================================================
# Start TensorBoard in background
# ============================================================
TB_PORT=$((6006 + SLURM_JOB_ID % 1000))
TB_LOGDIR="$SLURM_SUBMIT_DIR/tensorboard"

echo "=========================================="
echo "TENSORBOARD CONNECTION INFO"
echo "=========================================="
echo "TensorBoard starting on port: $TB_PORT"
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

# Run training
echo "Starting UNet training..."
python examples/train_unet_baseline.py
TRAIN_EXIT_CODE=$?

# Cleanup TensorBoard
echo "Stopping TensorBoard..."
kill $TB_PID 2>/dev/null

echo "=========================================="
echo "End time: $(date)"
echo "Training exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
