#!/bin/bash

#SBATCH --job-name=swin_baseline
#SBATCH --partition=nvidia-a100
#SBATCH --account=reguser
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/swin_%j.out
#SBATCH --error=logs/swin_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=george.stephenson@colorado.edu

# ============================================================
# CellMap SwinTransformer Baseline Training on Fiji
# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
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

# Create logs directory
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
echo "Starting SwinTransformer training..."
python examples/train_swin_baseline.py
TRAIN_EXIT_CODE=$?

# Cleanup TensorBoard
echo "Stopping TensorBoard..."
kill $TB_PID 2>/dev/null

echo "=========================================="
echo "End time: $(date)"
echo "Training exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
