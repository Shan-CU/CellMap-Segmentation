#!/bin/bash
#SBATCH --job-name=swin_baseline_blanca
#SBATCH --partition=blanca-biokem
#SBATCH --account=ucb-general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/swin_blanca_%j.out
#SBATCH --error=logs/swin_blanca_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shku2971@colorado.edu

# === Blanca-ready Slurm script for SwinTransformer training ===
# User-configurable variables (kept for convenience)
PARTITION="blanca-biokem"   # using biokem (2x A100 per node)
ACCOUNT="ucb-general"       # your Blanca account/project
EMAIL="shku2971@colorado.edu"    # mail notifications

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

cd $SLURM_SUBMIT_DIR

# --- Load modules or environment manager available on Blanca ---
module purge
# Example: module load miniconda/23-11  # adjust to Blanca's module name
module load anaconda || true

# Activate conda environment. Update the path if you install elsewhere.
conda activate /projects/$USER/software/anaconda/envs/cellmap || {
    echo "WARNING: Failed to activate /projects/$USER/software/anaconda/envs/cellmap"
    echo "Attempting to activate by name: cellmap"
    conda activate cellmap || true
}

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'n/a')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'n/a')"

mkdir -p logs tensorboard checkpoints

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Optional: stage data to local scratch (uncomment & update paths if Blanca provides $SLURM_SCRATCH)
# if [ -n "$SLURM_SCRATCH" ]; then
#   rsync -a /scratch/$USER/cellmap_data $SLURM_SCRATCH/ && export CELLMAP_DATA_DIR=$SLURM_SCRATCH/cellmap_data
# fi

# Start TensorBoard in background
TB_PORT=$((6006 + SLURM_JOB_ID % 1000))
TB_LOGDIR="$SLURM_SUBMIT_DIR/tensorboard"
tensorboard --logdir=$TB_LOGDIR --port=$TB_PORT --bind_all &
TB_PID=$!

echo "Starting SwinTransformer training..."
python examples/train_swin_baseline.py
TRAIN_EXIT_CODE=$?

echo "Stopping TensorBoard..."
kill $TB_PID 2>/dev/null || true

echo "End time: $(date)"
exit $TRAIN_EXIT_CODE
