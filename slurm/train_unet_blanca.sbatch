#!/bin/bash
#SBATCH --job-name=unet_baseline_blanca
#SBATCH --partition=blanca-biokem
#SBATCH --account=blanca-biokem
#SBATCH --qos=blanca-biokem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/unet_blanca_%j.out
#SBATCH --error=logs/unet_blanca_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=george.stephenson@colorado.edu

# === Blanca-ready Slurm script for UNet training ===
# User-configurable variables (kept for convenience)
PARTITION="blanca-biokem"   # e.g. blanca-biokem (2x A100 per node)
ACCOUNT="ucb-general"         # your Blanca account/project name
EMAIL="george.stephenson@colorado.edu"    # mail notifications

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

cd $SLURM_SUBMIT_DIR

# --- Load modules or environment manager available on Blanca ---
module purge
module load anaconda 2>/dev/null || true

# If the module system doesn't expose Anaconda on this node, fall back to a user install.
if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/software/anaconda/etc/profile.d/conda.sh" ]; then
        source "$HOME/software/anaconda/etc/profile.d/conda.sh"
    elif [ -f "$HOME/software/anaconda/bin/activate" ]; then
        # Older installs sometimes only ship the activate script.
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

echo "Starting UNet training with DDP (torchrun)..."
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $N_GPUS GPUs, launching $N_GPUS processes with torchrun"
torchrun --standalone --nproc_per_node=$N_GPUS examples/train_unet_baseline.py
TRAIN_EXIT_CODE=$?

echo "Stopping TensorBoard..."
kill $TB_PID 2>/dev/null || true

echo "End time: $(date)"
exit $TRAIN_EXIT_CODE
