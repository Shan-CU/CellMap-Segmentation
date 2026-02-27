#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/tb_%j.out
#SBATCH --error=runs/ablation/logs/tb_%j.err

# ============================================================================
# TensorBoard server for Phase 2 architecture comparison
# ============================================================================
# Watches runs/ablation/ for all experiment subdirectories.
# Access via: ssh -L 6006:<node>:6006 longleaf.unc.edu
# Then open http://localhost:6006
# ============================================================================

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

echo "TensorBoard starting on $(hostname):6006"
echo "SSH tunnel: ssh -L 6006:$(hostname):6006 longleaf.unc.edu"
echo "URL: http://localhost:6006"

tensorboard --logdir runs/ablation --port 6006 --bind_all --reload_interval 60
