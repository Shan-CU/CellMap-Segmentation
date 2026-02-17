#!/bin/bash
#SBATCH --job-name=convert_nifti_v2
#SBATCH --partition=general_big
#SBATCH --account=rc_alain_pi
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/convert_nifti_v2_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/convert_nifti_v2_%j.err

set -euo pipefail

echo "=== Convert Zarr → NIfTI v2 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Start:  $(date)"
echo ""

# Activate conda
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

echo "Python: $(which python)"
echo ""

# Run conversion (workers=0 → auto-detect from CPU count)
python auto3dseg/convert_zarr_to_nifti_v2.py --workers 0

echo ""
echo "=== Done: $(date) ==="
