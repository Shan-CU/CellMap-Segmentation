#!/bin/bash
#SBATCH --job-name=viz_gt_overlay
#SBATCH --partition=general
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/viz_gt_overlay_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/viz_gt_overlay_%j.err
#SBATCH --dependency=afterok:32303380

set -euo pipefail

echo "=== Visualize GT Overlay ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
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

# Step 1: Validate the regenerated NIfTI data
echo "=== Step 1: Validate NIfTI data ==="
python auto3dseg/validate_nifti_v2.py --workers 4
echo ""

# Step 2: Generate overlay PNGs (20 diverse crops)
echo "=== Step 2: Generate GT overlay PNGs ==="
python auto3dseg/visualize_gt_overlay.py --crops 20
echo ""

echo "=== Done: $(date) ==="
echo "Overlays saved to: auto3dseg/nifti_data_v2/qa_overlays/"
