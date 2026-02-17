#!/bin/bash
#SBATCH --job-name=infer_step3
#SBATCH --reservation=gsgeorge_9034
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=150g
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/infer_step3_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/infer_step3_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# Step 3 ONLY: Re-run inference with tuned thresholds
# ============================================================
# Steps 0-2 already completed by job 31119803.
# Peak RSS was ~140 GB, so 200 GB is plenty.
# No TTA needed (Step 3 skips --tta), ~10-12 hrs for 46 volumes.

set -euo pipefail

echo "============================================================"
echo " MONAI CellMap — Step 3: Inference with Tuned Thresholds"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

# --- Environment ---
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

echo "Python: $(which python)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --- Paths ---
REPO_DIR="/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation"
EXP_DIR="$REPO_DIR/experiments/monai_cellmap"
RUNS_DIR="/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap"
PRED_DIR="$RUNS_DIR/predictions"
cd "$REPO_DIR"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo ""
echo "=== Step 3: Re-evaluate with tuned thresholds ==="
echo "  Thresholds: $PRED_DIR/thresholds.json"
echo "  Ensemble:   $RUNS_DIR/ensemble_map.json"
echo "  Output:     ${PRED_DIR}_tuned"
echo ""

python "$EXP_DIR/inference.py" \
    --ensemble \
    --postprocess \
    --split val \
    --sw-roi-size 128,128,128 \
    --sw-batch-size 4 \
    --overlap 0.5 \
    --output-dir "${PRED_DIR}_tuned" \
    --threshold-json "$PRED_DIR/thresholds.json" \
    --ensemble-json "$RUNS_DIR/ensemble_map.json"

echo ""
echo "============================================================"
echo " Step 3 complete: $(date)"
echo " Tuned results: ${PRED_DIR}_tuned"
echo "============================================================"
