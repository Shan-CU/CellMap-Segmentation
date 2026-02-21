#!/bin/bash
#SBATCH --job-name=monai_infer
#SBATCH --reservation=gsgeorge_9034
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH --mem=900g
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/inference_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/inference_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# MONAI CellMap Inference — 4-Step Pipeline
# ============================================================
# Step 0: Evaluate ensemble (fast crop-based → get ensemble_map.json)
# Step 1: Full sliding-window inference + TTA + ensemble + prob maps
# Step 2: Threshold tuning on probability maps
# Step 3: Re-run inference with tuned thresholds (no probs, faster)
#
# Memory: 900 GB RAM for 800³ volumes (14ch × float32 × 3 models parallel).
# GPU: 3× L40S 48 GB — one per model, run in parallel.
# Time: ~6-10 hours for 46 val volumes × 3 models × 8 TTA flips (parallel).
#
# Usage:
#   sbatch experiments/monai_cellmap/slurm/inference.sbatch
#   sbatch --dependency=afterok:<train_jobid> experiments/monai_cellmap/slurm/inference.sbatch

set -euo pipefail

echo "============================================================"
echo " MONAI CellMap — Inference Pipeline"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

# --- Environment ---
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

# --- Verify ---
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "MONAI: $(python -c 'import monai; print(monai.__version__)')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --- Paths ---
REPO_DIR="/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation"
EXP_DIR="$REPO_DIR/experiments/monai_cellmap"
RUNS_DIR="/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap"
PRED_DIR="$RUNS_DIR/predictions"
cd "$REPO_DIR"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo ""
echo "=== Step 0: Evaluate ensemble (crop-based, fast) ==="
echo "  - Run each model on val crops to get per-class Dice"
echo "  - Pick best model per class → ensemble_map.json"
echo ""

python "$EXP_DIR/evaluate_ensemble.py"

echo ""
echo "=== Step 1: Full inference on validation set ==="
echo "  - 3 models, per-class ensemble from Step 0"
echo "  - TTA: 8 flip combinations"
echo "  - Sliding window: 128³, overlap 0.5, Gaussian weighting"
echo "  - Saving probability maps for threshold tuning"
echo ""

python "$EXP_DIR/inference.py" \
    --ensemble \
    --tta \
    --postprocess \
    --split val \
    --sw-roi-size 128,128,128 \
    --sw-batch-size 4 \
    --overlap 0.5 \
    --output-dir "$PRED_DIR" \
    --save-probs \
    --ensemble-json "$RUNS_DIR/ensemble_map.json"

echo ""
echo "=== Step 2: Threshold tuning ==="
echo ""

python "$EXP_DIR/tune_thresholds.py" \
    --probs-dir "$PRED_DIR" \
    --output "$PRED_DIR/thresholds.json"

echo ""
echo "=== Step 3: Re-evaluate with tuned thresholds ==="
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
echo " Inference complete: $(date)"
echo " Ensemble map: $RUNS_DIR/ensemble_map.json"
echo " Results: $PRED_DIR"
echo " Tuned results: ${PRED_DIR}_tuned"
echo "============================================================"
