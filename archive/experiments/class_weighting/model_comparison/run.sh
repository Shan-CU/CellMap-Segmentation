#!/bin/bash
# ============================================================
# Model Comparison Launcher
#
# Usage:
#   ./run.sh submit-all     Submit all 4 model jobs to Blanca
#   ./run.sh submit MODEL   Submit a single model (unet|resnet|swin|vit)
#   ./run.sh summary        Print comparison table from results
#   ./run.sh tensorboard    Start TensorBoard on port 6008
#   ./run.sh status         Check SLURM job status
#   ./run.sh local MODEL    Run locally (single GPU, no SLURM)
# ============================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defer set -u until after conda activation (avoids CONDA_BACKUP_* errors)
_activate_conda() {
    if command -v micromamba &>/dev/null; then
        eval "$(micromamba shell hook -s bash)"
        micromamba activate csc
    elif command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate csc
    else
        echo "ERROR: No conda/micromamba found" >&2
        exit 2
    fi
}

COMMAND="${1:-help}"
MODEL="${2:-}"

case "$COMMAND" in

# ─── Submit all 4 models to Blanca ───────────────────────────
submit-all)
    echo "Submitting all 4 model comparison jobs to Blanca Biokem..."
    mkdir -p logs
    for model in unet resnet swin vit; do
        SBATCH_FILE="train_${model}_blanca.sbatch"
        if [ ! -f "$SBATCH_FILE" ]; then
            echo "  ERROR: $SBATCH_FILE not found" >&2
            continue
        fi
        JOB_ID=$(sbatch "$SBATCH_FILE" | awk '{print $4}')
        echo "  $model → Job $JOB_ID ($SBATCH_FILE)"
    done
    echo ""
    echo "Monitor with: squeue -u \$USER"
    ;;

# ─── Submit a single model ───────────────────────────────────
submit)
    if [ -z "$MODEL" ]; then
        echo "Usage: ./run.sh submit MODEL  (unet|resnet|swin|vit)" >&2
        exit 1
    fi
    SBATCH_FILE="train_${MODEL}_blanca.sbatch"
    if [ ! -f "$SBATCH_FILE" ]; then
        echo "ERROR: $SBATCH_FILE not found" >&2
        exit 1
    fi
    mkdir -p logs
    JOB_ID=$(sbatch "$SBATCH_FILE" | awk '{print $4}')
    echo "$MODEL → Job $JOB_ID"
    ;;

# ─── Print comparison summary ────────────────────────────────
summary)
    _activate_conda
    set -u
    python train.py --summary
    ;;

# ─── TensorBoard ─────────────────────────────────────────────
tensorboard)
    echo "Starting TensorBoard on port 6008..."
    echo "Open: http://localhost:6008"
    tensorboard --logdir runs --port 6008 --bind_all
    ;;

# ─── Check SLURM job status ──────────────────────────────────
status)
    echo "Current SLURM jobs:"
    squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.6D %.4C %.15R" || \
        echo "squeue not available (not on a SLURM cluster?)"
    echo ""
    echo "Completed results:"
    for f in results/mc_*_results.json; do
        if [ -f "$f" ]; then
            model=$(python -c "import json; d=json.load(open('$f')); print(d.get('model_name','?'))")
            dice=$(python -c "import json; d=json.load(open('$f')); print(f\"{d.get('best_dice',0):.4f}\")")
            elapsed=$(python -c "import json; d=json.load(open('$f')); print(f\"{d.get('elapsed_min',0):.0f}m\")")
            echo "  ✓ $model: Dice=$dice ($elapsed)"
        fi
    done
    [ ! -f results/mc_*_results.json ] 2>/dev/null && echo "  (no results yet)"
    ;;

# ─── Run locally (single GPU, no SLURM) ──────────────────────
local)
    if [ -z "$MODEL" ]; then
        echo "Usage: ./run.sh local MODEL  (unet|resnet|swin|vit)" >&2
        exit 1
    fi
    _activate_conda
    set -u
    mkdir -p logs checkpoints runs results visualizations metrics features
    echo "Running $MODEL locally (single GPU)..."
    python train.py \
        --model "$MODEL" \
        --resume \
        --no_compile
    ;;

# ─── Run all models locally (sequential) ─────────────────────
local-all)
    _activate_conda
    set -u
    mkdir -p logs checkpoints runs results visualizations metrics features
    for model in unet resnet swin vit; do
        echo ""
        echo "═══════════════════════════════════════"
        echo "  Running $model locally..."
        echo "═══════════════════════════════════════"
        python train.py --model "$model" --resume --no_compile
    done
    echo ""
    echo "All models complete. Summary:"
    python train.py --summary
    ;;

# ─── Help ─────────────────────────────────────────────────────
help|*)
    echo "Model Comparison Launcher"
    echo ""
    echo "Usage: ./run.sh COMMAND [MODEL]"
    echo ""
    echo "Commands:"
    echo "  submit-all          Submit all 4 models to Blanca Biokem"
    echo "  submit MODEL        Submit one model (unet|resnet|swin|vit)"
    echo "  summary             Print comparison results table"
    echo "  tensorboard         Start TensorBoard (port 6008)"
    echo "  status              Check SLURM job status + completed results"
    echo "  local MODEL         Run one model locally (single GPU)"
    echo "  local-all           Run all 4 models locally (sequential)"
    echo "  help                Show this help"
    ;;
esac
