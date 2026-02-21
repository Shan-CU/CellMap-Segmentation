#!/bin/bash
# ============================================================
# Submit all 4 model comparison jobs to Blanca Biokem
# Each runs on its own node with 2× A100 GPUs
# ============================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Submitting Model Comparison Jobs — Blanca Biokem    ║"
echo "║  Loss: Balanced Softmax Tversky τ=1.0               ║"
echo "║  Models: UNet, ResNet, Swin, ViT (all 2D)           ║"
echo "║  Hardware: 2× A100 per job (DDP)                    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

mkdir -p logs

JOBS=()
for model in unet resnet swin vit; do
    SBATCH_FILE="train_${model}_blanca.sbatch"
    if [ ! -f "$SBATCH_FILE" ]; then
        echo "  ✗ $SBATCH_FILE not found, skipping $model"
        continue
    fi
    JOB_ID=$(sbatch "$SBATCH_FILE" | awk '{print $4}')
    JOBS+=("$JOB_ID")
    echo "  ✓ ${model^^} → Job $JOB_ID"
done

echo ""
echo "Submitted ${#JOBS[@]} jobs."
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  sacct -j $(IFS=,; echo "${JOBS[*]}") --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize"
echo ""
echo "View results (after completion):"
echo "  ./run.sh summary"
echo "  python analyze_results.py"
