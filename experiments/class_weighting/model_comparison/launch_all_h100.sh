#!/bin/bash
# ============================================================
# Submit all 4 model comparison jobs to Blanca H100 (preemptable)
# Each runs on bgpu-g6-u[16,18] with 3× H100 GPUs
# Auto-requeues on preemption with checkpoint resume
# ============================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Submitting Model Comparison Jobs — H100 Preemptable     ║"
echo "║  Loss: Balanced Softmax Tversky τ=1.0                   ║"
echo "║  Models: UNet, ResNet, Swin, ViT (all 2D)               ║"
echo "║  Hardware: 3× H100 per job (DDP, preemptable)            ║"
echo "║  Wall time: 24h (auto-requeue on preemption)             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

mkdir -p logs

JOBS=()
for model in unet resnet swin vit; do
    SBATCH_FILE="train_${model}_h100.sbatch"
    if [ ! -f "$SBATCH_FILE" ]; then
        echo "  ✗ $SBATCH_FILE not found, skipping $model"
        continue
    fi
    JOB_ID=$(sbatch "$SBATCH_FILE" | awk '{print $4}')
    JOBS+=("$JOB_ID")
    echo "  ✓ ${model^^} → Job $JOB_ID (preemptable, 3× H100)"
done

echo ""
echo "Submitted ${#JOBS[@]} preemptable jobs."
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  sacct -j $(IFS=,; echo "${JOBS[*]}") --format=JobID,JobName,State,Elapsed,MaxRSS"
echo ""
echo "Note: Jobs auto-resubmit on preemption. Check logs/ for requeue history."
echo ""
echo "View results (after completion):"
echo "  ./run.sh summary"
echo "  python analyze_results.py"
