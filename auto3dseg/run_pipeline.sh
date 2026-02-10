#!/bin/bash
# ============================================================
# Auto3DSeg Quick Start - Submit all pipeline steps
#
# Cluster setup:
#   Convert & Analyze → Sycamore 'batch' partition (CPU, fast scheduling)
#   Train             → Longleaf 'l40-gpu' partition (2x L40S GPUs)
#
# Usage (run from Sycamore login node):
#   cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
#
#   # Option 1: Run just the data analysis (recommended first)
#   bash auto3dseg/run_pipeline.sh analyze
#
#   # Option 2: Submit training job on Longleaf (after analysis is done)
#   bash auto3dseg/run_pipeline.sh train
#
#   # Option 3: Convert only
#   bash auto3dseg/run_pipeline.sh convert
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Ensure log directory exists
mkdir -p logs

MODE="${1:-analyze}"

case "$MODE" in
    convert)
        echo "Submitting data conversion job (Sycamore batch)..."
        sbatch auto3dseg/auto3dseg_convert.sbatch
        ;;

    analyze)
        echo "=== Auto3DSeg Data Analysis Pipeline ==="
        echo "  Cluster: Sycamore (batch partition, CPU-only)"
        echo ""
        echo "Step 1: Converting zarr → NIfTI..."
        JOB1=$(sbatch --parsable auto3dseg/auto3dseg_convert.sbatch)
        echo "  Submitted job $JOB1 (conversion)"

        echo "Step 2: Running data analysis (after conversion)..."
        JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 auto3dseg/auto3dseg_analyze.sbatch)
        echo "  Submitted job $JOB2 (analysis, depends on $JOB1)"

        echo ""
        echo "Monitor with:"
        echo "  squeue -u $USER"
        echo "  tail -f logs/auto3dseg_convert_${JOB1}.out"
        echo "  tail -f logs/auto3dseg_analyze_${JOB2}.out"
        echo ""
        echo "After completion, review:"
        echo "  cat auto3dseg/work_dir/datastats.yaml"
        echo "  python auto3dseg/interpret_results.py"
        echo ""
        echo "Then submit training on Longleaf:"
        echo "  bash auto3dseg/run_pipeline.sh train"
        ;;

    train)
        echo "=== Auto3DSeg Training ==="
        echo "  Cluster: Longleaf (l40-gpu partition, 2x L40S)"
        echo ""

        # Verify data exists
        DATALIST="auto3dseg/nifti_data/datalist.json"
        if [ ! -f "$DATALIST" ]; then
            echo "ERROR: $DATALIST not found!"
            echo "Run 'bash auto3dseg/run_pipeline.sh analyze' first."
            exit 1
        fi

        # Submit to Longleaf via ssh
        echo "Submitting training job to Longleaf..."
        TRAIN_JOB=$(ssh longleaf.unc.edu "cd $PROJECT_DIR && sbatch --parsable auto3dseg/auto3dseg_train.sbatch" 2>/dev/null)
        if [ -n "$TRAIN_JOB" ]; then
            echo "  Submitted Longleaf job $TRAIN_JOB"
            echo ""
            echo "Monitor with:"
            echo "  ssh longleaf.unc.edu 'squeue -u $USER'"
            echo "  tail -f logs/auto3dseg_train_${TRAIN_JOB}.out"
        else
            echo "  Could not submit via SSH. Log into Longleaf and run:"
            echo "    cd $PROJECT_DIR"
            echo "    sbatch auto3dseg/auto3dseg_train.sbatch"
        fi
        ;;

    *)
        echo "Usage: $0 {convert|analyze|train}"
        echo ""
        echo "  convert  - Convert zarr → NIfTI (Sycamore batch, CPU)"
        echo "  analyze  - Convert + data analysis (Sycamore batch, CPU)"
        echo "  train    - Full Auto3DSeg training (Longleaf l40-gpu, 2x L40S)"
        echo ""
        echo "Recommended workflow:"
        echo "  1. bash auto3dseg/run_pipeline.sh analyze"
        echo "  2. python auto3dseg/interpret_results.py"
        echo "  3. bash auto3dseg/run_pipeline.sh train"
        exit 1
        ;;
esac
