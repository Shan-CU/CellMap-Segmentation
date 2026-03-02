#!/bin/bash
# ============================================================================
# Archive Phase 2 v2/v3 runs and kill their jobs
# ============================================================================
# Run ONCE before launching v4:
#   cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
#   bash training/slurm/phase2v4/archive_v2v3.sh
# ============================================================================

set -euo pipefail
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# --- Kill all running p2/p2v3 jobs ---
echo "=== Killing Phase 2 v2/v3 jobs ==="

JOBS=$(squeue -u "$(whoami)" -h -o '%i %j' 2>/dev/null | grep -E '^[0-9]+ p2_|^[0-9]+ p2v3_' || true)

if [ -n "$JOBS" ]; then
    echo "$JOBS" | while IFS=' ' read -r jid jname; do
        echo "  Cancelling ${jid} (${jname})"
        scancel "$jid" 2>/dev/null || true
    done
    echo "  Waiting 10s for GPU release..."
    sleep 10
else
    echo "  No v2/v3 jobs found."
fi

echo ""

# --- Archive run directories ---
echo "=== Archiving v2/v3 run directories ==="
ARCHIVE_DIR="runs/ablation/phase2_v2v3_archive"
mkdir -p "$ARCHIVE_DIR"

COUNT=0
for run_dir in runs/ablation/p2_* runs/ablation/p2v3_*; do
    if [ -d "$run_dir" ]; then
        name=$(basename "$run_dir")
        echo "  Moving ${name} → ${ARCHIVE_DIR}/"
        mv "$run_dir" "$ARCHIVE_DIR/"
        COUNT=$((COUNT + 1))
    fi
done

echo ""
echo "Archived ${COUNT} run directories to ${ARCHIVE_DIR}/"
echo ""
echo "Contents:"
ls -1 "$ARCHIVE_DIR/" 2>/dev/null || echo "  (empty)"
echo ""
echo "Ready to launch v4: bash training/slurm/phase2v4/launch_all.sh"
