#!/bin/bash
# ============================================================================
# Kill all Phase 2 v4 jobs
# ============================================================================
# Usage: Run on Longleaf login node:
#   bash training/slurm/phase2v4/kill_all.sh
# ============================================================================

echo "Cancelling all p2v4 jobs for $(whoami)..."
echo ""

JOBS=$(squeue -u "$(whoami)" -h -o '%i %j' 2>/dev/null | grep 'p2v4' || true)

if [ -z "$JOBS" ]; then
    echo "No p2v4 jobs found."
    exit 0
fi

echo "$JOBS" | while IFS=' ' read -r jid jname; do
    echo "  Cancelling ${jid} (${jname})"
    scancel "$jid"
done

N=$(echo "$JOBS" | wc -l)
echo ""
echo "Cancelled ${N} jobs."
