#!/bin/bash
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=11-00:00:00
#SBATCH --job-name=tb_phase2
#SBATCH --output=runs/ablation/logs/tensorboard_%j.out
#SBATCH --error=runs/ablation/logs/tensorboard_%j.err

# ============================================================================
# Phase 2 TensorBoard Server
# Monitors all Phase 2 experiments (2D + 3D) in a single TensorBoard instance.
# Access via SSH tunnel: ssh -L 6006:<node>:6006 longleaf.unc.edu
# Then open http://localhost:6006 in your browser.
# ============================================================================

set -euo pipefail

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# Build --logdir_spec from existing tensorboard directories (no symlinks needed)
LOGDIR_SPEC=""
for dir in runs/ablation/arch_*/tensorboard; do
    if compgen -G "$dir/events.out.tfevents.*" > /dev/null 2>&1; then
        name=$(echo "$dir" | sed 's|runs/ablation/||; s|/tensorboard||')
        if [ -n "$LOGDIR_SPEC" ]; then
            LOGDIR_SPEC="${LOGDIR_SPEC},"
        fi
        LOGDIR_SPEC="${LOGDIR_SPEC}${name}:$(pwd)/${dir}"
        echo "  ✓ ${name}"
    fi
done

if [ -z "$LOGDIR_SPEC" ]; then
    echo "ERROR: No tensorboard event files found. Waiting 5 min and retrying..."
    sleep 300
    exec "$0"
fi

PORT=${TB_PORT:-6006}
NODE=$(hostname)

echo "============================================"
echo "Phase 2 TensorBoard"
echo "Node: ${NODE} | Port: ${PORT}"
echo "Runs: ${LOGDIR_SPEC}"
echo ""
echo "SSH tunnel command:"
echo "  ssh -L ${PORT}:${NODE}:${PORT} longleaf.unc.edu"
echo "Then open: http://localhost:${PORT}"
echo "============================================"

# Launch TensorBoard with explicit logdir_spec (more reliable than symlinks)
tensorboard \
    --logdir_spec="${LOGDIR_SPEC}" \
    --port=${PORT} \
    --bind_all \
    --reload_interval=30 \
    --reload_multifile=true \
    --load_fast=false

