#!/bin/bash
# ============================================================================
# Launch Phase 2 TensorBoard locally (login node or interactive session)
# 
# Usage:
#   ./training/slurm/launch_tb_phase2.sh          # default port 6006
#   ./training/slurm/launch_tb_phase2.sh 6007      # custom port
#
# Access via SSH tunnel:
#   ssh -L 6006:<hostname>:6006 longleaf.unc.edu   (if on Longleaf)
#   ssh -L 6006:<hostname>:6006 sycamore.unc.edu   (if on Sycamore)
# Then open http://localhost:6006
# ============================================================================

set -euo pipefail

PORT=${1:-6006}

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# Create Phase 2 TB directory with symlinks
TB_DIR="runs/ablation/tb_phase2"
mkdir -p "${TB_DIR}"

echo "Setting up Phase 2 TensorBoard symlinks..."

# 2D experiments
for model in resnet unet swin vit; do
    src="$(pwd)/runs/ablation/arch_2d_${model}/tensorboard"
    if [ -d "$src" ]; then
        ln -sfn "$src" "${TB_DIR}/2d_${model}"
        echo "  ✓ 2d_${model}"
    else
        echo "  ✗ 2d_${model} (not yet created)"
    fi
done

# 3D experiments
for model in segresnet swinunetr unet resnet vitnet; do
    src="$(pwd)/runs/ablation/arch_3d_${model}/tensorboard"
    if [ -d "$src" ]; then
        ln -sfn "$src" "${TB_DIR}/3d_${model}"
        echo "  ✓ 3d_${model}"
    else
        echo "  ✗ 3d_${model} (not yet created)"
    fi
done

echo ""
echo "============================================"
echo "Phase 2 TensorBoard"
echo "Host: $(hostname) | Port: ${PORT}"
echo ""
echo "SSH tunnel:"
echo "  ssh -L ${PORT}:$(hostname):${PORT} $(hostname -f 2>/dev/null || echo 'cluster.unc.edu')"
echo "Browser: http://localhost:${PORT}"
echo "============================================"
echo ""

tensorboard \
    --logdir="${TB_DIR}" \
    --port=${PORT} \
    --bind_all \
    --reload_interval=30 \
    --samples_per_plugin="images=100,scalars=0"
