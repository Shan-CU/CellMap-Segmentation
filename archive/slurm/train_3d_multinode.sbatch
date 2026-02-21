#!/bin/bash
#SBATCH --job-name=3d_multinode
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/logs/%x_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/logs/%x_%j.err
#SBATCH --partition=h100_mn
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=1400G
#SBATCH --time=5-00:00:00

# ============================================================================
# MULTI-NODE MONSTER 3D TRAINING - ALL 8 H100s
# ============================================================================
# This job uses BOTH Sycamore H100 nodes via InfiniBand:
#
# Total resources:
#   - 8x NVIDIA H100 80GB (640GB GPU memory total)
#   - 512 CPU cores (256 per node)
#   - 3TB System RAM (1.5TB per node)
#   - NDR InfiniBand 2x 800Gbps interconnect
#
# NOTE: Requires special QoS access for h100_mn multi-node!
# Contact UNC Research Computing helpdesk to request access.
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL="REPLACE_WITH_BEST_3D_MODEL"
BATCH_SIZE=4
NUM_WORKERS=16
PREFETCH=4
LEARNING_RATE=3e-5
EPOCHS=150

# ============================================================================
# MULTI-NODE SETUP
# ============================================================================

echo "============================================"
echo "MULTI-NODE MONSTER TRAINING"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Total GPUs: $((SLURM_NNODES * 4))"
echo "============================================"

# Get master node address for distributed training
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

echo "Master node: $MASTER_ADDR:$MASTER_PORT"

# Initialize environment
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# ============================================================================
# MULTI-NODE TRAINING
# ============================================================================

# Use srun to launch on all nodes
# torchrun handles the multi-GPU aspect within each node
# srun handles the multi-node aspect

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -c '
    # Each node runs torchrun with 4 GPUs
    # SLURM_NODEID identifies which node we are on (0 or 1)
    
    export MAMBA_ROOT_PREFIX="$HOME/micromamba"
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate csc
    
    cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
    
    # Calculate node rank
    NODE_RANK=$SLURM_NODEID
    WORLD_SIZE=$((SLURM_NNODES * 4))
    
    echo "Node $NODE_RANK: Starting training with 4 GPUs"
    
    torchrun \
        --nnodes='$SLURM_NNODES' \
        --node_rank=$NODE_RANK \
        --nproc_per_node=4 \
        --master_addr='$MASTER_ADDR' \
        --master_port='$MASTER_PORT' \
        experiments/model_comparison/train_comparison.py \
        --model '$MODEL' \
        --dim 3D \
        --epochs '$EPOCHS' \
        --batch_size '$BATCH_SIZE' \
        --lr '$LEARNING_RATE' \
        --num_workers '$NUM_WORKERS' \
        --prefetch_factor '$PREFETCH' \
        --pin_memory \
        --amp
'

echo "============================================"
echo "Multi-node training complete!"
echo "============================================"
