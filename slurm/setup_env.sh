#!/bin/bash
# ============================================================
# CellMap Environment Setup for CU Alpine Cluster
# Run this ONCE to set up your conda environment
# ============================================================

# Load anaconda module
module purge
module load anaconda

# Create environment in /projects (more storage than /home)
echo "Creating cellmap environment..."
conda create -n cellmap python=3.11 -y

# Activate environment
conda activate /projects/$USER/software/anaconda/envs/cellmap

# Install PyTorch with CUDA support (A100 compatible)
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
echo "Installing CellMap dependencies..."
pip install -e .

# Downgrade cellmap-data to compatible version (if needed)
pip install cellmap-data==2025.9.9.1720

# Install additional utilities
pip install tensorboard

echo "=========================================="
echo "Setup complete!"
echo "To activate: conda activate /projects/$USER/software/anaconda/envs/cellmap"
echo "=========================================="
