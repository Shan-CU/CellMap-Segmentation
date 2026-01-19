#!/bin/bash
# ============================================================
# CellMap Environment Setup for CU Alpine Cluster
# Run this ONCE to set up your conda environment
# ============================================================

# Load anaconda module if available; otherwise fall back to user install.
module purge
module load anaconda 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
	if [ -f "$HOME/software/anaconda/etc/profile.d/conda.sh" ]; then
		source "$HOME/software/anaconda/etc/profile.d/conda.sh"
	elif [ -f "$HOME/software/anaconda/bin/activate" ]; then
		source "$HOME/software/anaconda/bin/activate"
	fi
fi

if ! command -v conda >/dev/null 2>&1; then
	echo "ERROR: conda not found. Run this inside a Blanca compute job, or install conda under ~/software/anaconda." >&2
	exit 2
fi

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
