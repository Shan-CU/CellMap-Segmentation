#!/bin/bash
# Setup script for Shenron (4x RTX 2080 Ti, EPYC, 62GB RAM)
# Run this once to set up the environment

set -e

echo "=============================================="
echo "Setting up CellMap environment on Shenron"
echo "=============================================="

# Check if miniforge is installed
if [ ! -d "$HOME/miniforge3" ]; then
    echo "Installing Miniforge (fast conda alternative)..."
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p $HOME/miniforge3
    rm /tmp/miniforge.sh
    
    # Initialize for bash
    $HOME/miniforge3/bin/conda init bash
    echo "Miniforge installed. Please restart your shell and run this script again."
    exit 0
fi

# Source conda
source $HOME/miniforge3/bin/activate

# Create cellmap environment if it doesn't exist
if ! conda env list | grep -q "cellmap"; then
    echo "Creating cellmap environment with Python 3.11..."
    mamba create -n cellmap python=3.11 -y
fi

# Activate environment
conda activate cellmap

echo "Installing PyTorch with CUDA 12.1 support..."
# PyTorch 2.2+ with CUDA 12.1 (compatible with CUDA 13.1 driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing cellmap-segmentation-challenge..."
cd /scratch/users/gest9386/CellMap-Segmentation
pip install -e ".[dev]"

echo "Installing additional dependencies..."
pip install tensorboard monai einops timm

echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)')
"

echo "Creating symlink to data..."
# Link the data directory
if [ ! -L "/scratch/users/gest9386/CellMap-Segmentation/data" ]; then
    ln -s /volatile/cellmap/data /scratch/users/gest9386/CellMap-Segmentation/data
    echo "Created symlink: data -> /volatile/cellmap/data"
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source ~/miniforge3/bin/activate cellmap"
echo ""
echo "To run quick test:"
echo "  cd /scratch/users/gest9386/CellMap-Segmentation/experiments/loss_optimization"
echo "  python train_local.py --mode quick_test"
echo ""
