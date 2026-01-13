# CellMap Training on CU Alpine Cluster

## Quick Start

### 1. First Time Setup (Run Once)

SSH to Alpine or use [Open OnDemand](https://ondemand.rc.colorado.edu) → Clusters → Alpine Shell:
```bash
ssh your_username@login.rc.colorado.edu
```

Get a compile node and set up environment:
```bash
# Get a compile node (required for installing packages)
acompile

# Load anaconda module
module load anaconda

# Create conda environment
conda create -n cellmap python=3.11 -y
conda activate /projects/$USER/software/anaconda/envs/cellmap

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone and install the project
cd /projects/$USER
git clone https://github.com/Shan-CU/CellMap-Segmentation.git
cd CellMap-Segmentation
pip install -e .

# IMPORTANT: Downgrade cellmap-data to fix zarr compatibility issue
# See "Known Issues" section below for details
pip install cellmap-data==2025.9.9.1720

# Install tensorboard for monitoring
pip install tensorboard

# Create directories
mkdir -p logs tensorboard

# Exit compile node
exit
```

### 2. Download Training Data

The training data (~42GB) should be on `/scratch/alpine` for faster I/O:
```bash
cd /projects/$USER/CellMap-Segmentation

# Option A: Use the project's CLI (recommended)
csc fetch-data --dest /scratch/alpine/$USER/cellmap_data
ln -s /scratch/alpine/$USER/cellmap_data data

# Option B: Manual download
mkdir -p /scratch/alpine/$USER/cellmap_data
cd /scratch/alpine/$USER/cellmap_data
wget https://cellmap-challenge.janelia.org/data/matched_res_no_pad.zip
unzip matched_res_no_pad.zip
cd /projects/$USER/CellMap-Segmentation
ln -s /scratch/alpine/$USER/cellmap_data data
```

### 3. Submit Training Jobs

```bash
cd /projects/$USER/CellMap-Segmentation

# Submit UNet baseline
sbatch slurm/train_unet.sbatch

# Submit SwinTransformer baseline  
sbatch slurm/train_swin.sbatch
```

### 4. Monitor Jobs

```bash
# Check job status
squeue --user=$USER

# Watch job progress (updates every 60 seconds)
squeue --user=$USER --iterate=60

# View live output
tail -f logs/unet_JOBID.out

# Cancel a job
scancel JOBID
```

### 5. TensorBoard Monitoring

TensorBoard starts automatically with each job. Connection info is printed in the job output:

```bash
# View job output to get TensorBoard connection info
cat logs/unet_JOBID.out | grep -A5 "TENSORBOARD"
```

To connect from your local machine:
```bash
# SSH tunnel (replace NODE and PORT from job output)
ssh -L PORT:NODE:PORT your_username@login.rc.colorado.edu

# Then open in browser
http://localhost:PORT
```

**Alternative: OnDemand**
1. Go to https://ondemand.rc.colorado.edu
2. Launch "Jupyter Session" 
3. Open terminal and run: `tensorboard --logdir=/projects/$USER/CellMap-Segmentation/tensorboard`

---

## Known Issues

### Zarr Compatibility Error
If you see `GroupNotFoundError: group not found at path ''` when training:

**Cause:** Incompatibility between `cellmap-data` >= 2025.12 and zarr v2 data format.

**Fix:** Downgrade cellmap-data:
```bash
pip install cellmap-data==2025.9.9.1720
```

This is already included in the setup instructions above.

---

## SLURM Job Parameters Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--partition=aa100` | A100 GPU nodes | NVIDIA A100 GPUs |
| `--gres=gpu:1` | 1 GPU | Request single GPU |
| `--mem=32G` | 32GB RAM | CPU memory |
| `--time=24:00:00` | 24 hours | Max runtime |
| `--cpus-per-task=8` | 8 CPU cores | For data loading |

### Alternative Partitions

| Partition | GPUs | Notes |
|-----------|------|-------|
| `aa100` | NVIDIA A100 (40GB) | Best for deep learning |
| `ami100` | AMD MI100 | Requires ROCm |
| `al40` | NVIDIA L40 | Newer, good alternative |
| `atesting_a100` | A100 (MIG slice) | Quick tests, 1hr max |

---

## Troubleshooting

### Job pending for too long?
```bash
# Check estimated start time
squeue --user=$USER --start

# Try the testing partition for quick tests
sbatch --partition=atesting_a100 --qos=testing --time=01:00:00 slurm/train_quick_test.sbatch
```

### Out of memory?
Increase `--mem` or reduce batch size in training config.

### GPU not found?
Make sure you have `--gres=gpu:1` and correct partition.

### Conda not activating?
Use full path: `conda activate /projects/$USER/software/anaconda/envs/cellmap`
