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

# Pin cellmap-data version (2025.12.24 has EmptyImage bug, 2025.12.12 works)
pip install cellmap-data==2025.12.12.1737

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

## Alpine A100 Hardware Optimizations

Based on [CU Research Computing Documentation](https://github.com/ResearchComputing/Documentation):

### Hardware Specs (aa100 partition)
- **3x NVIDIA A100 80GB GPUs** per node
- **64 CPU cores** (AMD Milan)
- **~243GB RAM** (3.8GB/core)
- **~300GB local SSD** (`$SLURM_SCRATCH`)
- **2x25 Gb Ethernet + RoCE** network

### Optimizations Applied

| Optimization | Before | After | Impact |
|--------------|--------|-------|--------|
| GPUs | 1 | 3 | ~3x throughput with DataParallel |
| CPU cores | 8 | 16 | Faster data loading |
| Memory | 32GB | 128GB | Larger batches possible |
| Batch size | 8 | 16×n_gpus | Better GPU utilization |
| Data loading | Default | pin_memory, persistent_workers | Faster CPU→GPU transfer |
| Local scratch | Not used | Available | ~10x faster I/O |

### Data I/O Best Practices

1. **Fastest**: Copy data to `$SLURM_SCRATCH` (local SSD, ~300GB)
2. **Fast**: Use `/scratch/alpine/$USER` (10TB, network)
3. **Avoid**: `/projects` or `/home` for training I/O

```bash
# Stage data to local scratch at job start (optional, for max speed)
rsync -a /scratch/alpine/$USER/cellmap_data $SLURM_SCRATCH/
export CELLMAP_DATA_DIR=$SLURM_SCRATCH/cellmap_data
```

---

## SLURM Job Parameters Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--partition=aa100` | A100 GPU nodes | NVIDIA A100 80GB GPUs |
| `--gres=gpu:3` | 3 GPUs | All 3 A100s on the node |
| `--mem=128G` | 128GB RAM | ~Half node memory |
| `--time=24:00:00` | 24 hours | Max runtime (normal QoS) |
| `--cpus-per-task=16` | 16 CPU cores | For parallel data loading |
| `--qos=long` | 7 days | Use for longer runs |

### Alternative Partitions

| Partition | GPUs | VRAM | Notes |
|-----------|------|------|-------|
| `aa100` | 3x NVIDIA A100 | 80GB | Best for deep learning |
| `ami100` | 3x AMD MI100 | 32GB | Requires ROCm |
| `al40` | 3x NVIDIA L40 | 48GB | Good alternative |
| `atesting_a100` | 1x A100 (MIG) | 20GB | Quick testing (1hr limit) |
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
