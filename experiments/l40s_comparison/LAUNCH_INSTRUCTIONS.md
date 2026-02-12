# Launch Instructions for Copilot Agent on Sycamore

**Target**: UNC Longleaf cluster (`longleaf.unc.edu`)  
**User**: `gsgeorge`  
**Branch**: `feature/l40s-model-comparison`  
**Task**: Submit 8 L40S GPU training jobs for CellMap segmentation model comparison

---

## Prerequisites

You can SSH to Longleaf from Sycamore:
```bash
ssh longleaf.unc.edu
```

---

## Step-by-Step Launch Procedure

### 1. SSH into Longleaf and navigate to the repo

```bash
ssh longleaf.unc.edu
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
```

### 2. Pull the latest code and switch to the correct branch

```bash
git fetch origin
git checkout feature/l40s-model-comparison
git pull origin feature/l40s-model-comparison
```

### 3. Verify the experiment files exist

```bash
ls experiments/l40s_comparison/
# Should show: config.py  losses.py  train.py  README.md  slurm/  ...

ls experiments/l40s_comparison/slurm/
# Should show: launch_all.sh  train_unet_2d.sbatch  train_resnet_2d.sbatch  ...  (8 .sbatch files)
```

### 4. Ensure the conda/micromamba environment is available

```bash
/nas/longleaf/home/gsgeorge/.local/bin/micromamba run -n csc python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
```

Expected output: `PyTorch 2.x.x, CUDA True`

### 5. Create log and output directories

```bash
mkdir -p experiments/l40s_comparison/{checkpoints,tensorboard,visualizations,metrics,results,slurm/logs}
```

### 6. Submit all 8 jobs at once

```bash
bash experiments/l40s_comparison/slurm/launch_all.sh
```

This submits 8 SLURM jobs to the `l40-gpu` partition requesting L40S GPUs:

| Job | Script | GPUs | Batch | Eff. Batch |
|-----|--------|------|-------|------------|
| UNet 2D | `train_unet_2d.sbatch` | 2 | 26 | 208 |
| ResNet 2D | `train_resnet_2d.sbatch` | 2 | 28 | 224 |
| Swin 2D | `train_swin_2d.sbatch` | 2 | 18 | 144 |
| ViT 2D | `train_vit_2d.sbatch` | 2 | 10 | 80 |
| UNet 3D | `train_unet_3d.sbatch` | 2 | 5 | 40 |
| ResNet 3D | `train_resnet_3d.sbatch` | 2 | 5 | 40 |
| Swin 3D | `train_swin_3d.sbatch` | **4** | 2 | 48 |
| ViT 3D | `train_vit_3d.sbatch` | 2 | 1 | 16 |

**Alternatively**, submit a subset:
```bash
# Only 2D models
bash experiments/l40s_comparison/slurm/launch_all.sh 2d

# Only 3D models
bash experiments/l40s_comparison/slurm/launch_all.sh 3d

# Only a specific architecture (both 2D and 3D)
bash experiments/l40s_comparison/slurm/launch_all.sh unet
bash experiments/l40s_comparison/slurm/launch_all.sh swin

# Submit a single job directly
sbatch experiments/l40s_comparison/slurm/train_swin_3d.sbatch
```

### 7. Verify jobs are queued/running

```bash
squeue -u gsgeorge
```

You should see up to 8 jobs with names like `unet_2d_l40`, `swin_3d_l40`, etc.  
All should be on partition `l40-gpu`.

---

## SLURM Details (all jobs share these settings)

| Setting | Value | Why |
|---------|-------|-----|
| `--partition` | `l40-gpu` | L40S 48GB GPUs |
| `--qos` | `gpu_access` | Required for GPU partitions |
| `--mem` | `512g` | **Critical** — 200g causes OOM during 14-class label expansion |
| `--time` | `11-00:00:00` | 11-day max wall time (L40S partition limit) |
| `--cpus-per-task` | `32` | Sufficient for data loading |
| `--mail-user` | `gsgeorge@ad.unc.edu` | Email on BEGIN/END/FAIL |

The `module load cuda/12.2` and micromamba environment activation are handled inside each sbatch script.

---

## Monitoring

### Check job status
```bash
squeue -u gsgeorge --format="%.10i %.20j %.8T %.10M %.6D %R"
```

### Watch a specific job's output in real-time
```bash
# Replace <JOBID> with the actual SLURM job ID
tail -f experiments/l40s_comparison/slurm/logs/unet_2d_<JOBID>.out
```

### Check GPU utilization on a running node
```bash
# Find which node a job is running on
squeue -u gsgeorge -o "%.10i %.20j %.20R"
# Then SSH to that node
ssh <nodename> nvidia-smi
```

### TensorBoard (from a login node with port forwarding)
```bash
# On Longleaf login node:
tensorboard --logdir /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/l40s_comparison/tensorboard --port 6007 --bind_all

# Or via SSH tunnel from Sycamore:
ssh -L 6007:localhost:6007 longleaf.unc.edu "cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation && tensorboard --logdir experiments/l40s_comparison/tensorboard --port 6007"
```

---

## Troubleshooting

### Job stuck in PENDING
```bash
# Check why it's pending
squeue -u gsgeorge -o "%.10i %.20j %.8T %.20R"
# If REASON is "Resources", L40S nodes are full — just wait
# If REASON is "QOSMaxGRESPerUser", you've hit the GPU limit — cancel some jobs
```

### OOM (Out of Memory) on GPU
If a job crashes with CUDA OOM, reduce the batch size in the `.sbatch` file:
```bash
# Edit the specific sbatch file, e.g.:
vi experiments/l40s_comparison/slurm/train_swin_3d.sbatch
# Reduce BATCH_SIZE by ~25% and resubmit
sbatch experiments/l40s_comparison/slurm/train_swin_3d.sbatch
```

### OOM on RAM (host memory)
The `--mem=512g` should be sufficient. If it's not, the error will say `oom-kill`.
This was validated with MONAI Auto3DSeg Job 30343731 which peaked at ~222GB.

### Cancel all jobs
```bash
scancel -u gsgeorge --partition=l40-gpu
```

### Cancel a specific job
```bash
scancel <JOBID>
```

---

## What These Jobs Do

Each job trains one of 8 segmentation models (UNet/ResNet/SwinTransformer/ViT-V-Net in 2D and 3D) on the CellMap FIB-SEM 14-class organelle segmentation task using:

- **Loss**: `BalancedSoftmaxPartialTverskyLoss` — the winning configuration from 3 prior experiments (τ=1.0, α=0.6, β=0.4)
- **Partial annotations**: Each crop only labels a subset of the 14 classes; unannotated channels are masked from loss via NaN detection
- **AMP**: BFloat16 mixed precision on L40S (Ada Lovelace architecture)
- **DDP**: Multi-GPU distributed training via torchrun
- **Early stopping**: Training stops if validation Dice doesn't improve for 20 epochs
- **Checkpoints**: Best model (by val Dice) + periodic saves every 25 epochs

Results will appear in:
- `experiments/l40s_comparison/checkpoints/` — model weights
- `experiments/l40s_comparison/tensorboard/` — training curves
- `experiments/l40s_comparison/visualizations/` — input/GT/prediction figures
- `experiments/l40s_comparison/metrics/` — per-class Dice scores
