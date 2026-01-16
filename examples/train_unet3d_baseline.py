# UNet 3D Baseline Training Config with DDP (Optimized for Alpine A100)
# Run with: torchrun --nproc_per_node=3 examples/train_unet3d_baseline.py
# Monitor with: tensorboard --logdir tensorboard
#
# 3D MODEL ADVANTAGES:
#   - Sees 3D context (organelles span multiple slices)
#   - Better for volumetric EM data
#   - More parameters but captures 3D structure
#
# DDP ADVANTAGES OVER DATAPARALLEL:
#   - One process per GPU avoids Python GIL bottleneck
#   - Efficient gradient synchronization via NCCL all-reduce
#   - Lower memory overhead (each process holds only its gradients)
#   - Better batch distribution across processes
#
# TRADE-OFF: Much higher memory usage - smaller batch sizes required

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from upath import UPath
from cellmap_segmentation_challenge.models import UNet_3D
from cellmap_segmentation_challenge.utils import get_tested_classes
from cellmap_segmentation_challenge.utils.ddp import (
    setup_ddp, cleanup_ddp, is_main_process, get_world_size, get_local_rank
)

# ============================================================
# DDP SETUP - Must be called before any CUDA operations
# ============================================================
local_rank, world_size = setup_ddp()
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

# %% Hyperparameters - Optimized for 3D training on Alpine A100
learning_rate = 1e-4  # Peak LR after warmup

# Batch size - 3D uses MUCH more memory than 2D
# A100 80GB can handle batch=4 per GPU for 32×256×256 volumes
# With DDP, each GPU gets its own batch_size (no multiplication needed)
n_gpus = world_size if world_size > 0 else 1
batch_size_per_gpu = 4  # Small due to 3D memory requirements
batch_size = batch_size_per_gpu  # Per-GPU batch size (DDP distributes across processes)
gradient_accumulation_steps = 8  # Effective batch per GPU = 4 * 8 = 32, total = 32 * n_gpus
if is_main_process():
    print(f"Using {n_gpus} GPU(s) with DDP, batch_size_per_gpu={batch_size}, effective_batch_total={batch_size * gradient_accumulation_steps * n_gpus}")

# 3D input - 32 slices of 256×256
# This captures ~256nm of depth context (32 slices × 8nm/slice)
input_array_info = {
    "shape": (32, 256, 256),  # Depth × Height × Width
    "scale": (8, 8, 8),       # 8nm isotropic voxels
}
target_array_info = {
    "shape": (32, 256, 256),
    "scale": (8, 8, 8),
}

# Training schedule - adjusted for 3D (slower iterations)
# 3D is ~10-20x slower per iteration than 2D
# With 3 A100s and batch=12, ~10-15 seconds/iteration
# 24 hours = 86,400 seconds = ~6,000-8,000 iterations possible
epochs = 200  # Fewer epochs due to slower iterations
iterations_per_epoch = 50  # Checkpoint every ~10 minutes
warmup_steps = 200  # ~4 epochs of warmup
random_seed = 42

# Use all 14 classes
classes = [
    'ecs', 'pm', 'mito_mem', 'mito_lum', 'mito_ribo',
    'golgi_mem', 'golgi_lum', 'ves_mem', 'ves_lum',
    'endo_mem', 'endo_lum', 'er_mem', 'er_lum', 'nuc'
]
if is_main_process():
    print(f"Training UNet 3D baseline with {len(classes)} classes: {classes}")

# Model - UNet 3D
model_name = "unet3d_baseline"
model_to_load = "unet3d_baseline"
_model = UNet_3D(1, len(classes))

# ============================================================
# DDP: Wrap model with DistributedDataParallel
# ============================================================
# Move model to the correct device first
_model = _model.to(device)

if world_size > 1:
    if is_main_process():
        print(f"Using DistributedDataParallel with {n_gpus} GPUs")
    model = DDP(_model, device_ids=[local_rank], output_device=local_rank)
else:
    model = _model

load_model = "latest"

# Optimizer: AdamW with weight decay
optimizer = torch.optim.AdamW(
    _model.parameters(), 
    lr=learning_rate,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# LR Scheduler: Cosine annealing with warmup
total_steps = epochs * iterations_per_epoch
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    total_steps=total_steps,
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1000,
)

# Paths
logs_save_path = UPath("tensorboard/{model_name}").path
model_save_path = UPath("checkpoints/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit.csv"

# Data augmentation - 3D compatible
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.5}},  # Also flip in z
    "transpose": {"axes": ["x", "y"]},  # Only transpose x/y
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

# Training optimizations
max_grad_norm = 1.0
validation_time_limit = 60  # Longer validation for 3D
validation_batch_limit = 5  # Fewer batches due to 3D size
filter_by_scale = True

# Dataset handling
force_all_classes = False

# Dataloader - fewer workers for 3D (more memory per sample)
n_workers = 8  # Reduced from 16 due to 3D memory
if is_main_process():
    print(f"Using {n_workers} dataloader workers per process")

dataloader_kwargs = {
    "num_workers": n_workers,
    "pin_memory": False,
    "persistent_workers": True if n_workers > 0 else False,
    "multiprocessing_context": "spawn" if n_workers > 0 else None,
}

# Mixed precision - critical for 3D to fit in memory
use_mixed_precision = True

# ============================================================
# DDP-SPECIFIC SETTINGS
# ============================================================
# These are used by the train function to enable DDP-specific behavior
use_ddp = world_size > 1  # Enable DDP mode in training
ddp_local_rank = local_rank  # GPU index for this process
ddp_world_size = world_size  # Total number of processes

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    try:
        train(__file__)
    finally:
        # Clean up DDP process group
        cleanup_ddp()
