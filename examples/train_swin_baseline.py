# SwinTransformer Baseline Training Config with DDP (Optimized for Alpine A100)
# Run with: torchrun --nproc_per_node=3 examples/train_swin_baseline.py
# Monitor with: tensorboard --logdir tensorboard
#
# ALPINE A100 OPTIMIZATIONS:
#   - 3x A100 80GB GPUs with DistributedDataParallel (DDP)
#   - DDP is preferred over DataParallel for better scaling
#   - 16 CPU workers for parallel data loading per process
#   - Mixed precision (AMP) for TensorCore acceleration
#   - pin_memory + persistent_workers for fast data transfer
#
# DDP ADVANTAGES OVER DATAPARALLEL:
#   - One process per GPU avoids Python GIL bottleneck
#   - Efficient gradient synchronization via NCCL all-reduce
#   - Lower memory overhead (each process holds only its gradients)
#   - Better batch distribution across processes
#
# Based on Lauenburg & eminorhan winning configs:
#   - Frequent validation for better learning signal
#   - LR warmup + cosine decay (CRITICAL for transformers)
#   - Gradient clipping for stability
#   - AdamW optimizer (essential for transformers)

import multiprocessing
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from upath import UPath
from cellmap_segmentation_challenge.models import SwinTransformer
from cellmap_segmentation_challenge.utils import get_tested_classes
from cellmap_segmentation_challenge.utils.ddp import (
    setup_ddp, cleanup_ddp, is_main_process, get_world_size, get_local_rank
)

# ============================================================
# DDP SETUP - Must be called before any CUDA operations
# Skip DDP setup in DataLoader worker processes (spawned by multiprocessing)
# ============================================================
_is_worker = multiprocessing.parent_process() is not None

if not _is_worker:
    local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
else:
    # DataLoader worker - just read env vars, don't initialize DDP
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cpu")  # Workers don't need GPU

# %% Hyperparameters - Optimized for 24-HOUR transformer training on Alpine A100
learning_rate = 1e-4  # Peak LR after warmup

# Batch size optimized for multi-GPU A100 (80GB VRAM each)
# Transformers use more memory than UNets, but A100 80GB is generous
# With DDP, each GPU gets its own batch_size (no multiplication needed)
n_gpus = world_size if world_size > 0 else 1
batch_size_per_gpu = 16  # A100 80GB can handle 16 for SwinTransformer 256x256
batch_size = batch_size_per_gpu  # Per-GPU batch size (DDP distributes across processes)
gradient_accumulation_steps = 4  # Effective batch per GPU = 16 * 4 = 64, total = 64 * n_gpus
if is_main_process():
    print(f"Using {n_gpus} GPU(s) with DDP, batch_size_per_gpu={batch_size}, effective_batch_total={batch_size * gradient_accumulation_steps * n_gpus}")

input_array_info = {
    "shape": (1, 256, 256),  # Must be divisible by patch_size and window_size
    "scale": (8, 8, 8),
}
target_array_info = {
    "shape": (1, 256, 256),
    "scale": (8, 8, 8),
}

# Training schedule - OPTIMIZED FOR 24-HOUR RUN
# Transformers train slower than UNets, so same epochs but may complete fewer
# With 3 A100s and batch=48, ~3 seconds/iteration (transformers are slower)
# 24 hours = 86,400 seconds = ~28,000 iterations possible
# We use 400 epochs × 100 iterations = 40,000 steps (accounting for overhead)
epochs = 400  # Slightly fewer than UNet due to slower per-iteration time
iterations_per_epoch = 100  # Frequent validation/checkpoints
warmup_steps = 1000  # ~10 epochs warmup (CRITICAL for transformers!)
random_seed = 42

# Use common classes that have good data coverage
classes = [
    'ecs', 'pm', 'mito_mem', 'mito_lum', 'mito_ribo',
    'golgi_mem', 'golgi_lum', 'ves_mem', 'ves_lum',
    'endo_mem', 'endo_lum', 'er_mem', 'er_lum', 'nuc'
]
if is_main_process():
    print(f"Training SwinTransformer baseline with {len(classes)} classes: {classes}")

# Model - SwinTransformer
model_name = "swin_baseline"
model_to_load = "swin_baseline"
_model = SwinTransformer(
    patch_size=[4, 4],
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=[7, 7],
    num_classes=len(classes)
)

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

# Optimizer: AdamW with weight decay (essential for transformers)
# Note: Use _model.parameters() to get actual model params (not DDP wrapper)
optimizer = torch.optim.AdamW(
    _model.parameters(), 
    lr=learning_rate,
    weight_decay=1e-4,  # Regularization
    betas=(0.9, 0.999)
)

# LR Scheduler: Cosine annealing with warmup (critical for transformers)
total_steps = epochs * iterations_per_epoch
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    total_steps=total_steps,
    pct_start=0.05,  # 5% warmup
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1000,
)

# Paths
logs_save_path = UPath("tensorboard/{model_name}").path
model_save_path = UPath("checkpoints/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit.csv"

# Data augmentation
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

# Training optimizations
max_grad_norm = 1.0  # Gradient clipping - CRITICAL for transformers
validation_time_limit = 30
validation_batch_limit = 10
filter_by_scale = True

# ============================================================
# DATASET HANDLING (from Lauenburg's winning solution)
# ============================================================
# force_all_classes=False allows training on datasets that only have SOME of
# the requested classes. This is critical because many crops don't have all
# organelle types. With False, a crop is included if it has at least one class.
force_all_classes = False

# ============================================================
# ALPINE A100 DATALOADER OPTIMIZATIONS
# ============================================================
# Use 16 CPU cores for parallel data loading (set in SLURM script)
# pin_memory=False because cellmap-data moves data to GPU internally
# persistent_workers=True keeps workers alive between epochs (reduces overhead)
# multiprocessing_context="spawn" is REQUIRED for CUDA - fork doesn't work!
n_workers = 16  # Hardcoded for Alpine (SLURM sets 32 CPUs, use half for dataloading)
if is_main_process():
    print(f"Using {n_workers} dataloader workers per process")

dataloader_kwargs = {
    "num_workers": n_workers,
    "pin_memory": False,  # cellmap-data moves data to GPU, can't pin CUDA tensors
    "persistent_workers": True if n_workers > 0 else False,
    # Don't use spawn with DDP - it causes workers to re-init DDP
    # Fork is the default and works fine with DDP since each rank is a separate process
}

# ============================================================
# MIXED PRECISION (AMP) - A100 TensorCores are optimized for BF16
# BF16 is preferred for transformers (better numerical stability than FP16)
# ============================================================
use_mixed_precision = True  # Enable AMP for ~2x speedup on A100

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
