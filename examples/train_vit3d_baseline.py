# ViT-V-Net 3D Baseline Training Config with DDP (Optimized for Alpine A100)
# Run with: torchrun --nproc_per_node=3 examples/train_vit3d_baseline.py
# Monitor with: tensorboard --logdir tensorboard
#
# 3D VISION TRANSFORMER ADVANTAGES:
#   - Global attention captures long-range dependencies
#   - Better at understanding 3D spatial relationships
#   - State-of-the-art for volumetric medical imaging
#
# DDP ADVANTAGES OVER DATAPARALLEL:
#   - One process per GPU avoids Python GIL bottleneck
#   - Efficient gradient synchronization via NCCL all-reduce
#   - Lower memory overhead (each process holds only its gradients)
#
# TRADE-OFF: Transformers are memory-hungry - small batches required

import multiprocessing
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from upath import UPath
from cellmap_segmentation_challenge.models import ViTVNet
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

# %% Hyperparameters - Optimized for 3D ViT training on Alpine A100
learning_rate = 1e-4  # Peak LR after warmup

# Batch size - 3D ViT uses MUCH more memory than 3D UNet
# A100 80GB can handle batch=2-3 per GPU for 128×128×128 volumes
# With DDP, each GPU gets its own batch_size
n_gpus = world_size if world_size > 0 else 1
batch_size_per_gpu = 2  # Very small due to ViT memory requirements
batch_size = batch_size_per_gpu  # Per-GPU batch size
gradient_accumulation_steps = 16  # Effective batch per GPU = 2 * 16 = 32, total = 32 * n_gpus
if is_main_process():
    print(f"Using {n_gpus} GPU(s) with DDP, batch_size_per_gpu={batch_size}, effective_batch_total={batch_size * gradient_accumulation_steps * n_gpus}")

# 3D input - ViTVNet expects isotropic cubes, 128^3 is standard
# This captures full 3D context at 8nm resolution = ~1µm cube
input_array_info = {
    "shape": (128, 128, 128),  # Isotropic cube
    "scale": (8, 8, 8),        # 8nm isotropic voxels
}
target_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}

# Training schedule - adjusted for 3D ViT (very slow iterations)
# 3D ViT is ~20-50x slower per iteration than 2D
# With 3 A100s and batch=6, ~20-30 seconds/iteration
# 24 hours = 86,400 seconds = ~3,000-4,000 iterations possible
epochs = 100  # Fewer epochs due to very slow iterations
iterations_per_epoch = 40  # Checkpoint every ~15-20 minutes
warmup_steps = 100  # ~2.5 epochs of warmup
random_seed = 42

# Use all 14 classes
classes = [
    'ecs', 'pm', 'mito_mem', 'mito_lum', 'mito_ribo',
    'golgi_mem', 'golgi_lum', 'ves_mem', 'ves_lum',
    'endo_mem', 'endo_lum', 'er_mem', 'er_lum', 'nuc'
]
if is_main_process():
    print(f"Training ViT-V-Net 3D baseline with {len(classes)} classes: {classes}")

# Model - ViT-V-Net 3D
model_name = "vit3d_baseline"
model_to_load = "vit3d_baseline"
_model = ViTVNet(
    out_channels=len(classes),
    config="ViT-V-Net",
    img_size=input_array_info["shape"]
)

# ============================================================
# DDP: Wrap model with DistributedDataParallel
# ============================================================
# Move model to the correct device first
_model = _model.to(device)

if world_size > 1:
    if is_main_process():
        print(f"Using DistributedDataParallel with {n_gpus} GPUs")
    # find_unused_parameters=True for ViT (attention weights may not all be used)
    model = DDP(_model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)
else:
    model = _model

load_model = "latest"

# Optimizer: AdamW with weight decay (essential for transformers)
optimizer = torch.optim.AdamW(
    _model.parameters(), 
    lr=learning_rate,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# LR Scheduler: Cosine annealing with warmup (critical for transformers)
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
max_grad_norm = 1.0  # Gradient clipping - CRITICAL for transformers
validation_time_limit = 60  # Longer validation for 3D
validation_batch_limit = 3  # Very few batches due to 3D size
filter_by_scale = True

# Dataset handling
force_all_classes = False

# ============================================================
# DATALOADER CONFIGURATION
# ============================================================
n_workers = 0
if is_main_process():
    print(f"Using {n_workers} dataloader workers (DDP compatibility mode)")

dataloader_kwargs = {
    "num_workers": n_workers,
    "pin_memory": False,
}

# Mixed precision - CRITICAL for 3D ViT to fit in memory
use_mixed_precision = True

# ============================================================
# DDP-SPECIFIC SETTINGS
# ============================================================
use_ddp = world_size > 1
ddp_local_rank = local_rank
ddp_world_size = world_size

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    try:
        train(__file__)
    finally:
        cleanup_ddp()
