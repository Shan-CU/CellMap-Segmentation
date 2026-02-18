# 2D UNet Training Config — Round 2 (35 atomic classes)
#
# Uses the CSC framework's native train() function.
# R2 updates over V1 (train_2d_unet.py):
#   - 14 → 35 atomic classes (covers all 48 tested classes via group composition)
#   - get_tested_classes() for automatic class list
#   - AdamW optimizer with weight decay
#   - CosineAnnealingLR scheduler
#   - InstanceNorm + Dropout=0.1 for regularization
#   - Gradient clipping (max_grad_norm=1.0)
#   - force_all_classes=False (critical for partial annotations)
#   - filter_by_scale=True
#   - 1× L40S GPU (single GPU, no DDP)

import torch
from upath import UPath
from cellmap_segmentation_challenge.models import UNet_2D
from cellmap_segmentation_challenge.utils import get_tested_classes

# ============================================================
# Classes — all 48 tested (35 atomic + 16 groups, handled by CSC)
# ============================================================
classes = get_tested_classes()

# ============================================================
# Hyperparameters
# ============================================================
learning_rate = 1e-4
batch_size = 32  # L40S 48GB handles 32×(1,256,256) easily for UNet
gradient_accumulation_steps = 2  # Effective batch = 64

input_array_info = {
    "shape": (1, 256, 256),
    "scale": (8, 8, 8),
}
target_array_info = {
    "shape": (1, 256, 256),
    "scale": (8, 8, 8),
}

# Training schedule
# UNet 2D: ~2-3 s/iteration on L40S with batch=32
# 250 it/ep × 2000 ep = 500k total steps
# At ~2.5s/it: 500k × 2.5 = 1.25M s ≈ 14.5 days (within 11-day limit with margin)
epochs = 2000
iterations_per_epoch = 250
random_seed = 42

# ============================================================
# Model — 2D UNet with InstanceNorm + Dropout
# ============================================================
model_name = "2d_unet_r2"
model_to_load = "2d_unet_r2"
model = UNet_2D(
    n_channels=1,
    n_classes=len(classes),
    trilinear=False,
    use_instancenorm=True,
    dropout=0.1,
)

load_model = "latest"

# ============================================================
# Optimizer — AdamW with weight decay
# ============================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)

# ============================================================
# LR Scheduler — CosineAnnealingLR
# ============================================================
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=learning_rate / 1000,
)

# ============================================================
# Paths
# ============================================================
logs_save_path = UPath("experiments/monai_cellmap/tensorboard_2d/{model_name}").path
model_save_path = UPath("experiments/monai_cellmap/checkpoints_2d/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit_r2.csv"  # New CSV for 48 classes (old datasplit.csv has only 14)

# ============================================================
# Data & Training Settings
# ============================================================
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

max_grad_norm = 1.0
validation_time_limit = 60
validation_batch_limit = 20
filter_by_scale = True
force_all_classes = False
device = "cuda"

dataloader_kwargs = {
    "num_workers": 2,
    "pin_memory": True,
}

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)
