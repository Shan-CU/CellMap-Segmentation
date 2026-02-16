# 2D UNet Training Config — Optimized for L40S Single GPU (5.5-day run)
#
# Uses the CSC framework's native train() function.
# Improvements over examples/train_2D.py:
#   - All 14 tested classes (not just 2)
#   - AdamW optimizer with weight decay
#   - OneCycleLR cosine scheduler with warmup
#   - Gradient clipping for stability
#   - InstanceNorm + Dropout for better generalization
#   - Larger effective batch via gradient accumulation
#   - force_all_classes=False (critical for partial annotations)
#   - Longer training (5.5 days on single L40S)

import torch
from upath import UPath
from cellmap_segmentation_challenge.models import UNet_2D

# ============================================================
# Hyperparameters
# ============================================================
learning_rate = 1e-4  # Peak LR (OneCycleLR will ramp up then decay)
batch_size = 32  # L40S 48GB can easily handle 32×(1,256,256) for 2D UNet
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
# UNet 2D is fast: ~2-3 s/iteration on L40S with batch=32
# 5.5 days = 475,200 s → ~160k-240k iterations possible
# Use 1000 epochs × 200 iterations = 200,000 total steps
epochs = 1000
iterations_per_epoch = 200
random_seed = 42

# All 14 classes from the challenge
classes = [
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "er_mem", "er_lum", "nuc",
]

# ============================================================
# Model — 2D UNet with InstanceNorm + Dropout
# ============================================================
model_name = "2d_unet_l40s"
model_to_load = "2d_unet_l40s"
model = UNet_2D(
    n_channels=1,
    n_classes=len(classes),
    trilinear=False,
    use_instancenorm=True,  # Better than BatchNorm for segmentation
    dropout=0.1,            # Mild regularization
)

load_model = "latest"  # Resume from latest checkpoint if exists

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
# LR Scheduler — CosineAnnealingLR (stepped per-epoch by CSC framework)
# ============================================================
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,          # Full cosine cycle over all epochs
    eta_min=learning_rate / 1000,  # Minimum LR
)

# ============================================================
# Paths — save inside experiments/monai_cellmap/
# ============================================================
logs_save_path = UPath("experiments/monai_cellmap/tensorboard_2d/{model_name}").path
model_save_path = UPath("experiments/monai_cellmap/checkpoints_2d/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit.csv"

# ============================================================
# Data & Training Settings
# ============================================================
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

max_grad_norm = 1.0         # Gradient clipping
validation_time_limit = 60  # 60s max per validation
validation_batch_limit = 20 # At most 20 val batches
filter_by_scale = True      # Only use data at matching resolution
force_all_classes = False   # Allow partial annotations
device = "cuda"

# Dataloader config (single GPU, no DDP)
dataloader_kwargs = {
    "num_workers": 2,
    "pin_memory": True,
}

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)
