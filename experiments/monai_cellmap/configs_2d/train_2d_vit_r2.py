# 2D ViT-V-Net Training Config — Round 2 (35 atomic classes)
#
# Uses the CSC framework's native train() function.
# ViTVNet2D combines a CNN encoder with a Vision Transformer for global context,
# and a V-Net style decoder with skip connections. This is our most powerful
# 2D architecture — global attention captures long-range dependencies.
#
# R2 updates: 35 classes, get_tested_classes(), AdamW, CosineAnnealingLR,
# attention_dropout=0.1, gradient clipping, force_all_classes=False.
#
# Experimental findings applied:
#   - attention_dropout_rate=0.1 (prevents attention collapse)
#   - dropout_rate=0.1 (regularization)
#   - Lower LR 5e-5 (transformer stability)
#   - img_size=256 (matches our 256×256 input patches)

import torch
from upath import UPath
from cellmap_segmentation_challenge.models import ViTVNet2D
from cellmap_segmentation_challenge.utils import get_tested_classes

# ============================================================
# Classes — all 48 tested (35 atomic + 16 groups, handled by CSC)
# ============================================================
classes = get_tested_classes()

# ============================================================
# Hyperparameters
# ============================================================
learning_rate = 5e-5  # Lower LR for transformer stability
batch_size = 4  # ViT is VRAM-hungry; b8 still OOM'd on L40S (44.4 GB usable)
gradient_accumulation_steps = 16  # Effective batch = 64 (unchanged)

input_array_info = {
    "shape": (1, 256, 256),
    "scale": (8, 8, 8),
}
target_array_info = {
    "shape": (1, 256, 256),
    "scale": (8, 8, 8),
}

# Training schedule — 200 it/ep × 2000 ep = 400k total steps
# At ~2s/it: 400k × 2 = 800k s ≈ 9.3 days (within 11-day limit)
epochs = 2000
iterations_per_epoch = 200
random_seed = 42

# ============================================================
# Model — ViTVNet2D (ViT-Base config)
# ============================================================
model_name = "2d_vit_r2"
model_to_load = "2d_vit_r2"

vit_config = {
    "img_size": 256,           # Must match input_array_info shape
    "patch_size": 16,
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "mlp_dim": 3072,
    "decoder_channels": (256, 128, 64, 16),
    "dropout_rate": 0.1,
    "attention_dropout_rate": 0.1,  # Prevents attention collapse
    "down_factor": 2,
}

model = ViTVNet2D(
    config=vit_config,
    in_channels=1,
    num_classes=len(classes),
)

load_model = "latest"

# ============================================================
# Optimizer — AdamW (essential for transformers)
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

max_grad_norm = 1.0  # CRITICAL for transformers
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
