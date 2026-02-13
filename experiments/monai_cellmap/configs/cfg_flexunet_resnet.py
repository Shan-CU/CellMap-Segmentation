"""
FlexibleUNet + ResNet34 encoder config (CryoET 1st-place winner style).

FlexibleUNet allows plugging in any torchvision backbone as the encoder.
The CryoET winner used ResNet34 with aggressive Mixup.

Reference: IMPLEMENTATION_SPEC.md §5.4
"""

from copy import deepcopy

from common_config import basic_cfg

cfg = deepcopy(basic_cfg)
cfg.name = "flexunet_resnet34"
cfg.output_dir = f"/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/{cfg.name}"

# --- Model ---
cfg.model = "mdl_cellmap"
cfg.backbone_type = "flexunet"
cfg.backbone_args = dict(
    spatial_dims=3,
    in_channels=cfg.in_channels,
    out_channels=cfg.num_classes,
    backbone="resnet34",
    pretrained=False,  # no ImageNet pretrained for 3D
)
cfg.deep_supervision = False
cfg.multi_scale_heads = True
cfg.lvl_weights = [0, 0, 0, 1]  # only final scale

# --- Mixup: CryoET winner used this aggressively ---
cfg.mixup_p = 1.0
cfg.mixup_beta = 1.0

# --- Patches ---
cfg.roi_size = [96, 96, 96]
cfg.num_samples = 4
cfg.batch_size = 4  # smaller patches → can fit more per batch

# --- Training ---
cfg.lr = 1e-3
cfg.optimizer = "Adam"
cfg.weight_decay = 0.0
cfg.epochs = 600
cfg.eval_epochs = 5
