"""
SegResNet with deep supervision config — L40S variant.

Same model as cfg_segresnet (32f), but with 128³ patches to fit L40S 48GB VRAM.
160³ is the H100 setting; 144³ OOM'd on L40S; 128³ is safe (~32 GB).

Moved from H100 to L40S to free GPUs for SegResNet-Wide (48f).
"""

from copy import deepcopy

from common_config import basic_cfg

cfg = deepcopy(basic_cfg)
cfg.name = "segresnet_ds_r3_l40s"
cfg.output_dir = f"/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/{cfg.name}"

# --- Model (identical to cfg_segresnet) ---
cfg.model = "mdl_cellmap"
cfg.backbone_type = "segresnet"
cfg.backbone_args = dict(
    spatial_dims=3,
    in_channels=cfg.in_channels,
    out_channels=cfg.num_classes,
    init_filters=32,
    blocks_down=(1, 2, 2, 4, 4),
    norm="INSTANCE",
)
cfg.deep_supervision = True
cfg.ds_weights = [1.0, 0.5, 0.25, 0.125]

# --- Patches: 128³ b2 on L40S 48GB — ~32 GB safe.
#     144³ OOM'd at ~45 GB on L40S. 160³ is H100-only. ---
cfg.roi_size = [128, 128, 128]
cfg.num_samples = 4
cfg.batch_size = 2

# --- Training (same schedule as H100 version) ---
cfg.lr = 2e-4
cfg.epochs = 600
cfg.eval_epochs = 5
