"""
SegResNet with deep supervision config.

SegResNetDS is MONAI's SegResNet variant with built-in deep supervision
outputs at multiple decoder scales.

Reference: IMPLEMENTATION_SPEC.md §5.2
"""

from copy import deepcopy

from common_config import basic_cfg

cfg = deepcopy(basic_cfg)
cfg.name = "segresnet_ds_r2"
cfg.output_dir = f"/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/{cfg.name}"

# --- Model ---
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

# --- Patches: 160³ b2 on H100 80GB — est. ~45 GB / 80 GB (35 GB headroom).
#     128³ was safe on L40S 48GB (~32 GB). 144³ OOM'd at 45 GB on L40S.
#     160³ gives 1.95× more voxels per patch than 128³. ---
cfg.roi_size = [160, 160, 160]
cfg.num_samples = 4
cfg.batch_size = 2

# --- Training (Round 2: 600 epochs — R1 best Dice was at ep519/600;
#     keep 600 to ensure full convergence. ~41h on H100, well within 120h limit) ---
cfg.lr = 2e-4
cfg.epochs = 600
cfg.eval_epochs = 5
