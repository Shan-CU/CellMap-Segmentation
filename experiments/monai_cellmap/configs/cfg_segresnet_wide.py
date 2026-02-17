"""
SegResNet-Wide (48 init_filters) config.

Wider variant of SegResNet with 48 initial filters (vs 32 in the standard
config). This provides ~2.25× more feature channels throughout the network,
capturing richer representations at the cost of more VRAM.

Trained on H100 80GB GPUs for maximum patch size. Uses 192³ crops
(up from 128³ on L40S 48GB) for richer spatial context.

Purpose: Ensemble diversity — same architecture family but architecturally
distinct due to wider feature maps. Makes different errors than the narrow
variant, improving ensemble quality.

Reference: IMPLEMENTATION_SPEC.md §5.2
"""

from copy import deepcopy

from common_config import basic_cfg

cfg = deepcopy(basic_cfg)
cfg.name = "segresnet_wide_r2"
cfg.output_dir = f"/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/{cfg.name}"

# --- Model ---
cfg.model = "mdl_cellmap"
cfg.backbone_type = "segresnet"
cfg.backbone_args = dict(
    spatial_dims=3,
    in_channels=cfg.in_channels,
    out_channels=cfg.num_classes,
    init_filters=48,                  # WIDE: 48 vs 32 standard
    blocks_down=(1, 2, 2, 4, 4),
    norm="INSTANCE",
)
cfg.deep_supervision = True
cfg.ds_weights = [1.0, 0.5, 0.25, 0.125]

# --- Patches: 192³ b2 on H100 80GB — est. ~65 GB / 80 GB (15 GB headroom).
#     128³ was for L40S 48GB. H100 allows 3.375× more voxels per patch. ---
cfg.roi_size = [192, 192, 192]
cfg.num_samples = 4
cfg.batch_size = 2

# --- Training: same schedule as standard SegResNet ---
cfg.lr = 2e-4
cfg.epochs = 300
cfg.eval_epochs = 5
