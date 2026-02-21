"""
FlexibleUNet + ResNet34 encoder config (CryoET 1st-place winner style).

FlexibleUNet allows plugging in any torchvision backbone as the encoder.
The CryoET winner used ResNet34 with aggressive Mixup.

Reference: IMPLEMENTATION_SPEC.md §5.4
"""

from copy import deepcopy

from common_config import basic_cfg

cfg = deepcopy(basic_cfg)
cfg.name = "flexunet_resnet34_r3"
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

# --- Mixup: DISABLED for Round 2 ---
# Round 1 finding: Mixup + partial annotations = mixed targets dilute toward
# zero for unannotated channels, subtly suppressing recall. FlexUNet was the
# only model using Mixup and had the best ves/endo Dice, but disabling it
# should improve all rare classes with the new recall-biased loss.
cfg.mixup_p = 0.0
cfg.mixup_beta = 1.0

# --- Patches: 160³ b2 on H100 80GB — 160³ b4 OOM'd at ~80 GB (job 1690639).
#     b2 halves memory vs b4. 160³ b2 est. ~40 GB / 80 GB (40 GB headroom).
#     H100 policy: jobs using <48 GB VRAM may be killed. b2 × num_samples=4
#     → 47 GB per GPU (below threshold). Bump num_samples 4→5 → ~58 GB. ---
cfg.roi_size = [160, 160, 160]
cfg.num_samples = 5
cfg.batch_size = 2

# --- Training (Round 2: 300 epochs — best was at ep204/600) ---
cfg.lr = 1e-3
cfg.optimizer = "Adam"
cfg.weight_decay = 0.0
cfg.epochs = 300
cfg.eval_epochs = 5
