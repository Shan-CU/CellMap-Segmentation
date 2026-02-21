"""
Base configuration for 3D ablation experiments.

Uses SegResNet 32f at 128³ as a fast test bed (~3.5h/100ep on 2×H100).
Short runs (100 epochs) to compare strategies before committing to full
600-epoch training.

All ablation configs import this base and override only the relevant
hyperparameters being tested.

Ablation axes (matching the 2D experiments):
  1. masking_strategies: bbox_bg_weight sweep, foreground-only, no_mask, masksup
  2. class_weighting: balanced_softmax τ sweep, seesaw, plain Tversky
  3. tversky_tuning: α/β sweep for false-positive vs false-negative balance

Each axis produces ~6-10 configs, each running 100 epochs on 1×GPU or 2×GPU.
"""

from copy import deepcopy
from common_config import basic_cfg

cfg = deepcopy(basic_cfg)

# === Experiment identity ===
cfg.name = "ablation_base"
cfg.output_dir = "/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/ablations/ablation_base"

# === Model: SegResNet 32f (smallest, fastest) ===
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

# === Patches: 128³ fits on both H100 80GB and L40S 48GB ===
cfg.roi_size = [128, 128, 128]
cfg.num_samples = 4
cfg.batch_size = 2

# === Short ablation runs: 100 epochs (~3.5h on 2×H100) ===
cfg.epochs = 100
cfg.lr = 2e-4
cfg.eval_epochs = 5
cfg.save_every_n_epochs = 25  # save checkpoints at 25, 50, 75, 100

# === Loss defaults (R3 settings — will be overridden per experiment) ===
cfg.loss_type = "balanced_softmax_tversky"
cfg.tversky_alpha = 0.6
cfg.tversky_beta = 0.4
cfg.tau = 1.0
cfg.update_interval = 50
cfg.bbox_pad_fraction = 0.05
cfg.bbox_bg_weight = 0.05
cfg.masksup_ratio = 0.0
cfg.masksup_recon_weight = 0.5

# === Resources ===
cfg.num_workers = 8  # 8 workers/rank × 2 ranks = 16 total (fits 32 CPUs on H100, 16 on L40S)
cfg.pin_memory = False  # disabled: in-memory cache + parallel processes causes OOM with pinning
cfg.cache_rate = 1.0
cfg.max_cache_file_size_mb = 500

ablation_base_cfg = cfg
