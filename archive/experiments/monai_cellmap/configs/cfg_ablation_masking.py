"""
Masking Strategy Ablation Configs for 3D.

Equivalent to experiments/masking_strategies/ from 2D, adapted for 3D.

KEY FINDING from R3: bbox_bg_weight=0.05 was too aggressive for 35-class 3D
(95% loss reduction outside bbox → model ignored spatial context → R3 WORSE than R2).

This experiment sweeps:
  1. bbox_bg_weight: 0.0 (no_mask), 0.05, 0.1, 0.2, 0.5, 1.0
  2. bbox_pad_fraction: 0.05 (tight), 0.10, 0.15 (loose)
  3. masksup: 0.0 (off), 0.3
  4. foreground-only (no bbox, just FG masking)

Fixed: BalancedSoftmax Tversky (α=0.6, β=0.4, τ=1.0)
Model: SegResNet 32f, 128³, 100 epochs
"""

from copy import deepcopy
from cfg_ablation_base import ablation_base_cfg

# ============================================================
# Helper to create named configs
# ============================================================

def _make_cfg(name, **overrides):
    """Create an ablation config with the given name and overrides."""
    c = deepcopy(ablation_base_cfg)
    c.name = f"abl_mask_{name}"
    c.output_dir = f"/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/ablations/mask_{name}"
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


# ============================================================
# Experiment 1: bbox_bg_weight sweep (most critical)
#
# R3 used 0.05 → too aggressive. R2 had no masking (effectively 1.0).
# Hypothesis: optimal value is somewhere between 0.1-0.5 for 35-class 3D.
# ============================================================

CONFIGS = {}

# Baseline: NO spatial masking at all (equivalent to R2 loss)
# This means bbox_bg_weight=1.0 (voxels outside bbox weighted same as inside)
CONFIGS['no_mask'] = _make_cfg('no_mask',
    bbox_bg_weight=1.0,
    bbox_pad_fraction=0.0,
)

# R3 setting (tight): pad=0.05, bg=0.05
CONFIGS['bbox_tight_bg005'] = _make_cfg('bbox_tight_bg005',
    bbox_pad_fraction=0.05,
    bbox_bg_weight=0.05,
)

# Sweep bbox_bg_weight with tight padding
CONFIGS['bbox_tight_bg010'] = _make_cfg('bbox_tight_bg010',
    bbox_pad_fraction=0.05,
    bbox_bg_weight=0.10,
)

CONFIGS['bbox_tight_bg020'] = _make_cfg('bbox_tight_bg020',
    bbox_pad_fraction=0.05,
    bbox_bg_weight=0.20,
)

CONFIGS['bbox_tight_bg050'] = _make_cfg('bbox_tight_bg050',
    bbox_pad_fraction=0.05,
    bbox_bg_weight=0.50,
)

# ============================================================
# Experiment 2: bbox_pad_fraction sweep (secondary)
#
# Tight (0.05) means 5% of bbox extent padded. Maybe 3D needs looser.
# ============================================================

CONFIGS['bbox_loose_bg010'] = _make_cfg('bbox_loose_bg010',
    bbox_pad_fraction=0.10,
    bbox_bg_weight=0.10,
)

CONFIGS['bbox_loose_bg020'] = _make_cfg('bbox_loose_bg020',
    bbox_pad_fraction=0.10,
    bbox_bg_weight=0.20,
)

CONFIGS['bbox_wide_bg010'] = _make_cfg('bbox_wide_bg010',
    bbox_pad_fraction=0.15,
    bbox_bg_weight=0.10,
)

# ============================================================
# Experiment 3: MaskSup reconstruction (didn't test in 3D yet)
#
# masksup_r0.3 was +12% Dice in 2D. Worth testing in 3D.
# Combine with moderate bbox masking.
# ============================================================

CONFIGS['masksup_r03_bg020'] = _make_cfg('masksup_r03_bg020',
    bbox_pad_fraction=0.05,
    bbox_bg_weight=0.20,
    masksup_ratio=0.3,
    masksup_recon_weight=0.5,
)

CONFIGS['masksup_r03_bg010'] = _make_cfg('masksup_r03_bg010',
    bbox_pad_fraction=0.05,
    bbox_bg_weight=0.10,
    masksup_ratio=0.3,
    masksup_recon_weight=0.5,
)

# ============================================================
# Experiment 4: Foreground-only masking (no bbox)
#
# Only zero loss on black-padding EM voxels, no per-class bbox.
# This isolates the foreground masking benefit from bbox masking.
# ============================================================

CONFIGS['fg_only'] = _make_cfg('fg_only',
    bbox_bg_weight=1.0,  # no bbox weighting
    bbox_pad_fraction=0.0,
    # Foreground masking is always on (done in model forward())
    # so setting bbox_bg_weight=1.0 means only FG masking active
)


# ============================================================
# Priority ordering for sequential runs
# ============================================================

PRIORITY_ORDER = [
    # Tier 1: Most informative comparisons (run first)
    'no_mask',              # Baseline (R2-equivalent)
    'fg_only',              # Isolate FG masking
    'bbox_tight_bg020',     # Moderate masking (hypothesis: sweet spot)
    'bbox_tight_bg050',     # Mild masking

    # Tier 2: Refine the sweet spot
    'bbox_tight_bg010',     # Between 0.05 and 0.20
    'bbox_tight_bg005',     # R3 setting (known to be too aggressive)
    'bbox_loose_bg020',     # Looser padding + moderate weight

    # Tier 3: Extra comparisons if time permits
    'bbox_loose_bg010',
    'bbox_wide_bg010',
    'masksup_r03_bg020',
    'masksup_r03_bg010',
]


def get_config(name):
    """Get a specific masking ablation config by name."""
    return CONFIGS[name]


def get_all_configs():
    """Get all configs in priority order."""
    return [(name, CONFIGS[name]) for name in PRIORITY_ORDER]


# Default export: first priority config
cfg = CONFIGS[PRIORITY_ORDER[0]]
