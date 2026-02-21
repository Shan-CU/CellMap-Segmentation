"""
Tversky α/β Tuning Ablation Configs for 3D.

Equivalent to the Tversky tuning section of experiments/class_weighting/ from 2D.

The 2D class_weighting experiment showed universally low precision / high recall
across all models. In 3D with partial annotations, the FP/FN balance is even
more critical because unannotated regions create implicit false positives.

Sweeps:
  - α (FP penalty): 0.5, 0.6, 0.7, 0.8
  - β (FN penalty): 0.4, 0.5, 0.6, 0.7
  Key constraint: keep β ≥ 0.4 (low β kills structural signal)

Fixed: BalancedSoftmax τ=1.0, SegResNet 32f, 128³, 100 epochs
Uses moderate bbox masking (updated after masking ablation).
"""

from copy import deepcopy
from cfg_ablation_base import ablation_base_cfg

DEFAULT_BBOX_BG_WEIGHT = 0.20
DEFAULT_BBOX_PAD_FRACTION = 0.05


def _make_cfg(name, **overrides):
    c = deepcopy(ablation_base_cfg)
    c.name = f"abl_tversky_{name}"
    c.output_dir = f"/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/ablations/tversky_{name}"
    c.bbox_bg_weight = DEFAULT_BBOX_BG_WEIGHT
    c.bbox_pad_fraction = DEFAULT_BBOX_PAD_FRACTION
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


CONFIGS = {}

# Current default (from 2D experiments)
CONFIGS['a06_b04'] = _make_cfg('a06_b04',
    tversky_alpha=0.6,
    tversky_beta=0.4,
)

# Symmetric — equal FP/FN penalty
CONFIGS['a05_b05'] = _make_cfg('a05_b05',
    tversky_alpha=0.5,
    tversky_beta=0.5,
)

# Higher FP penalty (more precision-biased)
CONFIGS['a07_b04'] = _make_cfg('a07_b04',
    tversky_alpha=0.7,
    tversky_beta=0.4,
)

CONFIGS['a07_b06'] = _make_cfg('a07_b06',
    tversky_alpha=0.7,
    tversky_beta=0.6,
)

CONFIGS['a08_b06'] = _make_cfg('a08_b06',
    tversky_alpha=0.8,
    tversky_beta=0.6,
)

# Higher overall penalty (both high)
CONFIGS['a06_b06'] = _make_cfg('a06_b06',
    tversky_alpha=0.6,
    tversky_beta=0.6,
)

# More recall-biased (might help with very sparse classes)
CONFIGS['a04_b06'] = _make_cfg('a04_b06',
    tversky_alpha=0.4,
    tversky_beta=0.6,
)

# Strong precision bias (from 2D tuning experiment)
CONFIGS['a08_b07'] = _make_cfg('a08_b07',
    tversky_alpha=0.8,
    tversky_beta=0.7,
)


PRIORITY_ORDER = [
    'a06_b04',   # Current default
    'a07_b06',   # Strong FP + strong FN (2D tuning suggested this range)
    'a08_b06',   # Very strong FP penalty
    'a05_b05',   # Symmetric baseline
    'a06_b06',   # Equal but higher overall
    'a07_b04',   # More FP penalty, same FN
    'a04_b06',   # Recall-biased
    'a08_b07',   # Maximum penalty
]


def get_config(name):
    return CONFIGS[name]


def get_all_configs():
    return [(name, CONFIGS[name]) for name in PRIORITY_ORDER]


cfg = CONFIGS[PRIORITY_ORDER[0]]
