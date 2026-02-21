"""
Class Weighting Ablation Configs for 3D.

Equivalent to experiments/class_weighting/ from 2D, adapted for 3D.

Tests different class-weighting strategies with the best masking settings
from the masking ablation (run masking first, then use winner here).

Until masking results are in, uses a moderate bbox_bg_weight=0.20 as default.

Strategies tested:
  1. Plain Tversky (no logit adjustment) — baseline
  2. Balanced Softmax τ sweep: 0.5, 1.0, 2.0
  3. Disabled balanced softmax (τ=0.0 → no adjustment)

Fixed: SegResNet 32f, 128³, 100 epochs, α=0.6, β=0.4
"""

from copy import deepcopy
from cfg_ablation_base import ablation_base_cfg

# ============================================================
# Default masking: moderate (updated after masking ablation)
# ============================================================
DEFAULT_BBOX_BG_WEIGHT = 0.20
DEFAULT_BBOX_PAD_FRACTION = 0.05


def _make_cfg(name, **overrides):
    """Create a class-weighting ablation config."""
    c = deepcopy(ablation_base_cfg)
    c.name = f"abl_weight_{name}"
    c.output_dir = f"/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/ablations/weight_{name}"
    # Apply default moderate masking
    c.bbox_bg_weight = DEFAULT_BBOX_BG_WEIGHT
    c.bbox_pad_fraction = DEFAULT_BBOX_PAD_FRACTION
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


# ============================================================
# Configs
# ============================================================

CONFIGS = {}

# Baseline: Plain Tversky (no logit adjustment at all)
CONFIGS['plain_tversky'] = _make_cfg('plain_tversky',
    loss_type='tversky',
    # Note: PartialTverskyLoss doesn't do bbox masking or balanced softmax.
    # It only does annotation mask. So this config + bbox masking won't
    # actually use bbox masking. Need to verify behaviour.
)

# Balanced Softmax τ sweep
CONFIGS['balsoftmax_tau_0.0'] = _make_cfg('balsoftmax_tau_0.0',
    loss_type='balanced_softmax_tversky',
    tau=0.0,  # No adjustment — equivalent to unweighted
)

CONFIGS['balsoftmax_tau_0.5'] = _make_cfg('balsoftmax_tau_0.5',
    loss_type='balanced_softmax_tversky',
    tau=0.5,
)

CONFIGS['balsoftmax_tau_1.0'] = _make_cfg('balsoftmax_tau_1.0',
    loss_type='balanced_softmax_tversky',
    tau=1.0,  # Theory-optimal, 2D experiment winner
)

CONFIGS['balsoftmax_tau_2.0'] = _make_cfg('balsoftmax_tau_2.0',
    loss_type='balanced_softmax_tversky',
    tau=2.0,
)

CONFIGS['balsoftmax_tau_3.0'] = _make_cfg('balsoftmax_tau_3.0',
    loss_type='balanced_softmax_tversky',
    tau=3.0,  # Aggressive — might help with 35-class extreme imbalance
)


# ============================================================
# Priority ordering
# ============================================================

PRIORITY_ORDER = [
    'balsoftmax_tau_1.0',    # Current default (2D winner)
    'balsoftmax_tau_0.0',    # No adjustment baseline
    'balsoftmax_tau_2.0',    # Stronger adjustment
    'balsoftmax_tau_0.5',    # Milder adjustment
    'plain_tversky',         # No balanced softmax at all
    'balsoftmax_tau_3.0',    # Very aggressive
]


def get_config(name):
    return CONFIGS[name]


def get_all_configs():
    return [(name, CONFIGS[name]) for name in PRIORITY_ORDER]


cfg = CONFIGS[PRIORITY_ORDER[0]]
