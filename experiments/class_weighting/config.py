"""
Configuration for class-weighting experiments on Rocinante.

Hardware: 2x RTX 3090 (25GB each), AMD Ryzen 9 5950X, 252GB RAM
Data: /home/spuser/ws/CellMap-Segmentation/data (local)

This experiment fixes the loss function to **per-class Tversky (α=0.6, β=0.4)**
— the best loss from the Shenron comparison — and varies only the class
weighting strategy.

Strategies tested:
- Static weights: uniform, manual, inverse-freq, sqrt-inverse, log-inverse,
  effective-number  (from compute_class_frequencies.py)
- Dynamic weights: Class-Balanced (CB), Balanced Softmax, Seesaw
"""

import os
import torch
from pathlib import Path

# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

N_GPUS = 2
GPU_MEMORY_GB = 25
TOTAL_CPU_THREADS = 96
NUM_WORKERS = 2     # per-process; parallel mode runs 4 procs × 2 = 8 workers total

# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path("/home/spuser/ws/CellMap-Segmentation")
DATA_ROOT = REPO_ROOT / "data"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "class_weighting"

CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
TENSORBOARD_DIR = EXPERIMENT_DIR / "runs"
RESULTS_DIR = EXPERIMENT_DIR / "results"

# ============================================================
# CLASSES  (5 quick-test classes)
# ============================================================

QUICK_TEST_CLASSES = ['nuc', 'mito_mem', 'er_mem', 'pm', 'golgi_mem']

# ============================================================
# ESTIMATED VOXEL COUNTS
#
# Set to None to let CB/BalancedSoftmax/Seesaw losses estimate
# online from training data.  If compute_class_frequencies.py has
# been run, paste the counts here for reproducibility.
# ============================================================

ESTIMATED_VOXEL_COUNTS = None

# ============================================================
# FIXED TVERSKY PARAMETERS  (best from Shenron loss comparison)
# ============================================================

TVERSKY_ALPHA = 0.6   # FP weight — mild precision bias
TVERSKY_BETA = 0.4    # FN weight

# ============================================================
# AUTO-LOADED FREQUENCY WEIGHTS
#
# If compute_class_frequencies.py has been run, load the computed
# weight strategies from class_frequencies.json.  Otherwise fall
# back to uniform 1.0 (placeholder).
# ============================================================

import json as _json

_FREQ_FILE = EXPERIMENT_DIR / "class_frequencies.json"
_UNIFORM = {c: 1.0 for c in QUICK_TEST_CLASSES}

if _FREQ_FILE.exists():
    with open(_FREQ_FILE) as _f:
        _freq_data = _json.load(_f)
    _strategies = _freq_data.get('weight_strategies', {})
    _INV_FREQ_WEIGHTS   = _strategies.get('inv_freq',      _UNIFORM)
    _SQRT_INV_WEIGHTS   = _strategies.get('sqrt_inv',      _UNIFORM)
    _LOG_INV_WEIGHTS    = _strategies.get('log_inv',       _UNIFORM)
    _EFF_NUM_WEIGHTS    = _strategies.get('effective_num',  _UNIFORM)
    # Also populate voxel counts if available
    if ESTIMATED_VOXEL_COUNTS is None and 'estimated_voxel_counts' in _freq_data:
        ESTIMATED_VOXEL_COUNTS = _freq_data['estimated_voxel_counts']
    print(f"[config] Loaded frequency weights from {_FREQ_FILE.name}")
else:
    _INV_FREQ_WEIGHTS   = _UNIFORM
    _SQRT_INV_WEIGHTS   = _UNIFORM
    _LOG_INV_WEIGHTS    = _UNIFORM
    _EFF_NUM_WEIGHTS    = _UNIFORM
    print(f"[config] ⚠ {_FREQ_FILE.name} not found — using uniform placeholders. "
          f"Run compute_class_frequencies.py first.")

# ============================================================
# MODEL CONFIGURATION  (UNet 2D only)
# ============================================================

MODEL_CONFIG = {
    'name': 'unet_2d',
    'input_channels': 1,
    'input_shape': (1, 256, 256),
    'batch_size': 24,
}

SCALE = (8, 8, 8)

# ============================================================
# TRAINING CONFIGURATIONS
# ============================================================

QUICK_TEST_CONFIG = {
    'epochs': 5,
    'iterations_per_epoch': 20,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 1,
}

# Per the MD spec: 60 epochs, 100 iter/epoch
WEIGHTING_COMPARISON_CONFIG = {
    'epochs': 60,
    'iterations_per_epoch': 100,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 1,
}

# ============================================================
# LOSS CONFIGURATIONS TO TEST
#
# ALL use per-class Tversky (α=0.6, β=0.4).
# Only the weighting strategy varies.
# ============================================================

LOSS_CONFIGS = {
    # -----------------------------------------------------------
    # Static weight strategies (from MD file §5)
    # All use type='per_class_tversky' with α=0.6, β=0.4
    # -----------------------------------------------------------
    'weight_uniform': {
        'type': 'per_class_tversky',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'class_weights': {
            'nuc': 1.0, 'mito_mem': 1.0, 'er_mem': 1.0,
            'pm': 1.0, 'golgi_mem': 1.0,
        },
        'description': 'Tversky α=0.6 — uniform weights (baseline)',
    },

    'weight_manual': {
        'type': 'per_class_tversky',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'class_weights': {
            'nuc': 1.8, 'mito_mem': 1.3, 'er_mem': 1.6,
            'pm': 1.6, 'golgi_mem': 1.1,
        },
        'description': 'Tversky α=0.6 — manually tuned weights',
    },

    # NOTE: The following 4 configs have PLACEHOLDER weights.
    # Run compute_class_frequencies.py first, then paste the
    # computed values from class_frequencies.json here.

    'weight_inv_freq': {
        'type': 'per_class_tversky',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'class_weights': _INV_FREQ_WEIGHTS,
        'description': 'Tversky α=0.6 — inverse frequency weights',
    },

    'weight_sqrt_inv': {
        'type': 'per_class_tversky',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'class_weights': _SQRT_INV_WEIGHTS,
        'description': 'Tversky α=0.6 — sqrt inverse frequency weights',
    },

    'weight_log_inv': {
        'type': 'per_class_tversky',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'class_weights': _LOG_INV_WEIGHTS,
        'description': 'Tversky α=0.6 — log inverse frequency weights',
    },

    'weight_effective_num': {
        'type': 'per_class_tversky',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'class_weights': _EFF_NUM_WEIGHTS,
        'description': 'Tversky α=0.6 — effective number weights',
    },

    # -----------------------------------------------------------
    # Dynamic weight strategies (CB, Balanced Softmax, Seesaw)
    # These compute weights online from the training data
    # -----------------------------------------------------------
    'cb_beta_0.99': {
        'type': 'class_balanced',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'beta_cb': 0.99,
        'description': 'CB Tversky β_cb=0.99 (mild damping)',
    },
    'cb_beta_0.999': {
        'type': 'class_balanced',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'beta_cb': 0.999,
        'description': 'CB Tversky β_cb=0.999 (medium damping)',
    },
    'cb_beta_0.9999': {
        'type': 'class_balanced',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'beta_cb': 0.9999,
        'description': 'CB Tversky β_cb=0.9999 (strong damping)',
    },

    'balanced_softmax_tau_0.5': {
        'type': 'balanced_softmax',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'tau': 0.5,
        'description': 'Balanced Softmax Tversky τ=0.5 (mild)',
    },
    'balanced_softmax_tau_1.0': {
        'type': 'balanced_softmax',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'tau': 1.0,
        'description': 'Balanced Softmax Tversky τ=1.0 (standard)',
    },
    'balanced_softmax_tau_2.0': {
        'type': 'balanced_softmax',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'tau': 2.0,
        'description': 'Balanced Softmax Tversky τ=2.0 (strong)',
    },

    'seesaw_default': {
        'type': 'seesaw',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'p_mitigation': 0.8,
        'q_compensation': 2.0,
        'description': 'Seesaw Tversky p=0.8 q=2.0 (paper defaults)',
    },
    'seesaw_strong_mitigate': {
        'type': 'seesaw',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'p_mitigation': 1.0,
        'q_compensation': 2.0,
        'description': 'Seesaw Tversky p=1.0 q=2.0 (stronger mitigation)',
    },
    'seesaw_strong_compensate': {
        'type': 'seesaw',
        'alpha': TVERSKY_ALPHA,
        'beta': TVERSKY_BETA,
        'p_mitigation': 0.8,
        'q_compensation': 3.0,
        'description': 'Seesaw Tversky p=0.8 q=3.0 (stronger compensation)',
    },
}

# ============================================================
# OPTIMIZATION
# ============================================================

OPTIMIZER_CONFIG = {
    'name': 'adamw',
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'betas': (0.9, 0.999),
}

MAX_GRAD_NORM = 1.0
USE_AMP = True

# ============================================================
# SPATIAL TRANSFORMS (2D)
# ============================================================

SPATIAL_TRANSFORMS_2D = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

# ============================================================
# DATALOADER
# ============================================================

DATALOADER_CONFIG = {
    'num_workers': NUM_WORKERS,
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 4,
}

# ============================================================
# VALIDATION
# ============================================================

VALIDATION_CONFIG = {
    'batch_limit': 20,
    'time_limit': 60,
}

# ============================================================
# HELPERS
# ============================================================

def get_config(mode: str = 'quick_test'):
    configs = {
        'quick_test': QUICK_TEST_CONFIG,
        'weighting_comparison': WEIGHTING_COMPARISON_CONFIG,
    }
    return configs.get(mode, QUICK_TEST_CONFIG)


def ensure_dirs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
