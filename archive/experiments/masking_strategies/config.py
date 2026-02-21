# -*- coding: utf-8 -*-
"""
Configuration for masking strategy experiments on Rocinante.

Hardware: 2x RTX 3090 (25GB each), AMD Ryzen 9 5950X, 252GB RAM
Data: /home/spuser/ws/CellMap-Segmentation/data (local)

This experiment fixes the loss function to **BalancedSoftmax Tversky**
(alpha=0.6, beta=0.4, tau=1.0) -- the best config from the class-weighting
comparison (Dice=0.3171) -- and varies only the NaN/unannotated pixel
**masking strategy**.

Motivation:
  All 15 models from the class-weighting experiment showed universally
  low precision / high recall.  Root cause hypothesis: NaN masking during
  training creates a train/eval mismatch -- the model is never penalised
  for false positives on unannotated pixels, but evaluation counts them.

Strategies tested:
  0. no_mask         -- NaN -> 0, all pixels contribute
  1. masksup         -- Siamese reconstruction of randomly masked pixels
  2. regional_weight -- Grid-based adaptive per-region weighting
  3. uncertainty_eu  -- Epistemic uncertainty guided masking (MC-Dropout)
  4. uncertainty_au  -- Aleatoric uncertainty: exclude noisy pixels
  5. box_class_mask  -- Per-class bounding-box spatial masking
  6. salient_mask    -- Differential masking: low ratio for FG, high for BG
  7. entropy_mask    -- Dynamic entropy threshold masking
  8. class_presence  -- Mask entire image for absent classes, no mask for present
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
NUM_WORKERS = 2

# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path("/home/spuser/ws/CellMap-Segmentation")
DATA_ROOT = REPO_ROOT / "data"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "masking_strategies"

# Reference the class_weighting datasplit for reproducibility
CLASS_WEIGHT_DIR = REPO_ROOT / "experiments" / "class_weighting"
DATASPLIT_CSV = CLASS_WEIGHT_DIR / "datasplit.csv"

CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
TENSORBOARD_DIR = EXPERIMENT_DIR / "runs"
RESULTS_DIR = EXPERIMENT_DIR / "results"

# ============================================================
# CLASSES  (same 5 quick-test classes)
# ============================================================

QUICK_TEST_CLASSES = ['nuc', 'mito_mem', 'er_mem', 'pm', 'golgi_mem']

# ============================================================
# FIXED TVERSKY + BALANCED SOFTMAX PARAMETERS
# (best from class-weighting Exp 1)
# ============================================================

TVERSKY_ALPHA = 0.6
TVERSKY_BETA = 0.4
BALANCED_SOFTMAX_TAU = 1.0

# ============================================================
# MASKING STRATEGY CONFIGURATIONS
# ============================================================

MASKING_CONFIGS = {
    # ── Baseline: simple no-mask ──────────────────────────────
    'no_mask': {
        'strategy': 'no_mask',
        'description': 'No masking -- NaN treated as background (0), all pixels contribute',
    },

    # ── Strategy 1: MaskSup ───────────────────────────────────
    'masksup_r0.3': {
        'strategy': 'masksup',
        'mask_ratio': 0.3,
        'recon_weight': 0.5,
        'description': 'MaskSup -- 30% random mask, lambda=0.5 reconstruction loss',
    },
    'masksup_r0.5': {
        'strategy': 'masksup',
        'mask_ratio': 0.5,
        'recon_weight': 0.5,
        'description': 'MaskSup -- 50% random mask, lambda=0.5 reconstruction loss',
    },

    # ── Strategy 2: Regional Adaptive ─────────────────────────
    'regional_g8': {
        'strategy': 'regional_weight',
        'grid_size': 8,
        'momentum': 0.9,
        'description': 'Regional Adaptive -- 8x8 grid, momentum=0.9',
    },
    'regional_g16': {
        'strategy': 'regional_weight',
        'grid_size': 16,
        'momentum': 0.9,
        'description': 'Regional Adaptive -- 16x16 grid, momentum=0.9',
    },

    # ── Strategy 3: Epistemic Uncertainty ─────────────────────
    'uncertainty_eu': {
        'strategy': 'uncertainty_eu',
        'n_mc': 4,
        'uncertainty_weight': 2.0,
        'warmup_epochs': 5,
        'description': 'Epistemic Uncertainty -- 4 MC passes, up-weight=2x, warmup=5ep',
    },

    # ── Strategy 4: Aleatoric Uncertainty ─────────────────────
    'uncertainty_au': {
        'strategy': 'uncertainty_au',
        'au_threshold': 0.9,
        'description': 'Aleatoric Uncertainty -- exclude top 10% noisy pixels',
    },

    # ── Strategy 5: Box-Driven Class Masking ──────────────────
    'box_class_mask': {
        'strategy': 'box_class_mask',
        'pad_fraction': 0.15,
        'bg_weight': 0.1,
        'description': 'Box Class Mask -- padded bbox per class, bg_weight=0.1',
    },
    'box_class_mask_tight': {
        'strategy': 'box_class_mask',
        'pad_fraction': 0.05,
        'bg_weight': 0.05,
        'description': 'Box Class Mask tight -- small pad, bg_weight=0.05',
    },

    # ── Strategy 6: Salient Masking ───────────────────────────
    'salient_mask': {
        'strategy': 'salient_mask',
        'fg_mask_ratio': 0.15,
        'bg_mask_ratio': 0.5,
        'description': 'Salient Mask -- keep 85% FG, 50% BG',
    },
    'salient_mask_aggressive': {
        'strategy': 'salient_mask',
        'fg_mask_ratio': 0.1,
        'bg_mask_ratio': 0.7,
        'description': 'Salient Mask aggressive -- keep 90% FG, 30% BG',
    },

    # ── Strategy 7: Dynamic Entropy ───────────────────────────
    'entropy_mask': {
        'strategy': 'entropy_mask',
        'high_entropy_percentile': 0.9,
        'mid_entropy_boost': 1.5,
        'description': 'Dynamic Entropy -- exclude top 10%, boost mid-uncertainty 1.5x',
    },
    'entropy_mask_strict': {
        'strategy': 'entropy_mask',
        'high_entropy_percentile': 0.8,
        'mid_entropy_boost': 2.0,
        'description': 'Dynamic Entropy strict -- exclude top 20%, boost mid 2.0x',
    },

    # ── Strategy 8: Class-Presence Masking ────────────────────
    'class_presence': {
        'strategy': 'class_presence',
        'presence_threshold': 0.001,
        'description': 'Class Presence -- mask image for absent classes, no mask for present',
    },
    'class_presence_strict': {
        'strategy': 'class_presence',
        'presence_threshold': 0.01,
        'description': 'Class Presence strict -- need 1% positive pixels to count as present',
    },
}

# ============================================================
# MODEL CONFIGURATION  (UNet 2D -- same as class-weighting)
# ============================================================

MODEL_CONFIG = {
    'name': 'unet_2d',
    'input_channels': 1,
    'input_shape': (1, 256, 256),
    'batch_size': 24,
}

SCALE = (8, 8, 8)

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

TRAINING_CONFIG = {
    'epochs': 60,
    'iterations_per_epoch': 100,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 1,
}

QUICK_TEST_CONFIG = {
    'epochs': 1,
    'iterations_per_epoch': 20,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 1,
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
    'batch_limit': 360,
    'time_limit': 300,
}

# ============================================================
# HELPERS
# ============================================================

def get_config(mode: str = 'full'):
    configs = {
        'quick_test': QUICK_TEST_CONFIG,
        'full': TRAINING_CONFIG,
    }
    return configs.get(mode, TRAINING_CONFIG)


def ensure_dirs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
