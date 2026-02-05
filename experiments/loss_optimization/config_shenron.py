"""
Shenron-optimized configuration for loss optimization experiments.

Hardware: 4x RTX 2080 Ti (11GB each), EPYC 7302 (32 threads), 62GB RAM
Data: /volatile/cellmap/data (symlinked to ./data)
"""

import os
import torch
from pathlib import Path

# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

# Shenron has 4x RTX 2080 Ti
N_GPUS = 4
GPU_MEMORY_GB = 11
TOTAL_CPU_THREADS = 32

# Workers per GPU - leave some threads for main process
NUM_WORKERS = 6  # 6 * 4 = 24 workers, leaves 8 for main processes

# ============================================================
# PATHS (for Shenron)
# ============================================================

# Base paths
REPO_ROOT = Path("/scratch/users/gest9386/CellMap-Segmentation")
DATA_ROOT = Path("/volatile/cellmap/data")  # Symlinked
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "loss_optimization"

# Output paths
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
TENSORBOARD_DIR = EXPERIMENT_DIR / "runs"
RESULTS_DIR = EXPERIMENT_DIR / "results"

# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Input/output shapes
# 2D: Single Z-slice as input
# 2.5D: Multiple adjacent Z-slices as input channels (provides Z-context)
OUTPUT_SHAPE = (1, 256, 256)  # Always predict single slice

# Model configurations for 2D vs 2.5D comparison
MODEL_CONFIGS = {
    'unet_2d': {
        'input_channels': 1,
        'input_shape': (1, 256, 256),   # Single Z-slice
        'batch_size': 12,               # Safe for 11GB GPU
        'description': '2D UNet - single slice input',
    },
    'unet_25d': {
        'input_channels': 5,
        'input_shape': (5, 256, 256),   # 5 adjacent Z-slices as channels
        'batch_size': 8,                # Lower due to 5x input channels
        'description': '2.5D UNet - 5 adjacent Z-slices for context',
    },
}

# Legacy batch sizes dict for compatibility
BATCH_SIZES = {
    'unet_2d': 12,
    'unet_25d': 8,
    'resnet_2d': 10,
    'swin_2d': 6,
}

# Scale - use 8nm isotropic (common for this data)
SCALE = (8, 8, 8)

# ============================================================
# CLASSES
# ============================================================

# All 14 classes
ALL_CLASSES = [
    'ecs', 'pm', 'mito_mem', 'mito_lum', 'mito_ribo',
    'golgi_mem', 'golgi_lum', 'ves_mem', 'ves_lum',
    'endo_mem', 'endo_lum', 'er_mem', 'er_lum', 'nuc'
]

# Subset for quick testing (5 classes with varying difficulty)
QUICK_TEST_CLASSES = [
    'nuc',        # Hard - needs 3D context
    'mito_mem',   # Easy - distinctive
    'er_mem',     # Hard - thin, irregular
    'pm',         # Moderate - thin boundary  
    'golgi_mem',  # Easy - distinctive
]

# Class weights for loss (boost hard classes)
# Weights are INVERSELY proportional to Dice score from UNet 2D evaluation:
#   Dice < 0.15 -> weight 3.0-3.5 (critical)
#   Dice 0.15-0.25 -> weight 2.0-2.5 (hard)
#   Dice 0.25-0.40 -> weight 1.5 (moderate)
#   Dice > 0.40 -> weight 1.0 (good)
CLASS_LOSS_WEIGHTS = {
    # Critical (Dice < 0.15) - need heavy boosting
    'endo_mem': 3.0,    # 0.081 Dice - worst performer
    'endo_lum': 3.0,    # 0.099 Dice
    'nuc': 3.5,         # 0.111 Dice - composite class, needs 3D context
    'pm': 2.5,          # 0.113 Dice - thin boundary
    'er_mem': 2.5,      # 0.116 Dice
    'er_lum': 2.0,      # 0.143 Dice
    
    # Hard (Dice 0.15-0.25)
    'ves_mem': 2.0,     # 0.193 Dice
    'mito_mem': 1.8,    # 0.218 Dice - worse than expected
    
    # Moderate (Dice 0.25-0.40)
    'mito_lum': 1.5,    # 0.257 Dice
    'ves_lum': 1.5,     # 0.270 Dice
    'ecs': 1.5,         # 0.291 Dice
    'golgi_lum': 1.5,   # 0.317 Dice
    
    # Good (Dice > 0.40) - no boost needed
    'mito_ribo': 1.0,   # 0.643 Dice - good!
    'golgi_mem': 1.0,   # 0.680 Dice - best performer
}

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# Quick test configuration (5-10 min)
QUICK_TEST_CONFIG = {
    'epochs': 5,
    'iterations_per_epoch': 50,
    'batch_size': 12,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 2,
}

# Loss comparison configuration (1-2 hours)
LOSS_COMPARISON_CONFIG = {
    'epochs': 20,
    'iterations_per_epoch': 100,
    'batch_size': 12,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 5,
}

# 2D vs 2.5D Model comparison configuration (2-4 hours total)
# Run 2D first, then 2.5D, compare results
MODEL_COMPARISON_CONFIG = {
    'epochs': 30,
    'iterations_per_epoch': 100,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,  # 5 classes including nuc
    'validate_every': 5,
    'loss': 'per_class_weighted',   # Use best loss from prior experiments
}

# Full training configuration (8-12 hours)
FULL_TRAIN_CONFIG = {
    'epochs': 100,
    'iterations_per_epoch': 200,
    'batch_size': 12,
    'learning_rate': 1e-4,
    'classes': ALL_CLASSES,
    'validate_every': 10,
}

# ============================================================
# OPTIMIZATION
# ============================================================

# AdamW with weight decay
OPTIMIZER_CONFIG = {
    'name': 'adamw',
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'betas': (0.9, 0.999),
}

# Cosine annealing with warmup
SCHEDULER_CONFIG = {
    'name': 'one_cycle',
    'pct_start': 0.05,  # 5% warmup
    'anneal_strategy': 'cos',
    'div_factor': 25,
    'final_div_factor': 1000,
}

# Gradient clipping
MAX_GRAD_NORM = 1.0

# Mixed precision for 2x speedup
USE_AMP = True

# ============================================================
# LOSS CONFIGURATIONS TO TEST
# ============================================================

LOSS_CONFIGS = {
    'baseline_bce': {
        'type': 'bce',
        'description': 'Baseline BCEWithLogitsLoss (current default)',
    },
    
    'dice_bce': {
        'type': 'dice_bce',
        'bce_weight': 0.5,
        'dice_weight': 0.5,
        'description': 'Combined Dice + BCE loss',
    },
    
    'focal': {
        'type': 'focal',
        'alpha': 0.25,
        'gamma': 2.0,
        'description': 'Focal loss for hard example mining',
    },
    
    'combo': {
        'type': 'combo',
        'bce_weight': 0.3,
        'dice_weight': 0.5,
        'focal_weight': 0.2,
        'description': 'BCE + Dice + Focal combined',
    },
    
    'per_class_weighted': {
        'type': 'per_class_combo',
        'bce_weight': 0.4,
        'dice_weight': 0.6,
        'class_weights': CLASS_LOSS_WEIGHTS,
        'description': 'Per-class weighted Dice+BCE (boost hard classes)',
    },
    
    'tversky_precision': {
        'type': 'tversky',
        'alpha': 0.7,  # Higher = penalize FP more = higher precision
        'beta': 0.3,
        'description': 'Tversky loss favoring precision (reduce false positives)',
    },
    
    'tversky_recall': {
        'type': 'tversky',
        'alpha': 0.3,
        'beta': 0.7,  # Higher = penalize FN more = higher recall
        'description': 'Tversky loss favoring recall',
    },
}

# ============================================================
# DATA AUGMENTATION
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
    'prefetch_factor': 2,
}

# ============================================================
# VALIDATION
# ============================================================

VALIDATION_CONFIG = {
    'batch_limit': 20,  # Limit val batches for faster iteration
    'time_limit': 60,   # Max 60 seconds per validation
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def ensure_dirs():
    """Create output directories."""
    for d in [CHECKPOINT_DIR, TENSORBOARD_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_config(mode: str = 'quick_test') -> dict:
    """Get configuration for a training mode."""
    if mode == 'quick_test':
        return QUICK_TEST_CONFIG.copy()
    elif mode == 'loss_comparison':
        return LOSS_COMPARISON_CONFIG.copy()
    elif mode == 'model_comparison':
        return MODEL_COMPARISON_CONFIG.copy()
    elif mode == 'full_train':
        return FULL_TRAIN_CONFIG.copy()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_model_config(model_name: str = 'unet_2d') -> dict:
    """Get model-specific configuration."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name].copy()
