"""
Windows 4090-optimized configuration for loss optimization experiments.

Hardware: Single RTX 4090 (24GB VRAM), i9-14900KS (32 threads), 256GB RAM
Data: E:/CU_Boulder/rotations/KASINATH/CellMap-Segmentation/data
"""

import os
import torch
from pathlib import Path

# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

# Single RTX 4090 with 24GB VRAM - can handle much larger batches
N_GPUS = 1
GPU_MEMORY_GB = 24
TOTAL_CPU_THREADS = 32  # i9-14900KS

# Workers - Windows handles fewer workers well, leave headroom for main process
NUM_WORKERS = 8

# ============================================================
# PATHS (Windows)
# ============================================================

# Base paths
REPO_ROOT = Path(r"E:\CU_Boulder\rotations\KASINATH\CellMap-Segmentation")
DATA_ROOT = REPO_ROOT / "data"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "loss_optimization"

# Output paths
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
TENSORBOARD_DIR = EXPERIMENT_DIR / "runs"
RESULTS_DIR = EXPERIMENT_DIR / "results"

# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Output shape for predictions
OUTPUT_SHAPE = (1, 256, 256)  # Single Z-slice

# Model configurations - 4090 can handle much larger batches
MODEL_CONFIGS = {
    'unet_2d': {
        'input_channels': 1,
        'input_shape': (1, 256, 256),
        'batch_size': 56,  # 4090 can handle ~5x more than 2080 Ti
        'description': '2D UNet - single slice input',
    },
    'unet_25d': {
        'input_channels': 5,
        'input_shape': (5, 256, 256),
        'batch_size': 32,  # Lower due to 5x input channels
        'description': '2.5D UNet - 5 adjacent Z-slices for context',
    },
}

# Legacy batch sizes for compatibility
BATCH_SIZES = {
    'unet_2d': 56,
    'unet_25d': 32,
    'resnet_2d': 48,
    'swin_2d': 24,
}

# Scale - 8nm isotropic
SCALE = (8, 8, 8)

# ============================================================
# CLASSES
# ============================================================

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

# Class weights for loss - REVISED moderate weights (Feb 6 2026 results)
# Old aggressive weights (3.5× nuc) prevented learning entirely
# New strategy: Moderate boost (1.0-1.8×) to balance without starving any class
CLASS_LOSS_WEIGHTS = {
    'nuc': 1.8,
    'endo_mem': 1.7,
    'endo_lum': 1.7,
    'pm': 1.6,
    'er_mem': 1.6,
    'er_lum': 1.5,
    'ves_mem': 1.4,
    'mito_mem': 1.3,
    'mito_lum': 1.2,
    'ves_lum': 1.2,
    'ecs': 1.2,
    'golgi_lum': 1.2,
    'mito_ribo': 1.0,
    'golgi_mem': 1.1,
}

# ============================================================
# TRAINING CONFIGURATIONS
# ============================================================

# Quick test (5-10 min) - verify everything works
QUICK_TEST_CONFIG = {
    'epochs': 5,
    'iterations_per_epoch': 50,
    'batch_size': 56,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 2,
}

# Loss comparison: 30 epochs × 100 iters × 7 losses ≈ 2.5-3.5 hours
LOSS_COMPARISON_CONFIG = {
    'epochs': 30,
    'iterations_per_epoch': 100,
    'batch_size': 56,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 5,
}

# 2D vs 2.5D Model comparison
MODEL_COMPARISON_CONFIG = {
    'epochs': 50,
    'iterations_per_epoch': 100,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 5,
    'loss': 'per_class_weighted',
}

# Full training (2-4 hours)
FULL_TRAIN_CONFIG = {
    'epochs': 100,
    'iterations_per_epoch': 200,
    'batch_size': 56,
    'learning_rate': 1e-4,
    'classes': ALL_CLASSES,
    'validate_every': 10,
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

SCHEDULER_CONFIG = {
    'name': 'one_cycle',
    'pct_start': 0.05,
    'anneal_strategy': 'cos',
    'div_factor': 25,
    'final_div_factor': 1000,
}

MAX_GRAD_NORM = 1.0
USE_AMP = True  # Mixed precision for faster training

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
        'alpha': 0.7,
        'beta': 0.3,
        'description': 'Tversky loss favoring precision (reduce false positives)',
    },

    'tversky_recall': {
        'type': 'tversky',
        'alpha': 0.3,
        'beta': 0.7,
        'description': 'Tversky loss favoring recall',
    },

    'per_class_weighted_focal': {
        'type': 'per_class_combo',
        'bce_weight': 0.3,
        'dice_weight': 0.5,
        'focal_weight': 0.2,
        'focal_gamma': 2.0,
        'class_weights': CLASS_LOSS_WEIGHTS,
        'description': 'Per-class weighted Dice+BCE+Focal (best of both worlds)',
    },

    'per_class_tversky_recall': {
        'type': 'per_class_tversky',
        'alpha': 0.3,
        'beta': 0.7,
        'class_weights': CLASS_LOSS_WEIGHTS,
        'description': 'Per-class weighted Tversky favoring recall (thin structures)',
    },

    'tversky_precision_mild': {
        'type': 'tversky',
        'alpha': 0.6,
        'beta': 0.4,
        'description': 'Tversky mild precision bias (α=0.6)',
    },

    'tversky_precision_strong': {
        'type': 'tversky',
        'alpha': 0.8,
        'beta': 0.2,
        'description': 'Tversky strong precision bias (α=0.8)',
    },

    'per_class_tversky_precision': {
        'type': 'per_class_tversky',
        'alpha': 0.7,
        'beta': 0.3,
        'class_weights': CLASS_LOSS_WEIGHTS,
        'description': 'Per-class weighted Tversky precision (α=0.7)',
    },

    'per_class_tversky_precision_strong': {
        'type': 'per_class_tversky',
        'alpha': 0.8,
        'beta': 0.2,
        'class_weights': CLASS_LOSS_WEIGHTS,
        'description': 'Per-class weighted Tversky strong precision (α=0.8)',
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
    'persistent_workers': True,  # Keep workers alive between epochs
    'prefetch_factor': 4,        # 8 workers × 4 prefetch = 32 batches buffered (256GB RAM)
}

VALIDATION_CONFIG = {
    'batch_limit': 20,
    'time_limit': 60,
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
    configs = {
        'quick_test': QUICK_TEST_CONFIG,
        'loss_comparison': LOSS_COMPARISON_CONFIG,
        'model_comparison': MODEL_COMPARISON_CONFIG,
        'full_train': FULL_TRAIN_CONFIG,
    }
    if mode not in configs:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(configs.keys())}")
    return configs[mode].copy()


def get_model_config(model_name: str = 'unet_2d') -> dict:
    """Get model-specific configuration."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_CONFIGS[model_name].copy()
