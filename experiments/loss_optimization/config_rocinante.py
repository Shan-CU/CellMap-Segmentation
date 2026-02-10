"""
Rocinante-optimized configuration for loss optimization experiments.

Hardware: 2x RTX 3090 (25GB each), AMD Ryzen 9 5950X (32 threads), 252GB RAM
Data: /home/spuser/ws/CellMap-Segmentation/data (local)
"""

import os
import torch
from pathlib import Path

# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

# Rocinante has 2x RTX 3090
N_GPUS = 2
GPU_MEMORY_GB = 25
TOTAL_CPU_THREADS = 96  # 32 cores with 3 threads each

# Workers per GPU - Can use more workers with 252GB RAM
# 6 workers per GPU = 12 total, safe with file_system sharing
NUM_WORKERS = 6

# ============================================================
# PATHS (for Rocinante)
# ============================================================

# Base paths
REPO_ROOT = Path("/home/spuser/ws/CellMap-Segmentation")
DATA_ROOT = REPO_ROOT / "data"
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
        'input_shape': (1, 256, 256),
        'batch_size': 24,  # 2 GPUs × 24 = 48 per step - optimized for 25GB GPUs
        'description': '2D UNet - single slice input',
    },
    'unet_25d': {
        'input_channels': 5,
        'input_shape': (5, 256, 256),
        'batch_size': 16,  # 2 GPUs × 16 = 32 per step - 5x input channels
        'description': '2.5D UNet - 5 adjacent Z-slices for context',
    },
}

# Legacy batch sizes dict for compatibility
BATCH_SIZES = {
    'unet_2d': 24,
    'unet_25d': 16,
    'resnet_2d': 20,
    'swin_2d': 12,
}

# Scale - use 8nm isotropic (common for this data)
SCALE = (8, 8, 8)

# ============================================================
# CLASSES
# ============================================================

# All 14 classes
ALL_CLASSES = [
    'nuc', 'mito', 'mito_mem', 'mito_ribo', 'golgi', 'golgi_mem',
    'vesicle', 'vesicle_mem', 'mvb', 'mvb_mem', 'er', 'er_mem',
    'endo_mem', 'pm'
]

# Quick test with 5 challenging classes (including nuc, the hardest)
QUICK_TEST_CLASSES = ['nuc', 'mito_mem', 'er_mem', 'pm', 'golgi_mem']

# ============================================================
# CLASS WEIGHTS (empirically tuned for hard classes)
# ============================================================

CLASS_LOSS_WEIGHTS = {
    'nuc': 3.0,        # Hardest class - needs most attention
    'endo_mem': 2.5,   # Hard - thin structure, rare
    'er_mem': 2.5,     # Hard - thin structure, complex
    'pm': 2.0,         # Medium - clearer but still thin
    'mito_mem': 1.5,   # Medium-easy - visible but thin
    'golgi_mem': 1.5,  # Medium-easy
    'vesicle_mem': 1.5,
    'mvb_mem': 1.5,
    'mito': 1.0,       # Easier - large, clear structures
    'golgi': 1.0,
    'vesicle': 1.0,
    'mvb': 1.0,
    'er': 1.0,
    'mito_ribo': 1.0,
}

# ============================================================
# TRAINING CONFIGURATIONS
# ============================================================

# Quick test configuration (5-10 min with 2 GPUs)
QUICK_TEST_CONFIG = {
    'epochs': 5,
    'iterations_per_epoch': 20,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 1,
}

# Loss comparison configuration (2-4 hours with 2 GPUs, 20 epochs per loss)
LOSS_COMPARISON_CONFIG = {
    'epochs': 20,
    'iterations_per_epoch': 50,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 5,
}

# 2D vs 2.5D Model comparison configuration (2-4 hours with 2 GPUs)
# Run 2D first, then 2.5D, compare results
MODEL_COMPARISON_CONFIG = {
    'epochs': 50,
    'iterations_per_epoch': 100,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 5,
    'loss': 'per_class_weighted',
}

# Full training configuration (8-12 hours)
FULL_TRAIN_CONFIG = {
    'epochs': 100,
    'iterations_per_epoch': 200,
    'batch_size': 24,
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
    
    # --- Tversky precision exploration (based on friend's results: α=0.7 → 0.370 Dice) ---
    
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
    'num_workers': NUM_WORKERS,  # 6 per GPU with 252GB RAM
    'pin_memory': True,
    'persistent_workers': True,  # Keep workers alive between epochs
    'prefetch_factor': 4,  # Pre-load 4 batches per worker
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

def get_config(mode: str = 'quick_test'):
    """Get training configuration by mode."""
    configs = {
        'quick_test': QUICK_TEST_CONFIG,
        'loss_comparison': LOSS_COMPARISON_CONFIG,
        'model_comparison': MODEL_COMPARISON_CONFIG,
        'full_train': FULL_TRAIN_CONFIG,
    }
    return configs.get(mode, QUICK_TEST_CONFIG)


def get_model_config(model_name: str = 'unet_2d'):
    """Get model configuration by name."""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['unet_2d'])


def ensure_dirs():
    """Create output directories if they don't exist."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    """Get the appropriate device for training."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
