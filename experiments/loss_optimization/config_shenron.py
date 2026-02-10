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

# Workers per GPU - MUST BE 0 in DDP context to prevent fork bomb
# DataLoader workers + DDP + OpenMP = exponential thread explosion
NUM_WORKERS = 0  # Only safe option with DDP

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
        'input_shape': (1, 256, 256),
        'batch_size': 28,  # 4 GPUs × 28 = 112 per step - optimized for 11GB GPUs
        'description': '2D UNet - single slice input',
    },
    'unet_25d': {
        'input_channels': 5,
        'input_shape': (5, 256, 256),
        'batch_size': 18,  # 4 GPUs × 18 = 72 per step - 5x input channels
        'description': '2.5D UNet - 5 adjacent Z-slices for context',
    },
}

# Legacy batch sizes dict for compatibility
BATCH_SIZES = {
    'unet_2d': 28,
    'unet_25d': 18,
    'resnet_2d': 24,
    'swin_2d': 14,
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
# REVISED based on Feb 6 2026 experiment results:
# Previous weights (3.5× for nuc) were TOO aggressive and prevented learning
# New strategy: Moderate boost (1.2-1.8×) to balance without starving easy classes
#
# Feb 6 Results with old weights:
#   nuc: 0.000 (was 3.5×) - completely failed to learn
#   golgi_mem: 0.0001 (was 1.0×) - collapsed from starvation
#   mito_mem: 0.070 (was 1.8×) - showed some learning
#
CLASS_LOSS_WEIGHTS = {
    # Critical classes - moderate boost to encourage learning without preventing it
    'nuc': 1.8,         # 0.111 baseline - composite class (reduced from 3.5)
    'endo_mem': 1.7,    # 0.081 baseline - worst performer (reduced from 3.0)
    'endo_lum': 1.7,    # 0.099 baseline (reduced from 3.0)
    'pm': 1.6,          # 0.113 baseline - thin boundary (reduced from 2.5)
    'er_mem': 1.6,      # 0.116 baseline (reduced from 2.5)
    'er_lum': 1.5,      # 0.143 baseline (reduced from 2.0)
    
    # Hard classes - small boost
    'ves_mem': 1.4,     # 0.193 baseline (reduced from 2.0)
    'mito_mem': 1.3,    # 0.218 baseline (reduced from 1.8)
    
    # Moderate classes - minimal boost to maintain baseline
    'mito_lum': 1.2,    # 0.257 baseline (reduced from 1.5)
    'ves_lum': 1.2,     # 0.270 baseline (reduced from 1.5)
    'ecs': 1.2,         # 0.291 baseline (reduced from 1.5)
    'golgi_lum': 1.2,   # 0.317 baseline (reduced from 1.5)
    
    # Good classes - maintain some weight so they help train features
    'mito_ribo': 1.0,   # 0.643 baseline - keep as anchor
    'golgi_mem': 1.1,   # 0.680 baseline - slight boost to prevent starvation
}

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# Quick test configuration (5-10 min)
QUICK_TEST_CONFIG = {
    'epochs': 5,
    'iterations_per_epoch': 50,
    'batch_size': 28,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 2,
}

# Loss comparison configuration (~2.5-3 hours)
# UPDATED: 30 epochs to give weighted losses fair convergence time
LOSS_COMPARISON_CONFIG = {
    'epochs': 30,
    'iterations_per_epoch': 100,
    'batch_size': 28,
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,
    'validate_every': 5,
}

# 2D vs 2.5D Model comparison configuration (2-4 hours with 4 GPUs)
# Run 2D first, then 2.5D, compare results
# UPDATED: Increased epochs based on Feb 6 results (30 epochs insufficient)
MODEL_COMPARISON_CONFIG = {
    'epochs': 50,  # Increased from 30 - complex weighted loss needs more time
    'iterations_per_epoch': 100,  # Full training with 128GB RAM
    'learning_rate': 1e-4,
    'classes': QUICK_TEST_CLASSES,  # 5 classes including nuc
    'validate_every': 5,
    'loss': 'per_class_weighted',
}

# Full training configuration (8-12 hours)
FULL_TRAIN_CONFIG = {
    'epochs': 100,
    'iterations_per_epoch': 200,
    'batch_size': 28,
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
    'num_workers': 0,  # CRITICAL: Must be 0 with DDP to prevent fork bomb
    'pin_memory': True,  # Faster GPU transfer, using 128GB RAM
    'persistent_workers': False,
    'prefetch_factor': None,
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
