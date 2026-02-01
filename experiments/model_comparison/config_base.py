# Base Configuration for Model Comparison Experiments
# Shared settings across all models

import torch
from upath import UPath

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

# Experiment name and paths
EXPERIMENT_NAME = "transformer_vs_cnn_comparison"
BASE_EXPERIMENT_PATH = UPath(__file__).parent

# Output paths
RESULTS_PATH = BASE_EXPERIMENT_PATH / "results"
CHECKPOINTS_PATH = BASE_EXPERIMENT_PATH / "checkpoints"
TENSORBOARD_PATH = BASE_EXPERIMENT_PATH / "tensorboard"
VISUALIZATIONS_PATH = BASE_EXPERIMENT_PATH / "visualizations"
METRICS_PATH = BASE_EXPERIMENT_PATH / "metrics"

# ============================================================
# DATASET SETTINGS
# ============================================================

# Classes to train on (14 standard classes)
CLASSES = [
    'ecs', 'pm', 'mito_mem', 'mito_lum', 'mito_ribo',
    'golgi_mem', 'golgi_lum', 'ves_mem', 'ves_lum',
    'endo_mem', 'endo_lum', 'er_mem', 'er_lum', 'nuc'
]

# Data split
VALIDATION_PROB = 0.15
DATASPLIT_PATH = "datasplit.csv"

# ============================================================
# 2D EXPERIMENT SETTINGS
# ============================================================

INPUT_SHAPE_2D = (1, 256, 256)
INPUT_SCALE_2D = (8, 8, 8)

# Training settings for 2D
# Batch sizes optimized for H100 80GB with AMP (BFloat16)
# With mixed precision, memory usage is ~50% lower, allowing larger batches
BATCH_SIZE_2D = {
    'unet': 32,         # H100 + AMP: reduced from 64 to 32 for stability with InstanceNorm
    'resnet': 48,       # H100 + AMP: was 24, now 48
    'swin': 32,         # H100 + AMP: was 16, now 32
    'vit': 16,          # H100 + AMP: was 8, now 16
}

EPOCHS_2D = 100  # Shorter for comparison experiments
ITERATIONS_PER_EPOCH_2D = 1000

# ============================================================
# 3D EXPERIMENT SETTINGS
# ============================================================

INPUT_SHAPE_3D = (32, 256, 256)  # 32 slices
INPUT_SCALE_3D = (8, 8, 8)

# Batch sizes for 3D optimized for H100 80GB with AMP (BFloat16)
# 3D models are significantly more memory-intensive
# Conservative increases to ensure stability
BATCH_SIZE_3D = {
    'unet': 8,          # H100 + AMP: was 4, now 8
    'resnet': 8,        # H100 + AMP: was 4, now 8
    'vit': 2,           # H100 + AMP: was 1, now 2
    'swin': 2,          # H100 + AMP: was 1, now 2 (3D Swin is very memory-heavy)
}

EPOCHS_3D = 50  # Fewer epochs for 3D (slower iterations)
ITERATIONS_PER_EPOCH_3D = 500

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

# Learning rate (default)
LEARNING_RATE = 1e-4

# Model-specific learning rates (override default)
# Lower LR prevents mode collapse with class imbalance
LEARNING_RATE_OVERRIDE = {
    'unet_2d': 5e-5,    # Lower LR for stability with InstanceNorm + Dropout
    'unet_3d': 5e-5,
    'swin_2d': 5e-5,    # Transformers need lower LR to prevent attention collapse
    'swin_3d': 5e-5,
    'vit_2d': 5e-5,
    'vit_3d': 5e-5,
}

# ============================================================
# LOSS FUNCTION SETTINGS
# ============================================================
# Focal Loss + Dice is optimal for extreme class imbalance (14 classes, 7-8 rare)
# Focal Loss down-weights easy examples, focusing on hard negatives
# Dice Loss directly optimizes segmentation overlap

# Use Focal Loss instead of BCE (recommended for all models)
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0  # Focus parameter: higher = more focus on hard examples
FOCAL_ALPHA = 1.0  # Set to 1.0 since pos_weight handles class balance

# Combine with Dice Loss (recommended for all models)
USE_DICE_LOSS = True
DICE_LOSS_WEIGHT = 0.5  # Weight for Dice loss in combined loss

# Legacy: model-specific combined loss (deprecated, use global settings above)
USE_COMBINED_LOSS = {
    'unet_2d': True,
    'unet_3d': True,
}

# Optimizer settings
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)

# Gradient clipping
MAX_GRAD_NORM = 1.0

# Gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 4

# Class weight cap - prevents gradient instability from extremely rare classes
# Set to None to disable capping (use original weights)
# Recommended: 25-50 for stable training, None for aggressive rare-class learning
MAX_CLASS_WEIGHT = 25.0  # Lowered from 50 to reduce gradient variance

# ============================================================
# DATA AUGMENTATION
# ============================================================

SPATIAL_TRANSFORMS_2D = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

SPATIAL_TRANSFORMS_3D = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

# ============================================================
# VISUALIZATION SETTINGS
# ============================================================

# Fixed seed for reproducible sample selection
VISUALIZATION_SEED = 42

# Number of samples to save per epoch
VISUALIZATION_SAMPLES = 3

# Which epochs to save visualizations
VISUALIZATION_EPOCHS = [1, 5, 10, 25, 50, 100]

# ============================================================
# COMPARISON METRICS
# ============================================================

METRICS_TO_TRACK = [
    'bce_loss',           # Binary Cross Entropy
    'dice_loss',          # 1 - Dice coefficient
    'dice_score',         # Dice coefficient per class
    'iou',                # Intersection over Union
    'pixel_accuracy',     # Overall pixel accuracy
    'precision',          # Per-class precision
    'recall',             # Per-class recall
    'f1_score',           # F1 score (same as Dice for binary)
]

# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

# UNet-2D config
UNET_2D_CONFIG = {
    'n_channels': 1,
    'n_classes': len(CLASSES),
    'trilinear': False,
    'use_instancenorm': True,   # InstanceNorm for better small-batch stability
    'dropout': 0.2,             # Regularization to prevent mode collapse
}

# UNet-3D config
UNET_3D_CONFIG = {
    'n_channels': 1,
    'n_classes': len(CLASSES),
    'trilinear': False,
    'use_instancenorm': True,   # InstanceNorm for better small-batch stability
    'dropout': 0.2,             # Regularization to prevent mode collapse
}

# ResNet-2D config
RESNET_2D_CONFIG = {
    'input_nc': 1,
    'output_nc': len(CLASSES),
    'ngf': 64,
    'n_blocks': 6,
    'n_downsampling': 2,
}

# ResNet-3D config
RESNET_3D_CONFIG = {
    'input_nc': 1,
    'output_nc': len(CLASSES),
    'ngf': 64,
    'n_blocks': 6,
    'n_downsampling': 2,
}

# Swin Transformer 2D config
SWIN_2D_CONFIG = {
    'patch_size': [4, 4],
    'embed_dim': 96,
    'depths': [2, 2, 6, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': [7, 7],
    'num_classes': len(CLASSES),
    'drop_rate': 0.1,           # Dropout for regularization
    'attn_drop_rate': 0.1,      # Attention dropout to prevent collapse
    'drop_path_rate': 0.1,      # Stochastic depth for regularization
}

# Swin Transformer 3D config
SWIN_3D_CONFIG = {
    'patch_size': [2, 4, 4],
    'embed_dim': 96,
    'depths': [2, 2, 6, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': [4, 7, 7],
    'num_classes': len(CLASSES),
    'drop_rate': 0.1,           # Dropout for regularization
    'attn_drop_rate': 0.1,      # Attention dropout to prevent collapse
    'drop_path_rate': 0.1,      # Stochastic depth for regularization
}

# ViT-V-Net 3D config
VIT_3D_CONFIG = {
    'out_channels': len(CLASSES),
    'config': 'ViT-V-Net',
    'img_size': INPUT_SHAPE_3D,
}

# ViT-V-Net 2D config
VIT_2D_CONFIG = {
    'img_size': INPUT_SHAPE_2D[1],  # 256
    'patch_size': 16,
    'hidden_size': 768,
    'num_layers': 12,
    'num_heads': 12,
    'mlp_dim': 3072,
    'decoder_channels': (256, 128, 64, 16),
    'dropout_rate': 0.1,
    'attention_dropout_rate': 0.1,  # Attention dropout to prevent collapse
    'down_factor': 2,
}

# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {
    '2d': {
        'unet': {
            'class': 'UNet_2D',
            'config': UNET_2D_CONFIG,
            'batch_size': BATCH_SIZE_2D['unet'],
            'type': 'cnn',
        },
        'resnet': {
            'class': 'ResNet',
            'config': {**RESNET_2D_CONFIG, 'ndims': 2},
            'batch_size': BATCH_SIZE_2D['resnet'],
            'type': 'cnn',
        },
        'swin': {
            'class': 'SwinTransformer',
            'config': SWIN_2D_CONFIG,
            'batch_size': BATCH_SIZE_2D['swin'],
            'type': 'transformer',
        },
        'vit': {
            'class': 'ViTVNet2D',
            'config': VIT_2D_CONFIG,
            'batch_size': BATCH_SIZE_2D['vit'],
            'type': 'transformer',
        },
    },
    '3d': {
        'unet': {
            'class': 'UNet_3D',
            'config': UNET_3D_CONFIG,
            'batch_size': BATCH_SIZE_3D['unet'],
            'type': 'cnn',
        },
        'resnet': {
            'class': 'ResNet',
            'config': {**RESNET_3D_CONFIG, 'ndims': 3},
            'batch_size': BATCH_SIZE_3D['resnet'],
            'type': 'cnn',
        },
        'vit': {
            'class': 'ViTVNet',
            'config': VIT_3D_CONFIG,
            'batch_size': BATCH_SIZE_3D['vit'],
            'type': 'transformer',
        },
        'swin': {
            'class': 'SwinTransformer3D',
            'config': SWIN_3D_CONFIG,
            'batch_size': BATCH_SIZE_3D['swin'],
            'type': 'transformer',
        },
    },
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_model_config(model_name: str, dimension: str) -> dict:
    """Get configuration for a specific model."""
    if dimension not in MODEL_REGISTRY:
        raise ValueError(f"Unknown dimension: {dimension}. Use '2d' or '3d'")
    if model_name not in MODEL_REGISTRY[dimension]:
        raise ValueError(f"Unknown model: {model_name} for {dimension}")
    return MODEL_REGISTRY[dimension][model_name]


def get_input_shape(dimension: str) -> tuple:
    """Get input shape for the specified dimension."""
    if dimension == '2d':
        return INPUT_SHAPE_2D
    elif dimension == '3d':
        return INPUT_SHAPE_3D
    else:
        raise ValueError(f"Unknown dimension: {dimension}")


def get_spatial_transforms(dimension: str) -> dict:
    """Get spatial transforms for the specified dimension."""
    if dimension == '2d':
        return SPATIAL_TRANSFORMS_2D
    elif dimension == '3d':
        return SPATIAL_TRANSFORMS_3D
    else:
        raise ValueError(f"Unknown dimension: {dimension}")


def ensure_dirs_exist():
    """Create all necessary directories."""
    for path in [RESULTS_PATH, CHECKPOINTS_PATH, TENSORBOARD_PATH, 
                 VISUALIZATIONS_PATH, METRICS_PATH]:
        path.mkdir(parents=True, exist_ok=True)
