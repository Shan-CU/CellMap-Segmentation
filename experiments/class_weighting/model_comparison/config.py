"""
Configuration for Class-Weighted Model Comparison on Blanca Biokem.

Hardware: blanca-biokem A100 nodes (2× A100 80GB each)
Loss: Balanced Softmax Tversky τ=1.0 (best from class_weighting experiment)
Models: UNet 2D, ResNet 2D, Swin 2D, ViT 2D — all at same batch size for
        fair comparison.

Uses the class_weighting train.py data pipeline (NaN-safe normalization,
no Binarize transform, NaNtoNum on raw, etc.).
"""

import json
import os
import torch
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path("/home/spuser/ws/CellMap-Segmentation")
DATA_ROOT = REPO_ROOT / "data"
EXPERIMENT_DIR = Path(__file__).resolve().parent   # model_comparison/

# Parent class_weighting experiment (for loss module + frequency data)
CLASS_WEIGHT_DIR = EXPERIMENT_DIR.parent

CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
TENSORBOARD_DIR = EXPERIMENT_DIR / "runs"
RESULTS_DIR = EXPERIMENT_DIR / "results"
VISUALIZATIONS_DIR = EXPERIMENT_DIR / "visualizations"
METRICS_DIR = EXPERIMENT_DIR / "metrics"
FEATURES_DIR = EXPERIMENT_DIR / "features"

# ============================================================
# CLASSES  (same 5 quick-test classes from class_weighting)
# ============================================================

CLASSES = ['nuc', 'mito_mem', 'er_mem', 'pm', 'golgi_mem']
N_CLASSES = len(CLASSES)

# ============================================================
# ESTIMATED VOXEL COUNTS  (auto-load from class_weighting)
# ============================================================

ESTIMATED_VOXEL_COUNTS = None

_FREQ_FILE = CLASS_WEIGHT_DIR / "class_frequencies.json"
if _FREQ_FILE.exists():
    with open(_FREQ_FILE) as _f:
        _freq_data = json.load(_f)
    if 'estimated_voxel_counts' in _freq_data:
        ESTIMATED_VOXEL_COUNTS = _freq_data['estimated_voxel_counts']
    print(f"[config] Loaded voxel counts from {_FREQ_FILE}")
else:
    print(f"[config] ⚠ {_FREQ_FILE} not found — losses will estimate online")

# ============================================================
# LOSS CONFIGURATION  (best from class_weighting: balanced_softmax_tau_1.0)
# ============================================================

TVERSKY_ALPHA = 0.6
TVERSKY_BETA = 0.4

LOSS_CONFIG = {
    'type': 'balanced_softmax',
    'alpha': TVERSKY_ALPHA,
    'beta': TVERSKY_BETA,
    'tau': 1.0,
    'description': 'Balanced Softmax Tversky τ=1.0 (class_weighting winner)',
}

# ============================================================
# MODEL CONFIGURATIONS
#
# All models use the SAME batch size (24) for fair comparison.
# Input shape: (1, 256, 256) single Z-slice at 8nm isotropic.
# ============================================================

BATCH_SIZE = 24  # Same for all models for fair comparison
INPUT_SHAPE = (1, 256, 256)
INPUT_CHANNELS = 1
SCALE = (8, 8, 8)

# --- UNet 2D ---
UNET_CONFIG = {
    'n_channels': INPUT_CHANNELS,
    'n_classes': N_CLASSES,
}

# --- ResNet 2D ---
RESNET_CONFIG = {
    'input_nc': INPUT_CHANNELS,
    'output_nc': N_CLASSES,
    'ngf': 64,
    'n_blocks': 6,
    'n_downsampling': 2,
    'ndims': 2,
}

# --- Swin Transformer 2D ---
SWIN_CONFIG = {
    'patch_size': [4, 4],
    'embed_dim': 96,
    'depths': [2, 2, 6, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': [7, 7],
    'num_classes': N_CLASSES,
    'dropout': 0.1,
    'attention_dropout': 0.1,
    'stochastic_depth_prob': 0.1,
}

# --- ViT-V-Net 2D ---
VIT_CONFIG = {
    'img_size': INPUT_SHAPE[1],     # 256
    'patch_size': 16,
    'hidden_size': 768,
    'num_layers': 12,
    'num_heads': 12,
    'mlp_dim': 3072,
    'decoder_channels': (256, 128, 64, 16),
    'dropout_rate': 0.1,
    'attention_dropout_rate': 0.1,
    'down_factor': 2,
}

MODEL_REGISTRY = {
    'unet': {
        'class': 'UNet_2D',
        'config': UNET_CONFIG,
        'type': 'cnn',
        'lr': 1e-4,
    },
    'resnet': {
        'class': 'ResNet',
        'config': RESNET_CONFIG,
        'type': 'cnn',
        'lr': 1e-4,
    },
    'swin': {
        'class': 'SwinTransformer',
        'config': SWIN_CONFIG,
        'type': 'transformer',
        'lr': 5e-5,   # Transformers benefit from lower LR
    },
    'vit': {
        'class': 'ViTVNet2D',
        'config': VIT_CONFIG,
        'type': 'transformer',
        'lr': 5e-5,
    },
}

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

TRAINING_CONFIG = {
    'epochs': 100,
    'iterations_per_epoch': 200,
    'validate_every': 1,
    'weight_decay': 1e-4,
    'betas': (0.9, 0.999),
}

# AMP & compile
USE_AMP = True
MAX_GRAD_NORM = 1.0

# DataLoader
NUM_WORKERS = 0       # 0 is optimal for zarr datasets
DATALOADER_CONFIG = {
    'num_workers': NUM_WORKERS,
    'pin_memory': True,
    'persistent_workers': False,
    'prefetch_factor': None,
}

# Validation
VALIDATION_CONFIG = {
    'batch_limit': 20,
}

# Visualization
VISUALIZATION_SEED = 42
VIS_EVERY = 10         # Save visualizations every N epochs
CHECKPOINT_EVERY = 25  # Periodic checkpoints
VIS_SAMPLES = 5        # Fixed samples shared across all models

# ============================================================
# SPATIAL TRANSFORMS  (2D)
# ============================================================

SPATIAL_TRANSFORMS_2D = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

# ============================================================
# HELPERS
# ============================================================

def ensure_dirs():
    for d in [CHECKPOINT_DIR, TENSORBOARD_DIR, RESULTS_DIR,
              VISUALIZATIONS_DIR, METRICS_DIR, FEATURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
