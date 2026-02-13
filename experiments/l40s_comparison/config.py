"""
Configuration for L40S Model Comparison Experiment

Optimized for UNC Longleaf L40S partition (48GB VRAM, 11-day wall time).
Batch sizes derived from H100 config scaled for 45GB usable VRAM.
Uses Balanced Softmax Tversky loss (winner of class weighting experiment).

Hardware profile (from MONAI Auto3DSeg profiling):
  - L40S: 48GB VRAM (45GB usable), Ada Lovelace arch, TF32 support
  - Partition: l40-gpu, QOS: gpu_access
  - Max wall time: 11 days
  - RAM: 512GB required (200GB proven insufficient for 14-class label expansion)
  - Peak observed RAM: ~222GB (from MONAI Job 30343731)
"""

import torch
from upath import UPath

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

EXPERIMENT_NAME = "l40s_model_comparison"
BASE_EXPERIMENT_PATH = UPath(__file__).parent

RESULTS_PATH = BASE_EXPERIMENT_PATH / "results"
CHECKPOINTS_PATH = BASE_EXPERIMENT_PATH / "checkpoints"
TENSORBOARD_PATH = BASE_EXPERIMENT_PATH / "tensorboard"
VISUALIZATIONS_PATH = BASE_EXPERIMENT_PATH / "visualizations"
METRICS_PATH = BASE_EXPERIMENT_PATH / "metrics"

# ============================================================
# DATASET SETTINGS
# ============================================================

# All 14 classes (full challenge set)
CLASSES = [
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "er_mem", "er_lum", "nuc",
]

VALIDATION_PROB = 0.15
DATASPLIT_PATH = "datasplit.csv"

# Intensity clipping (from MONAI DataAnalyzer stats)
INTENSITY_CLIP_MIN = 10.6
INTENSITY_CLIP_MAX = 194.1

# ============================================================
# 2D EXPERIMENT SETTINGS
# ============================================================

INPUT_SHAPE_2D = (1, 256, 256)
INPUT_SCALE_2D = (8, 8, 8)

# Batch sizes for L40S 48GB with AMP (BFloat16)
# Scaled from H100 80GB values (factor ~0.56)
# Verified against MONAI profiling data
BATCH_SIZE_2D = {
    "unet": 26,    # H100: 32 → L40S: ~18, but 2D UNet is lightweight; conservative
    "resnet": 28,  # H100: 48 → L40S: ~27
    "swin": 18,    # H100: 32 → L40S: ~18
    "vit": 10,     # H100: 16 → L40S: ~9, round up
}

EPOCHS_2D = 200             # Long training — 11-day wall time available
ITERATIONS_PER_EPOCH_2D = 1000

# ============================================================
# 3D EXPERIMENT SETTINGS
# ============================================================

INPUT_SHAPE_3D = (32, 256, 256)
INPUT_SCALE_3D = (8, 8, 8)

# 3D batch sizes for L40S 48GB with AMP
# Much more conservative — 3D models are extremely memory-intensive
BATCH_SIZE_3D = {
    "unet": 5,     # H100: 8 → L40S: ~4-5
    "resnet": 5,   # H100: 8 → L40S: ~4-5
    "swin": 2,     # H100: 2 → L40S: 2 (already minimal)
    "vit": 1,      # H100: 2 → L40S: 1 (most memory-intensive)
}

EPOCHS_3D = 200             # Long training — 11-day wall time available
ITERATIONS_PER_EPOCH_3D = 500

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

LEARNING_RATE = 1e-4

# Model-specific LR overrides (lower for stability)
LEARNING_RATE_OVERRIDE = {
    "unet_2d": 5e-5,
    "unet_3d": 5e-5,
    "swin_2d": 5e-5,
    "swin_3d": 5e-5,
    "vit_2d": 5e-5,
    "vit_3d": 5e-5,
    "resnet_2d": 1e-4,  # ResNet is stable at default LR
    "resnet_3d": 1e-4,
}

# ============================================================
# LOSS FUNCTION SETTINGS (Balanced Softmax Tversky)
# ============================================================
# Winner of 15 class weighting configurations: balanced_softmax_tau_1.0
# Mean Dice: 0.5711

LOSS_TYPE = "balanced_softmax_tversky"

# Tversky parameters (from loss optimization experiment)
TVERSKY_ALPHA = 0.6  # FP weight (penalize false positives more)
TVERSKY_BETA = 0.4   # FN weight

# Balanced Softmax temperature (from class weighting experiment)
BALANCED_SOFTMAX_TAU = 1.0

# Class frequency update interval (batches)
FREQ_UPDATE_INTERVAL = 50

# ============================================================
# OPTIMIZER SETTINGS
# ============================================================

WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
MAX_GRAD_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 4

# ============================================================
# EARLY STOPPING
# ============================================================

EARLY_STOPPING_PATIENCE = 20  # Stop if val Dice doesn't improve for 20 epochs
EARLY_STOPPING_MIN_DELTA = 1e-4  # Minimum improvement to count as "better"

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

VISUALIZATION_SEED = 42
VISUALIZATION_SAMPLES = 3

# ============================================================
# L40S HARDWARE SETTINGS
# ============================================================

L40S_CONFIG = {
    "partition": "l40-gpu",
    "qos": "gpu_access",
    "mem": "512g",
    "max_wall_time": "11-00:00:00",
    "cpus_per_task": 32,
    "mail_user": "gsgeorge@ad.unc.edu",
}

# GPU allocation per model type
# 2D models: 2 GPUs (lightweight, don't need 4)
# 3D CNN models: 2 GPUs  
# Swin 3D: 4 GPUs (most memory-intensive transformer)
# ViT 3D: 2 GPUs
GPU_ALLOCATION = {
    "unet_2d": 2,
    "resnet_2d": 2,
    "swin_2d": 2,
    "vit_2d": 2,
    "unet_3d": 2,
    "resnet_3d": 2,
    "swin_3d": 4,
    "vit_3d": 2,
}

# num_workers per model type
# 3D: num_workers=0 REQUIRED (prevents memory leaks with zarr)
# 2D: num_workers=8 (safe, faster data loading)
NUM_WORKERS = {
    "unet_2d": 8,
    "resnet_2d": 8,
    "swin_2d": 8,
    "vit_2d": 8,
    "unet_3d": 0,
    "resnet_3d": 0,
    "swin_3d": 0,
    "vit_3d": 0,
}

# Gradient accumulation steps per model
# Increase for small-batch 3D models to maintain effective batch size
GRADIENT_ACCUMULATION_OVERRIDE = {
    "swin_3d": 6,   # Effective: 2*6*4GPUs = 48
    "vit_3d": 8,    # Effective: 1*8*2GPUs = 16
    "unet_3d": 4,   # Effective: 5*4*2GPUs = 40
    "resnet_3d": 4,  # Effective: 5*4*2GPUs = 40
}

# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

UNET_2D_CONFIG = {
    "n_channels": 1,
    "n_classes": len(CLASSES),
    "trilinear": False,
    "use_instancenorm": True,
    "dropout": 0.2,
}

UNET_3D_CONFIG = {
    "n_channels": 1,
    "n_classes": len(CLASSES),
    "trilinear": False,
    "use_instancenorm": True,
    "dropout": 0.2,
}

RESNET_2D_CONFIG = {
    "input_nc": 1,
    "output_nc": len(CLASSES),
    "ngf": 64,
    "n_blocks": 6,
    "n_downsampling": 2,
}

RESNET_3D_CONFIG = {
    "input_nc": 1,
    "output_nc": len(CLASSES),
    "ngf": 64,
    "n_blocks": 6,
    "n_downsampling": 2,
}

SWIN_2D_CONFIG = {
    "patch_size": [4, 4],
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": [7, 7],
    "num_classes": len(CLASSES),
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "stochastic_depth_prob": 0.1,
}

SWIN_3D_CONFIG = {
    "patch_size": [2, 4, 4],
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": [4, 7, 7],
    "num_classes": len(CLASSES),
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "stochastic_depth_prob": 0.1,
}

VIT_3D_CONFIG = {
    "out_channels": len(CLASSES),
    "config": "ViT-V-Net",
    "img_size": INPUT_SHAPE_3D,
}

VIT_2D_CONFIG = {
    "config": {
        "img_size": INPUT_SHAPE_2D[1],  # 256
        "patch_size": 16,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
        "decoder_channels": (256, 128, 64, 16),
        "dropout_rate": 0.1,
        "attention_dropout_rate": 0.1,
        "down_factor": 2,
    },
    "in_channels": 1,
    "num_classes": len(CLASSES),
}

# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {
    "2d": {
        "unet": {
            "class": "UNet_2D",
            "config": UNET_2D_CONFIG,
            "batch_size": BATCH_SIZE_2D["unet"],
            "type": "cnn",
        },
        "resnet": {
            "class": "ResNet",
            "config": {**RESNET_2D_CONFIG, "ndims": 2},
            "batch_size": BATCH_SIZE_2D["resnet"],
            "type": "cnn",
        },
        "swin": {
            "class": "SwinTransformer",
            "config": SWIN_2D_CONFIG,
            "batch_size": BATCH_SIZE_2D["swin"],
            "type": "transformer",
        },
        "vit": {
            "class": "ViTVNet2D",
            "config": VIT_2D_CONFIG,
            "batch_size": BATCH_SIZE_2D["vit"],
            "type": "transformer",
        },
    },
    "3d": {
        "unet": {
            "class": "UNet_3D",
            "config": UNET_3D_CONFIG,
            "batch_size": BATCH_SIZE_3D["unet"],
            "type": "cnn",
        },
        "resnet": {
            "class": "ResNet",
            "config": {**RESNET_3D_CONFIG, "ndims": 3},
            "batch_size": BATCH_SIZE_3D["resnet"],
            "type": "cnn",
        },
        "vit": {
            "class": "ViTVNet",
            "config": VIT_3D_CONFIG,
            "batch_size": BATCH_SIZE_3D["vit"],
            "type": "transformer",
        },
        "swin": {
            "class": "SwinTransformer3D",
            "config": SWIN_3D_CONFIG,
            "batch_size": BATCH_SIZE_3D["swin"],
            "type": "transformer",
        },
    },
}

# ============================================================
# METRICS
# ============================================================

METRICS_TO_TRACK = [
    "tversky_loss",
    "dice_score",
    "iou",
    "pixel_accuracy",
    "precision",
    "recall",
    "f1_score",
]

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
    if dimension == "2d":
        return INPUT_SHAPE_2D
    elif dimension == "3d":
        return INPUT_SHAPE_3D
    else:
        raise ValueError(f"Unknown dimension: {dimension}")


def get_spatial_transforms(dimension: str) -> dict:
    """Get spatial transforms for the specified dimension."""
    if dimension == "2d":
        return SPATIAL_TRANSFORMS_2D
    elif dimension == "3d":
        return SPATIAL_TRANSFORMS_3D
    else:
        raise ValueError(f"Unknown dimension: {dimension}")


def get_gpu_count(model_name: str, dimension: str) -> int:
    """Get number of GPUs for a model."""
    key = f"{model_name}_{dimension}"
    return GPU_ALLOCATION.get(key, 2)


def get_num_workers(model_name: str, dimension: str) -> int:
    """Get number of DataLoader workers for a model."""
    key = f"{model_name}_{dimension}"
    return NUM_WORKERS.get(key, 0)


def get_gradient_accumulation(model_name: str, dimension: str) -> int:
    """Get gradient accumulation steps for a model."""
    key = f"{model_name}_{dimension}"
    return GRADIENT_ACCUMULATION_OVERRIDE.get(key, GRADIENT_ACCUMULATION_STEPS)


def ensure_dirs_exist():
    """Create all necessary directories."""
    for path in [
        RESULTS_PATH,
        CHECKPOINTS_PATH,
        TENSORBOARD_PATH,
        VISUALIZATIONS_PATH,
        METRICS_PATH,
    ]:
        path.mkdir(parents=True, exist_ok=True)
