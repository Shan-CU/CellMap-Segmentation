"""
Model Comparison Experiment Package

Compares CNN vs Transformer architectures for cryo-EM segmentation:
- UNet (2D/3D)
- ResNet (2D/3D)
- Swin Transformer (2D)
- ViT-V-Net (3D)

Usage:
    # Train a single model
    python train_comparison.py --model unet --dim 2d --epochs 100
    
    # Train all models (via SLURM)
    sbatch slurm/run_all_alpine.sbatch
    
    # Visualize results
    python visualize_comparison.py --output figures/
    
    # Statistical analysis
    python analyze_results.py --output analysis/
"""

from .config_base import (
    CLASSES,
    MODEL_REGISTRY,
    get_model_config,
    get_input_shape,
    ensure_dirs_exist,
)

from .metrics_tracker import (
    MetricsTracker,
    DiceLoss,
    FocalLoss,
    ModelComparator,
)

from .feature_extractor import (
    FeatureExtractor,
    AttentionExtractor,
    visualize_feature_maps,
)

__all__ = [
    'CLASSES',
    'MODEL_REGISTRY',
    'get_model_config',
    'get_input_shape',
    'ensure_dirs_exist',
    'MetricsTracker',
    'DiceLoss',
    'FocalLoss',
    'ModelComparator',
    'FeatureExtractor',
    'AttentionExtractor',
    'visualize_feature_maps',
]
