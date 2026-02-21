# Model Comparison Experiment: Transformers vs. Traditional CNNs for Cryo-EM Segmentation

## Overview
This experiment compares the performance of transformer-based architectures (Swin Transformer, ViT) against traditional CNN architectures (UNet, ResNet) for cryo-EM/cryo-ET cell segmentation tasks. The goal is to demonstrate that transformers capture long-range dependencies better for cellular structures.

## Models Compared

### Traditional CNN Models (Baselines)
- **UNet-2D**: Classic encoder-decoder with skip connections
- **UNet-3D**: 3D extension with volumetric convolutions  
- **ResNet-2D**: Residual network for 2D slices
- **ResNet-3D**: 3D residual network

### Transformer Models
- **Swin Transformer**: Shifted window attention for 2D images
- **ViT-V-Net**: Vision Transformer with V-Net decoder for 3D volumes

## Metrics Collected

### Training Metrics
- **BCE Loss**: Binary Cross-Entropy with Logits (primary loss)
- **Dice Score**: F1-like overlap metric (1 - Dice Loss)
- **IoU (Jaccard)**: Intersection over Union per class
- **Pixel Accuracy**: Per-pixel classification accuracy

### Validation Metrics
- **Mean Validation Loss**: Averaged across validation set
- **Per-class Dice Scores**: Individual class performance
- **Mean IoU**: Averaged IoU across classes

### Instance Segmentation Metrics (for evaluation)
- **Hausdorff Distance**: Shape boundary accuracy
- **Surface Dice**: Surface-based overlap measure

## Experiment Structure

```
experiments/model_comparison/
├── README.md                          # This file
├── config_base.py                     # Base configuration
├── configs/
│   ├── unet2d_config.py              # UNet 2D config
│   ├── unet3d_config.py              # UNet 3D config  
│   ├── resnet2d_config.py            # ResNet 2D config
│   ├── resnet3d_config.py            # ResNet 3D config
│   ├── swin2d_config.py              # Swin Transformer 2D config
│   └── vit3d_config.py               # ViT-V-Net 3D config
├── train_comparison.py                # Unified training script
├── feature_extractor.py               # Feature map extraction hooks
├── metrics_tracker.py                 # Comprehensive metrics collection
├── visualize_comparison.py            # Generate comparison plots
├── analyze_results.py                 # Statistical analysis
└── slurm/
    ├── run_all_alpine.sbatch         # Run all models on Alpine
    ├── run_all_blanca.sbatch         # Run all models on Blanca
    └── run_single_model.sbatch       # Run single model template
```

## Running the Experiments

### Single Model Training
```bash
python experiments/model_comparison/train_comparison.py --model unet2d --epochs 100
```

### All Models (via SLURM)
```bash
sbatch experiments/model_comparison/slurm/run_all_alpine.sbatch
# or
sbatch experiments/model_comparison/slurm/run_all_blanca.sbatch
```

### Visualize Results
```bash
python experiments/model_comparison/visualize_comparison.py --output figures/
```

## Fixed Samples for Visualization
To ensure fair comparison, we use fixed sample indices for visualization:
- **2D**: Sample index 42 (seed-based selection from validation set)
- **3D**: Sample index 42 (same seed for consistency)

This ensures all models visualize predictions on the exact same input data.

## Expected Outcomes

Based on literature, we expect:
1. **Transformers** to perform better on:
   - Large, complex structures (mitochondria, ER networks)
   - Structures with long-range dependencies
   - Membrane segmentation (continuous surfaces)

2. **CNNs** to perform comparably/better on:
   - Small, localized structures (ribosomes, vesicles)
   - Tasks with limited training data
   - Inference speed

## Citation

If using this comparison framework, please cite:
```
CellMap Segmentation Challenge
Janelia Research Campus, HHMI
```

## References

1. Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
2. Chen, J., et al. "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation." arXiv 2021.
3. Hatamizadeh, A., et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors." MICCAI 2022.
