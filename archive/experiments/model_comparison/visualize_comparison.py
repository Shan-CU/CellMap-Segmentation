#!/usr/bin/env python
"""
Visualization Script for Model Comparison Results

Generates comprehensive plots comparing different model architectures:
- Training/validation loss curves
- Per-class Dice scores
- IoU comparisons
- Feature map visualizations
- Statistical analysis

Usage:
    python visualize_comparison.py --output figures/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Colors for model types
MODEL_COLORS = {
    'unet_2d': '#1f77b4',      # Blue
    'unet_3d': '#2ca02c',      # Green
    'resnet_2d': '#ff7f0e',    # Orange
    'resnet_3d': '#d62728',    # Red
    'swin_2d': '#9467bd',      # Purple
    'vit_3d': '#8c564b',       # Brown
}

MODEL_TYPES = {
    'unet_2d': 'CNN',
    'unet_3d': 'CNN',
    'resnet_2d': 'CNN',
    'resnet_3d': 'CNN',
    'swin_2d': 'Transformer',
    'vit_3d': 'Transformer',
}

MODEL_LABELS = {
    'unet_2d': 'UNet (2D)',
    'unet_3d': 'UNet (3D)',
    'resnet_2d': 'ResNet (2D)',
    'resnet_3d': 'ResNet (3D)',
    'swin_2d': 'Swin Transformer (2D)',
    'vit_3d': 'ViT-V-Net (3D)',
}


def load_metrics(metrics_dir: Path) -> Dict[str, Dict]:
    """Load all model metrics from directory."""
    metrics = {}
    for metrics_file in metrics_dir.glob('*_metrics.json'):
        model_name = metrics_file.stem.replace('_metrics', '')
        with open(metrics_file, 'r') as f:
            metrics[model_name] = json.load(f)
    return metrics


def load_per_class_metrics(metrics_dir: Path) -> Dict[str, Dict]:
    """Load per-class metrics from directory."""
    per_class = {}
    for metrics_file in metrics_dir.glob('*_per_class_metrics.json'):
        model_name = metrics_file.stem.replace('_per_class_metrics', '')
        with open(metrics_file, 'r') as f:
            per_class[model_name] = json.load(f)
    return per_class


def plot_loss_curves(
    metrics: Dict[str, Dict],
    output_path: Path,
    title: str = "Training and Validation Loss"
) -> None:
    """Plot loss curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    ax = axes[0]
    for model_name, data in sorted(metrics.items()):
        if 'train_loss_bce' in data and data['train_loss_bce']:
            epochs = data.get('epoch', list(range(len(data['train_loss_bce']))))
            color = MODEL_COLORS.get(model_name, 'gray')
            label = MODEL_LABELS.get(model_name, model_name)
            ax.plot(epochs, data['train_loss_bce'], color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('BCE Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(left=0)
    
    # Validation loss
    ax = axes[1]
    for model_name, data in sorted(metrics.items()):
        if 'val_loss_bce' in data and data['val_loss_bce']:
            epochs = data.get('epoch', list(range(len(data['val_loss_bce']))))
            color = MODEL_COLORS.get(model_name, 'gray')
            label = MODEL_LABELS.get(model_name, model_name)
            ax.plot(epochs, data['val_loss_bce'], color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('BCE Loss', fontsize=12)
    ax.set_title('Validation Loss', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(left=0)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'loss_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'loss_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'loss_curves.png'}")


def plot_dice_curves(
    metrics: Dict[str, Dict],
    output_path: Path,
    title: str = "Dice Score Comparison"
) -> None:
    """Plot Dice score curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training Dice
    ax = axes[0]
    for model_name, data in sorted(metrics.items()):
        if 'train_dice_mean' in data and data['train_dice_mean']:
            epochs = data.get('epoch', list(range(len(data['train_dice_mean']))))
            color = MODEL_COLORS.get(model_name, 'gray')
            label = MODEL_LABELS.get(model_name, model_name)
            ax.plot(epochs, data['train_dice_mean'], color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Training Dice Score', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([0, 1])
    ax.set_xlim(left=0)
    
    # Validation Dice
    ax = axes[1]
    for model_name, data in sorted(metrics.items()):
        if 'val_dice_mean' in data and data['val_dice_mean']:
            epochs = data.get('epoch', list(range(len(data['val_dice_mean']))))
            color = MODEL_COLORS.get(model_name, 'gray')
            label = MODEL_LABELS.get(model_name, model_name)
            ax.plot(epochs, data['val_dice_mean'], color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Validation Dice Score', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([0, 1])
    ax.set_xlim(left=0)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'dice_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'dice_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'dice_curves.png'}")


def plot_cnn_vs_transformer(
    metrics: Dict[str, Dict],
    output_path: Path
) -> None:
    """Create comparison plot between CNN and Transformer models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Separate models by type
    cnn_models = [m for m in metrics.keys() if MODEL_TYPES.get(m) == 'CNN']
    transformer_models = [m for m in metrics.keys() if MODEL_TYPES.get(m) == 'Transformer']
    
    # Colors
    cnn_color = '#1f77b4'
    transformer_color = '#ff7f0e'
    
    # Plot 1: Final validation Dice comparison (bar chart)
    ax = axes[0, 0]
    all_models = list(metrics.keys())
    final_dice = []
    colors = []
    labels = []
    
    for model in all_models:
        if 'val_dice_mean' in metrics[model] and metrics[model]['val_dice_mean']:
            final_dice.append(metrics[model]['val_dice_mean'][-1])
            colors.append(cnn_color if model in cnn_models else transformer_color)
            labels.append(MODEL_LABELS.get(model, model))
    
    if final_dice:
        bars = ax.bar(range(len(final_dice)), final_dice, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Final Dice Score', fontsize=12)
        ax.set_title('Final Validation Dice Score', fontsize=14)
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, val in zip(bars, final_dice):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Legend
    cnn_patch = mpatches.Patch(color=cnn_color, label='CNN')
    trans_patch = mpatches.Patch(color=transformer_color, label='Transformer')
    ax.legend(handles=[cnn_patch, trans_patch], loc='lower right')
    
    # Plot 2: Best validation Dice comparison
    ax = axes[0, 1]
    best_dice = []
    colors = []
    labels = []
    
    for model in all_models:
        if 'val_dice_mean' in metrics[model] and metrics[model]['val_dice_mean']:
            best_dice.append(max(metrics[model]['val_dice_mean']))
            colors.append(cnn_color if model in cnn_models else transformer_color)
            labels.append(MODEL_LABELS.get(model, model))
    
    if best_dice:
        bars = ax.bar(range(len(best_dice)), best_dice, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Best Dice Score', fontsize=12)
        ax.set_title('Best Validation Dice Score', fontsize=14)
        ax.set_ylim([0, 1])
        
        for bar, val in zip(bars, best_dice):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Validation Dice over epochs (grouped)
    ax = axes[1, 0]
    
    # CNN models
    for model in cnn_models:
        if 'val_dice_mean' in metrics[model] and metrics[model]['val_dice_mean']:
            epochs = metrics[model].get('epoch', list(range(len(metrics[model]['val_dice_mean']))))
            ax.plot(epochs, metrics[model]['val_dice_mean'], 
                    color=cnn_color, alpha=0.7, linewidth=2,
                    label=MODEL_LABELS.get(model, model))
    
    # Transformer models
    for model in transformer_models:
        if 'val_dice_mean' in metrics[model] and metrics[model]['val_dice_mean']:
            epochs = metrics[model].get('epoch', list(range(len(metrics[model]['val_dice_mean']))))
            ax.plot(epochs, metrics[model]['val_dice_mean'], 
                    color=transformer_color, alpha=0.7, linewidth=2,
                    label=MODEL_LABELS.get(model, model), linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Dice', fontsize=12)
    ax.set_title('Validation Dice: CNN vs Transformer', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([0, 1])
    
    # Plot 4: Convergence speed comparison
    ax = axes[1, 1]
    
    # Find epoch to reach 90% of best performance
    def epochs_to_threshold(dice_vals, threshold_pct=0.9):
        if not dice_vals:
            return None
        best = max(dice_vals)
        threshold = best * threshold_pct
        for i, val in enumerate(dice_vals):
            if val >= threshold:
                return i + 1
        return len(dice_vals)
    
    convergence = []
    colors = []
    labels = []
    
    for model in all_models:
        if 'val_dice_mean' in metrics[model] and metrics[model]['val_dice_mean']:
            epochs_needed = epochs_to_threshold(metrics[model]['val_dice_mean'])
            if epochs_needed:
                convergence.append(epochs_needed)
                colors.append(cnn_color if model in cnn_models else transformer_color)
                labels.append(MODEL_LABELS.get(model, model))
    
    if convergence:
        bars = ax.bar(range(len(convergence)), convergence, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Epochs to 90% of Best', fontsize=12)
        ax.set_title('Convergence Speed', fontsize=14)
        
        for bar, val in zip(bars, convergence):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('CNN vs Transformer Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'cnn_vs_transformer.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'cnn_vs_transformer.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'cnn_vs_transformer.png'}")


def plot_per_class_comparison(
    per_class: Dict[str, Dict],
    output_path: Path
) -> None:
    """Plot per-class Dice scores for all models."""
    if not per_class:
        print("No per-class metrics available.")
        return
    
    # Get all classes from first model
    first_model = list(per_class.keys())[0]
    if 'val_dice' not in per_class[first_model]:
        print("No per-class validation Dice metrics found.")
        return
    
    classes = list(per_class[first_model]['val_dice'].keys())
    
    # Create heatmap data
    models = list(per_class.keys())
    data = []
    
    for model in models:
        model_data = []
        for cls in classes:
            values = per_class[model]['val_dice'].get(cls, [])
            model_data.append(np.mean(values) if values else 0)
        data.append(model_data)
    
    data = np.array(data)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(classes)):
            text = f'{data[i, j]:.2f}'
            color = 'white' if data[i, j] < 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Dice Score', fontsize=12)
    
    ax.set_title('Per-Class Validation Dice Scores', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path / 'per_class_dice.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'per_class_dice.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'per_class_dice.png'}")


def plot_dimension_comparison(
    metrics: Dict[str, Dict],
    output_path: Path
) -> None:
    """Compare 2D vs 3D model performance."""
    # Separate by dimension
    models_2d = [m for m in metrics.keys() if '2d' in m]
    models_3d = [m for m in metrics.keys() if '3d' in m]
    
    if not models_2d or not models_3d:
        print("Need both 2D and 3D models for dimension comparison.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2D models
    ax = axes[0]
    for model in models_2d:
        if 'val_dice_mean' in metrics[model] and metrics[model]['val_dice_mean']:
            epochs = metrics[model].get('epoch', list(range(len(metrics[model]['val_dice_mean']))))
            color = MODEL_COLORS.get(model, 'gray')
            label = MODEL_LABELS.get(model, model)
            ax.plot(epochs, metrics[model]['val_dice_mean'], 
                    color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Dice', fontsize=12)
    ax.set_title('2D Models', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1])
    
    # 3D models
    ax = axes[1]
    for model in models_3d:
        if 'val_dice_mean' in metrics[model] and metrics[model]['val_dice_mean']:
            epochs = metrics[model].get('epoch', list(range(len(metrics[model]['val_dice_mean']))))
            color = MODEL_COLORS.get(model, 'gray')
            label = MODEL_LABELS.get(model, model)
            ax.plot(epochs, metrics[model]['val_dice_mean'], 
                    color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Dice', fontsize=12)
    ax.set_title('3D Models', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1])
    
    plt.suptitle('2D vs 3D Model Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / '2d_vs_3d.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / '2d_vs_3d.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / '2d_vs_3d.png'}")


def generate_summary_table(
    metrics: Dict[str, Dict],
    output_path: Path
) -> str:
    """Generate a summary table in Markdown format."""
    table = "# Model Comparison Summary\n\n"
    table += "| Model | Type | Best Val Dice | Final Val Dice | Best Val Loss | Epochs |\n"
    table += "|-------|------|---------------|----------------|---------------|--------|\n"
    
    for model_name, data in sorted(metrics.items()):
        model_type = MODEL_TYPES.get(model_name, 'Unknown')
        label = MODEL_LABELS.get(model_name, model_name)
        
        best_dice = max(data.get('val_dice_mean', [0])) if data.get('val_dice_mean') else 'N/A'
        final_dice = data.get('val_dice_mean', [0])[-1] if data.get('val_dice_mean') else 'N/A'
        best_loss = min(data.get('val_loss_bce', [0])) if data.get('val_loss_bce') else 'N/A'
        epochs = len(data.get('epoch', [])) if data.get('epoch') else 'N/A'
        
        if isinstance(best_dice, float):
            best_dice = f"{best_dice:.4f}"
        if isinstance(final_dice, float):
            final_dice = f"{final_dice:.4f}"
        if isinstance(best_loss, float):
            best_loss = f"{best_loss:.4f}"
        
        table += f"| {label} | {model_type} | {best_dice} | {final_dice} | {best_loss} | {epochs} |\n"
    
    # Save table
    with open(output_path / 'summary_table.md', 'w') as f:
        f.write(table)
    
    print(f"Saved: {output_path / 'summary_table.md'}")
    return table


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize model comparison results')
    parser.add_argument('--metrics_dir', type=str, default='metrics',
                        help='Directory containing metrics files')
    parser.add_argument('--output', type=str, default='figures',
                        help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Setup paths
    metrics_dir = Path(args.metrics_dir)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    print(f"Loading metrics from {metrics_dir}...")
    metrics = load_metrics(metrics_dir)
    per_class = load_per_class_metrics(metrics_dir)
    
    if not metrics:
        print("No metrics files found. Run training first.")
        return
    
    print(f"Found metrics for models: {list(metrics.keys())}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_loss_curves(metrics, output_path)
    plot_dice_curves(metrics, output_path)
    plot_cnn_vs_transformer(metrics, output_path)
    plot_per_class_comparison(per_class, output_path)
    plot_dimension_comparison(metrics, output_path)
    
    # Generate summary table
    print("\nGenerating summary table...")
    table = generate_summary_table(metrics, output_path)
    print("\n" + table)
    
    print(f"\nAll visualizations saved to {output_path}")


if __name__ == "__main__":
    main()
