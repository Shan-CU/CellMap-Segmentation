#!/usr/bin/env python3
"""
Analyze and visualize model comparison results.

Generates:
1. Grouped bar chart: per-class Dice for each model
2. Training curves: loss and Dice over epochs
3. Radar/spider chart: per-class performance
4. Parameter efficiency: Dice vs model size
5. Per-class metric heatmap

Usage:
    python analyze_results.py
"""

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Colors for models
MODEL_COLORS = {
    'unet':   '#1f77b4',  # blue
    'resnet': '#ff7f0e',  # orange
    'swin':   '#2ca02c',  # green
    'vit':    '#d62728',  # red
}


def load_results():
    """Load all model comparison result JSONs."""
    results = {}
    for rfile in sorted(RESULTS_DIR.glob("mc_*_results.json")):
        try:
            with open(rfile) as f:
                data = json.load(f)
            name = data.get('model_name', rfile.stem.replace('mc_', '').replace('_results', ''))
            results[name] = data
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def get_best_epoch(data):
    """Get the history entry with the best dice."""
    best_dice = data.get('best_dice', 0)
    history = data.get('history', [])
    for h in history:
        if abs(h.get('dice_mean', 0) - best_dice) < 1e-6:
            return h
    if history:
        return max(history, key=lambda h: h.get('dice_mean', 0))
    return None


def plot_per_class_dice(results):
    """Grouped bar chart: per-class Dice by model."""
    models = sorted(results.keys())
    if not models:
        return

    best_epochs = {m: get_best_epoch(results[m]) for m in models}
    classes = None
    for m in models:
        be = best_epochs[m]
        if be and 'per_class' in be:
            classes = list(be['per_class'].keys())
            break
    if classes is None:
        return

    x = np.arange(len(classes))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, model in enumerate(models):
        be = best_epochs[model]
        if be is None or 'per_class' not in be:
            continue
        vals = []
        for c in classes:
            v = be['per_class'].get(c, {})
            vals.append(v.get('dice', 0) if isinstance(v, dict) else v)
        bars = ax.bar(x + i * width, vals, width,
                      label=model.upper(), color=MODEL_COLORS.get(model, '#999'))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlabel('Class')
    ax.set_ylabel('Dice Score')
    ax.set_title('Per-Class Dice Score by Model (Best Epoch)')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'per_class_dice.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ per_class_dice.png")


def plot_training_curves(results):
    """Loss and Dice curves over epochs for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for model, data in sorted(results.items()):
        history = data.get('history', [])
        if not history:
            continue
        epochs = [h['epoch'] for h in history]
        color = MODEL_COLORS.get(model, '#999')

        # Training loss
        train_loss = [h.get('train_loss', 0) for h in history]
        val_loss = [h.get('val_loss', 0) for h in history]
        ax1.plot(epochs, train_loss, '-', color=color, alpha=0.5, linewidth=0.8)
        ax1.plot(epochs, val_loss, '-', color=color, label=f'{model.upper()}', linewidth=1.5)

        # Dice
        dice = [h.get('dice_mean', 0) for h in history]
        ax2.plot(epochs, dice, '-', color=color, label=f'{model.upper()}', linewidth=1.5)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Validation Loss Over Training')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Dice')
    ax2.set_title('Mean Dice Score Over Training')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'training_curves.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ training_curves.png")


def plot_radar_chart(results):
    """Radar/spider chart: per-class Dice for each model."""
    models = sorted(results.keys())
    best_epochs = {m: get_best_epoch(results[m]) for m in models}

    classes = None
    for m in models:
        be = best_epochs[m]
        if be and 'per_class' in be:
            classes = list(be['per_class'].keys())
            break
    if classes is None:
        return

    N = len(classes)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in models:
        be = best_epochs[model]
        if be is None or 'per_class' not in be:
            continue
        vals = []
        for c in classes:
            v = be['per_class'].get(c, {})
            vals.append(v.get('dice', 0) if isinstance(v, dict) else v)
        vals += vals[:1]
        color = MODEL_COLORS.get(model, '#999')
        ax.plot(angles, vals, '-o', color=color, linewidth=2, label=model.upper(), markersize=4)
        ax.fill(angles, vals, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('Per-Class Dice — Model Comparison', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'radar_chart.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ radar_chart.png")


def plot_param_efficiency(results):
    """Scatter: Dice vs parameter count."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, data in sorted(results.items()):
        params = data.get('trainable_params', 0)
        dice = data.get('best_dice', 0)
        elapsed = data.get('elapsed_min', 0)
        color = MODEL_COLORS.get(model, '#999')

        ax.scatter(params / 1e6, dice, s=max(elapsed / 2, 50), color=color,
                   edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(model.upper(), (params / 1e6, dice),
                   textcoords="offset points", xytext=(10, 5),
                   fontsize=10, fontweight='bold', color=color)

    ax.set_xlabel('Trainable Parameters (M)')
    ax.set_ylabel('Best Mean Dice')
    ax.set_title('Parameter Efficiency\n(bubble size ∝ training time)')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'param_efficiency.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ param_efficiency.png")


def plot_metrics_heatmap(results):
    """Heatmap: per-class metrics (Dice, IoU, Precision, Recall) for all models."""
    models = sorted(results.keys())
    best_epochs = {m: get_best_epoch(results[m]) for m in models}

    classes = None
    for m in models:
        be = best_epochs[m]
        if be and 'per_class' in be:
            classes = list(be['per_class'].keys())
            break
    if classes is None:
        return

    metrics = ['dice', 'iou', 'precision', 'recall']

    for metric in metrics:
        data_matrix = []
        for model in models:
            be = best_epochs[model]
            if be is None or 'per_class' not in be:
                data_matrix.append([0] * len(classes))
                continue
            row = []
            for c in classes:
                v = be['per_class'].get(c, {})
                row.append(v.get(metric, 0) if isinstance(v, dict) else 0)
            data_matrix.append(row)

        data_matrix = np.array(data_matrix)

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(data_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')

        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha='right')
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.upper() for m in models])

        for i in range(len(models)):
            for j in range(len(classes)):
                ax.text(j, i, f'{data_matrix[i, j]:.3f}', ha='center', va='center',
                       fontsize=9, color='white' if data_matrix[i, j] > 0.5 else 'black')

        ax.set_title(f'Per-Class {metric.capitalize()} — Model Comparison')
        fig.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        fig.savefig(PLOTS_DIR / f'heatmap_{metric}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ heatmap_{metric}.png")


def main():
    results = load_results()
    if not results:
        print("No results found in", RESULTS_DIR)
        print("Run training first, then analyze.")
        sys.exit(1)

    print(f"Found {len(results)} model results: {list(results.keys())}")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots:")
    plot_per_class_dice(results)
    plot_training_curves(results)
    plot_radar_chart(results)
    plot_param_efficiency(results)
    plot_metrics_heatmap(results)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == '__main__':
    main()
