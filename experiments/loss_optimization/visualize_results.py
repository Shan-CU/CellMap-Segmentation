#!/usr/bin/env python3
"""
Visualize and compare results from loss optimization experiments.

Usage:
    python visualize_results.py
    python visualize_results.py --results_dir ./results
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path) -> dict:
    """Load all result files."""
    results = {}
    for f in results_dir.glob('*_results.json'):
        with open(f) as fp:
            data = json.load(fp)
            results[data['loss_name']] = data
    return results


def plot_loss_comparison(results: dict, output_path: Path):
    """Create comparison plot of all loss functions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Plot 1: Training loss over epochs
    ax = axes[0, 0]
    for (name, data), color in zip(results.items(), colors):
        epochs = [h['epoch'] for h in data['history']]
        train_loss = [h['train_loss'] for h in data['history']]
        ax.plot(epochs, train_loss, label=name, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Dice over epochs
    ax = axes[0, 1]
    for (name, data), color in zip(results.items(), colors):
        epochs = [h['epoch'] for h in data['history']]
        dice = [h['dice_mean'] for h in data['history']]
        ax.plot(epochs, dice, label=name, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Dice')
    ax.set_title('Validation Dice')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Best Dice bar chart
    ax = axes[1, 0]
    names = list(results.keys())
    best_dice = [results[n]['best_dice'] for n in names]
    bars = ax.bar(range(len(names)), best_dice, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Best Mean Dice')
    ax.set_title('Best Dice by Loss Function')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, best_dice):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Per-class comparison for best loss
    ax = axes[1, 1]
    best_loss = max(results.keys(), key=lambda k: results[k]['best_dice'])
    best_data = results[best_loss]
    
    # Get final per-class results
    final_results = best_data['history'][-1]['per_class']
    classes = list(final_results.keys())
    dice_values = [final_results[c]['dice'] for c in classes]
    
    bars = ax.bar(range(len(classes)), dice_values)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Dice Score')
    ax.set_title(f'Per-Class Dice ({best_loss})')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Color bars by performance
    for bar, val in zip(bars, dice_values):
        if val < 0.2:
            bar.set_color('red')
        elif val < 0.4:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def plot_per_class_comparison(results: dict, output_path: Path):
    """Plot per-class Dice for each loss function."""
    
    # Get all classes from first result
    first_result = next(iter(results.values()))
    classes = list(first_result['history'][-1]['per_class'].keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(classes))
    width = 0.8 / len(results)
    
    for i, (name, data) in enumerate(results.items()):
        final_per_class = data['history'][-1]['per_class']
        dice_values = [final_per_class[c]['dice'] for c in classes]
        offset = (i - len(results)/2 + 0.5) * width
        bars = ax.bar(x + offset, dice_values, width, label=name)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Dice Score')
    ax.set_title('Per-Class Dice by Loss Function')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved per-class comparison to {output_path}")
    plt.close()


def print_summary(results: dict):
    """Print text summary of results."""
    
    print("\n" + "="*70)
    print("LOSS OPTIMIZATION RESULTS SUMMARY")
    print("="*70)
    
    # Sort by best dice
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['best_dice'],
        reverse=True
    )
    
    print("\nRanking by Best Dice Score:")
    print("-"*70)
    for rank, (name, data) in enumerate(sorted_results, 1):
        desc = data.get('loss_config', {}).get('description', 'N/A')
        print(f"{rank}. {name:25s} Dice={data['best_dice']:.4f}  ({desc})")
    
    print("\n" + "-"*70)
    print("Per-Class Performance (Best Loss: {})".format(sorted_results[0][0]))
    print("-"*70)
    
    best_data = sorted_results[0][1]
    final_per_class = best_data['history'][-1]['per_class']
    
    # Sort classes by dice
    sorted_classes = sorted(
        final_per_class.items(),
        key=lambda x: x[1]['dice'],
        reverse=True
    )
    
    for class_name, metrics in sorted_classes:
        dice = metrics['dice']
        status = "🟢" if dice > 0.4 else ("🟡" if dice > 0.2 else "🔴")
        print(f"  {status} {class_name:15s} Dice={dice:.4f}")
    
    print("\nKey:")
    print("  🟢 Good (>0.4)  🟡 Moderate (0.2-0.4)  🔴 Needs improvement (<0.2)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}")
        print("Run experiments first with: python train_local.py --mode loss_comparison")
        return
    
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found. Run experiments first.")
        return
    
    print(f"Found {len(results)} experiment results")
    
    # Create visualizations
    output_dir = args.results_dir / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    plot_loss_comparison(results, output_dir / 'loss_comparison.png')
    plot_per_class_comparison(results, output_dir / 'per_class_comparison.png')
    
    # Print summary
    print_summary(results)


if __name__ == '__main__':
    main()
