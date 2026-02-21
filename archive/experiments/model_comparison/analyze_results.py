#!/usr/bin/env python
"""
Statistical Analysis Script for Model Comparison

Performs rigorous statistical analysis to compare model performance:
- Paired t-tests between model pairs
- ANOVA for multi-model comparison
- Effect size calculation (Cohen's d)
- Confidence intervals
- Performance profiles

Usage:
    python analyze_results.py --metrics_dir metrics/ --output analysis/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats
import pandas as pd


def load_all_metrics(metrics_dir: Path) -> Dict[str, Dict]:
    """Load metrics from all models."""
    metrics = {}
    for metrics_file in metrics_dir.glob('*_metrics.json'):
        model_name = metrics_file.stem.replace('_metrics', '')
        with open(metrics_file, 'r') as f:
            metrics[model_name] = json.load(f)
    return metrics


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def paired_comparison(
    metrics: Dict[str, Dict],
    metric_name: str = 'val_dice_mean'
) -> pd.DataFrame:
    """
    Perform paired comparisons between all models.
    
    Returns a DataFrame with t-test results and effect sizes.
    """
    models = list(metrics.keys())
    n_models = len(models)
    
    results = []
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model_a = models[i]
            model_b = models[j]
            
            # Get the metric values (use last N epochs for stable comparison)
            values_a = metrics[model_a].get(metric_name, [])
            values_b = metrics[model_b].get(metric_name, [])
            
            if not values_a or not values_b:
                continue
            
            # Use last 20% of epochs for comparison (more stable)
            n_epochs = min(len(values_a), len(values_b))
            n_compare = max(int(n_epochs * 0.2), 5)
            
            values_a = np.array(values_a[-n_compare:])
            values_b = np.array(values_b[-n_compare:])
            
            # Paired t-test (if same length) or independent t-test
            if len(values_a) == len(values_b):
                t_stat, p_value = stats.ttest_rel(values_a, values_b)
            else:
                t_stat, p_value = stats.ttest_ind(values_a, values_b)
            
            # Effect size
            cohens_d = compute_cohens_d(values_a, values_b)
            
            # Confidence interval for difference
            diff = values_a - values_b if len(values_a) == len(values_b) else None
            if diff is not None:
                ci_low, ci_high = stats.t.interval(
                    0.95, len(diff) - 1,
                    loc=np.mean(diff), scale=stats.sem(diff)
                )
            else:
                ci_low, ci_high = np.nan, np.nan
            
            # Determine winner
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            winner = model_a if mean_a > mean_b else model_b
            
            results.append({
                'Model A': model_a,
                'Model B': model_b,
                'Mean A': mean_a,
                'Mean B': mean_b,
                'Difference': mean_a - mean_b,
                't-statistic': t_stat,
                'p-value': p_value,
                "Cohen's d": cohens_d,
                'CI (95%) Low': ci_low,
                'CI (95%) High': ci_high,
                'Significant (p<0.05)': p_value < 0.05,
                'Winner': winner,
            })
    
    return pd.DataFrame(results)


def anova_analysis(
    metrics: Dict[str, Dict],
    metric_name: str = 'val_dice_mean'
) -> Dict:
    """
    Perform one-way ANOVA across all models.
    """
    groups = []
    model_names = []
    
    for model_name, data in metrics.items():
        values = data.get(metric_name, [])
        if values:
            # Use last 20% of epochs
            n_epochs = len(values)
            n_compare = max(int(n_epochs * 0.2), 5)
            groups.append(values[-n_compare:])
            model_names.append(model_name)
    
    if len(groups) < 2:
        return {'error': 'Need at least 2 models for ANOVA'}
    
    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Effect size (eta-squared)
    # Total sum of squares
    all_values = np.concatenate(groups)
    grand_mean = np.mean(all_values)
    ss_total = np.sum((all_values - grand_mean) ** 2)
    
    # Between-group sum of squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    # Post-hoc Tukey HSD (if significant)
    tukey_results = None
    if p_value < 0.05:
        try:
            from scipy.stats import tukey_hsd
            tukey_results = tukey_hsd(*groups)
        except ImportError:
            tukey_results = "Tukey HSD not available"
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'significant': p_value < 0.05,
        'models': model_names,
        'tukey_hsd': tukey_results,
    }


def cnn_vs_transformer_analysis(
    metrics: Dict[str, Dict],
    metric_name: str = 'val_dice_mean'
) -> Dict:
    """
    Statistical comparison between CNN and Transformer model groups.
    """
    cnn_models = ['unet_2d', 'unet_3d', 'resnet_2d', 'resnet_3d']
    transformer_models = ['swin_2d', 'vit_3d']
    
    cnn_values = []
    transformer_values = []
    
    for model_name, data in metrics.items():
        values = data.get(metric_name, [])
        if not values:
            continue
        
        # Use best value for each model
        best_val = max(values)
        
        if model_name in cnn_models:
            cnn_values.append(best_val)
        elif model_name in transformer_models:
            transformer_values.append(best_val)
    
    if not cnn_values or not transformer_values:
        return {'error': 'Need both CNN and Transformer models'}
    
    cnn_arr = np.array(cnn_values)
    trans_arr = np.array(transformer_values)
    
    # Independent t-test
    t_stat, p_value = stats.ttest_ind(cnn_arr, trans_arr)
    
    # Mann-Whitney U test (non-parametric alternative)
    if len(cnn_arr) >= 3 and len(trans_arr) >= 3:
        u_stat, u_pvalue = stats.mannwhitneyu(cnn_arr, trans_arr, alternative='two-sided')
    else:
        u_stat, u_pvalue = np.nan, np.nan
    
    # Effect size
    cohens_d = compute_cohens_d(trans_arr, cnn_arr)  # Positive if Transformer > CNN
    
    return {
        'cnn_models': cnn_models,
        'transformer_models': transformer_models,
        'cnn_mean': np.mean(cnn_arr),
        'cnn_std': np.std(cnn_arr),
        'transformer_mean': np.mean(trans_arr),
        'transformer_std': np.std(trans_arr),
        't_statistic': t_stat,
        'p_value_ttest': p_value,
        'u_statistic': u_stat,
        'p_value_mannwhitney': u_pvalue,
        'cohens_d': cohens_d,
        'transformer_advantage': np.mean(trans_arr) > np.mean(cnn_arr),
    }


def compute_confidence_intervals(
    metrics: Dict[str, Dict],
    metric_name: str = 'val_dice_mean',
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Compute confidence intervals for each model's performance.
    """
    results = []
    
    for model_name, data in metrics.items():
        values = data.get(metric_name, [])
        if not values:
            continue
        
        # Use last 20% of epochs
        n_epochs = len(values)
        n_compare = max(int(n_epochs * 0.2), 5)
        values = np.array(values[-n_compare:])
        
        mean = np.mean(values)
        sem = stats.sem(values)
        ci_low, ci_high = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=sem)
        
        results.append({
            'Model': model_name,
            'Mean': mean,
            'Std': np.std(values),
            'SEM': sem,
            'CI Low': ci_low,
            'CI High': ci_high,
            'Best': np.max(values),
            'Worst': np.min(values),
        })
    
    return pd.DataFrame(results).sort_values('Mean', ascending=False)


def generate_latex_table(comparison_df: pd.DataFrame, output_path: Path) -> str:
    """Generate LaTeX table for publication."""
    latex = comparison_df.to_latex(
        index=False,
        float_format='%.4f',
        escape=False,
        column_format='l' * len(comparison_df.columns)
    )
    
    with open(output_path / 'comparison_table.tex', 'w') as f:
        f.write(latex)
    
    return latex


def generate_analysis_report(
    metrics: Dict[str, Dict],
    output_path: Path
) -> str:
    """Generate comprehensive analysis report."""
    report = ["# Statistical Analysis Report\n"]
    report.append("## Model Comparison: CNN vs Transformer for Cryo-EM Segmentation\n")
    
    # 1. Summary Statistics
    report.append("### 1. Summary Statistics\n")
    ci_df = compute_confidence_intervals(metrics, 'val_dice_mean')
    report.append("#### Validation Dice Score (95% CI)\n")
    report.append(ci_df.to_markdown(index=False))
    report.append("\n")
    
    # 2. Pairwise Comparisons
    report.append("\n### 2. Pairwise Comparisons\n")
    pairwise_df = paired_comparison(metrics, 'val_dice_mean')
    if not pairwise_df.empty:
        # Show only significant comparisons
        report.append("#### Statistically Significant Differences (p < 0.05)\n")
        sig_df = pairwise_df[pairwise_df['Significant (p<0.05)']].copy()
        if not sig_df.empty:
            report.append(sig_df[['Model A', 'Model B', 'Difference', 'p-value', "Cohen's d", 'Winner']].to_markdown(index=False))
        else:
            report.append("No statistically significant differences found.\n")
        report.append("\n")
        
        report.append("#### Full Pairwise Comparison Table\n")
        report.append(pairwise_df[['Model A', 'Model B', 'Mean A', 'Mean B', 'p-value', "Cohen's d"]].to_markdown(index=False))
    report.append("\n")
    
    # 3. ANOVA
    report.append("\n### 3. ANOVA Analysis\n")
    anova_results = anova_analysis(metrics, 'val_dice_mean')
    if 'error' not in anova_results:
        report.append(f"- **F-statistic**: {anova_results['f_statistic']:.4f}\n")
        report.append(f"- **p-value**: {anova_results['p_value']:.4e}\n")
        report.append(f"- **η² (effect size)**: {anova_results['eta_squared']:.4f}\n")
        report.append(f"- **Significant**: {'Yes' if anova_results['significant'] else 'No'}\n")
    report.append("\n")
    
    # 4. CNN vs Transformer
    report.append("\n### 4. CNN vs Transformer Comparison\n")
    cnn_trans = cnn_vs_transformer_analysis(metrics, 'val_dice_mean')
    if 'error' not in cnn_trans:
        report.append(f"#### CNN Models\n")
        report.append(f"- Mean Dice: {cnn_trans['cnn_mean']:.4f} ± {cnn_trans['cnn_std']:.4f}\n")
        report.append(f"\n#### Transformer Models\n")
        report.append(f"- Mean Dice: {cnn_trans['transformer_mean']:.4f} ± {cnn_trans['transformer_std']:.4f}\n")
        report.append(f"\n#### Statistical Test Results\n")
        report.append(f"- **t-statistic**: {cnn_trans['t_statistic']:.4f}\n")
        report.append(f"- **p-value (t-test)**: {cnn_trans['p_value_ttest']:.4e}\n")
        report.append(f"- **Cohen's d**: {cnn_trans['cohens_d']:.4f}\n")
        
        if cnn_trans['transformer_advantage']:
            report.append(f"\n**Conclusion**: Transformer models show {'statistically significant' if cnn_trans['p_value_ttest'] < 0.05 else 'non-significant'} advantage over CNNs.\n")
        else:
            report.append(f"\n**Conclusion**: CNN models show {'statistically significant' if cnn_trans['p_value_ttest'] < 0.05 else 'non-significant'} advantage over Transformers.\n")
    report.append("\n")
    
    # 5. Effect Size Interpretation
    report.append("\n### 5. Effect Size Interpretation Guide\n")
    report.append("| Cohen's d | Interpretation |\n")
    report.append("|-----------|----------------|\n")
    report.append("| < 0.2 | Negligible |\n")
    report.append("| 0.2 - 0.5 | Small |\n")
    report.append("| 0.5 - 0.8 | Medium |\n")
    report.append("| > 0.8 | Large |\n")
    report.append("\n")
    
    # Combine and save
    report_text = "\n".join(report)
    
    with open(output_path / 'analysis_report.md', 'w') as f:
        f.write(report_text)
    
    print(f"Saved analysis report to {output_path / 'analysis_report.md'}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of model comparison')
    parser.add_argument('--metrics_dir', type=str, default='metrics',
                        help='Directory containing metrics files')
    parser.add_argument('--output', type=str, default='analysis',
                        help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    metrics_dir = Path(args.metrics_dir)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    print(f"Loading metrics from {metrics_dir}...")
    metrics = load_all_metrics(metrics_dir)
    
    if not metrics:
        print("No metrics found. Run training first.")
        return
    
    print(f"Found metrics for: {list(metrics.keys())}")
    
    # Generate analysis
    print("\nGenerating analysis report...")
    report = generate_analysis_report(metrics, output_path)
    
    print("\n" + "="*60)
    print(report)
    print("="*60)
    
    # Save comparison tables
    print("\nSaving detailed tables...")
    
    pairwise_df = paired_comparison(metrics)
    pairwise_df.to_csv(output_path / 'pairwise_comparison.csv', index=False)
    
    ci_df = compute_confidence_intervals(metrics)
    ci_df.to_csv(output_path / 'confidence_intervals.csv', index=False)
    
    # Generate LaTeX table
    generate_latex_table(ci_df, output_path)
    
    print(f"\nAll analysis results saved to {output_path}")


if __name__ == "__main__":
    main()
