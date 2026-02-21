# Masking Strategy Experiments

Comprehensive evaluation of different NaN/unannotated pixel masking strategies.

## Quick Start

```bash
# Train all 15 strategies
./run_masking_tuning.sh train

# View results summary
./run_masking_tuning.sh summary

# Generate visualizations
./run_masking_tuning.sh visualize

# Evaluate on all validation datasets
./run_masking_tuning.sh evaluate-all
```

## Evaluation Options

### Standard Validation
The standard `summary` command evaluates all models on the mixed validation set (200 batches sampled from all datasets):

```bash
./run_masking_tuning.sh summary
```

**Output**: Single summary table with overall Dice/Precision/Recall/IoU averaged across all validation data.

### Comprehensive Dataset Evaluation
The `evaluate-all` command tests each model on **every validation dataset separately**:

```bash
# Evaluate all models on all 15 validation datasets
./run_masking_tuning.sh evaluate-all

# Evaluate specific datasets only
./run_masking_tuning.sh evaluate-all --datasets jrc_hela-2 jrc_jurkat-1 jrc_cos7-1a

# Evaluate specific strategies only
./run_masking_tuning.sh evaluate-all --strategies no_mask masksup_r0.3

# Faster evaluation (fewer batches)
./run_masking_tuning.sh evaluate-all --batch_limit 50

# Combine options
./run_masking_tuning.sh evaluate-all \
    --datasets jrc_hela-2 jrc_jurkat-1 \
    --strategies no_mask masksup_r0.3 entropy_mask \
    --batch_limit 100
```

**Available Validation Datasets** (15 total):
- `jrc_cos7-1a`, `jrc_cos7-1b` - COS-7 cells
- `jrc_fly-vnc-1` - Fly ventral nerve cord  
- `jrc_hela-2`, `jrc_hela-3` - HeLa cells
- `jrc_jurkat-1` - Jurkat cells
- `jrc_macrophage-2` - Macrophages
- `jrc_mus-heart-1` - Mouse heart
- `jrc_mus-kidney` - Mouse kidney
- `jrc_mus-liver`, `jrc_mus-liver-zon-1`, `jrc_mus-liver-zon-2` - Mouse liver
- `jrc_sum159-1`, `jrc_sum159-4` - SUM159 cells
- `jrc_ut21-1413-003` - UT21 cells

**Output Files** (in `evaluation_results/`):
- `all_results.json` - Complete raw results for all strategies × datasets
- `overall_summary.csv` - Mean performance across all datasets per strategy
- `per_dataset_summary.csv` - Detailed breakdown: each strategy on each dataset

**When to Use**:
- ✅ Need to understand which strategies generalize best across cell types
- ✅ Want to identify dataset-specific performance patterns
- ✅ Comparing model robustness across different organisms/tissues
- ✅ Selecting final model based on target deployment dataset

**Performance Note**: Full evaluation (~15 strategies × 15 datasets × 200 batches) takes ~2-3 hours. Use `--batch_limit` to trade speed for precision.

## Visualization

Generate prediction overlay images for visual comparison:

```bash
# Default: 20 samples from diverse datasets
./run_masking_tuning.sh visualize

# More samples
./run_masking_tuning.sh visualize --num_samples 50

# Focus on specific dataset
./run_masking_tuning.sh visualize --dataset jrc_hela-2

# Require more classes visible per sample
./run_masking_tuning.sh visualize --min_classes 3
```

**Output**: `visualizations/` directory with per-strategy subdirectories containing overlay images.

## Available Commands

| Command | Description |
|---------|-------------|
| `train` | Train all 15 strategies in parallel (4 per round, 2 per GPU) |
| `train-sequential` | Train all strategies one at a time on single GPU |
| `single <name>` | Train a single strategy |
| `quick-test` | Fast smoke test: 5 epochs × 20 iterations per strategy |
| `resume` | Resume training, skipping completed strategies |
| `summary` | Print comparison table (overall validation metrics) |
| `visualize` | Generate prediction overlay visualizations |
| `evaluate-all` | ★ Comprehensive: test all models on all datasets |
| `tensorboard` | Launch TensorBoard (default port 6007) |
| `status` | Show completion status and best metrics per strategy |
| `clean` | Remove all checkpoints, results, and logs |

## Masking Strategies

All strategies use **BalancedSoftmax Tversky** loss (α=0.6, β=0.4, τ=1.0) - best from class-weighting experiments.

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| `no_mask` | Baseline: NaN→0, all pixels contribute | - |
| `masksup_r0.3` | Mask-supervised reconstruction | mask_ratio=0.3 |
| `masksup_r0.5` | Mask-supervised reconstruction | mask_ratio=0.5 |
| `regional_g8` | Grid-based adaptive weighting | grid_size=8 |
| `regional_g16` | Grid-based adaptive weighting | grid_size=16 |
| `uncertainty_eu` | Epistemic uncertainty (MC-Dropout) | n_passes=10 |
| `uncertainty_au` | Aleatoric uncertainty estimation | - |
| `box_class_mask` | Per-class bounding box masking | box_margin=0.1 |
| `box_class_mask_tight` | Tight bounding box masking | box_margin=0.05 |
| `salient_mask` | Differential FG/BG masking | fg_ratio=0.3, bg_ratio=0.7 |
| `salient_mask_aggressive` | Aggressive differential masking | fg_ratio=0.2, bg_ratio=0.8 |
| `entropy_mask` | Dynamic entropy threshold masking | threshold=0.7 |
| `entropy_mask_strict` | Strict entropy masking | threshold=0.5 |
| `class_presence` | Mask images for absent classes | - |
| `class_presence_strict` | Strict class presence masking | require_all=True |

## Hardware

Optimized for:
- 2× RTX 3090 (25GB each)
- AMD Ryzen 9 5950X (16 cores)
- 252GB RAM

## Metrics

All evaluation modes report:
- **Dice**: 2TP / (2TP + FP + FN)
- **Precision**: TP / (TP + FP)  
- **Recall**: TP / (TP + FN)
- **IoU**: TP / (TP + FP + FN)

Reported both **per-class** and as **mean** across classes.

## Results Interpretation

### Overall Summary
Use `summary` command for quick comparison of strategies on mixed validation data. Good for initial ranking.

### Per-Dataset Analysis  
Use `evaluate-all` command to:
1. **Identify generalization**: Strategies with consistent performance across datasets generalize better
2. **Find specialists**: Some strategies may excel on specific cell types
3. **Detect overfitting**: Large variance across datasets indicates overfitting
4. **Guide deployment**: Select model based on target dataset similarity

### Example Workflow

```bash
# 1. Train all strategies
./run_masking_tuning.sh train

# 2. Quick overview
./run_masking_tuning.sh summary

# 3. Comprehensive evaluation
./run_masking_tuning.sh evaluate-all

# 4. Analyze results
python -c "
import pandas as pd
df = pd.read_csv('evaluation_results/per_dataset_summary.csv')
print(df.groupby('strategy')['dice'].agg(['mean', 'std']).sort_values('mean', ascending=False))
"

# 5. Visualize top performers
./run_masking_tuning.sh visualize --num_samples 30

# 6. Deep dive on specific datasets
./run_masking_tuning.sh evaluate-all --datasets jrc_hela-2 jrc_hela-3 --batch_limit 300
```

## File Structure

```
experiments/masking_strategies/
├── run_masking_tuning.sh          # Main launcher script
├── train.py                       # Training script
├── config.py                      # Configuration (15 strategies)
├── masking_losses.py              # Loss implementations
├── visualize_all_models.py        # Visualization script
├── evaluate_all_datasets.py       # ★ Comprehensive evaluation
├── checkpoints/                   # Model checkpoints
├── results/                       # Training metrics JSON
├── runs/                          # TensorBoard logs
├── visualizations/                # Prediction overlays
└── evaluation_results/            # ★ Per-dataset evaluation results
    ├── all_results.json
    ├── overall_summary.csv
    └── per_dataset_summary.csv
```
