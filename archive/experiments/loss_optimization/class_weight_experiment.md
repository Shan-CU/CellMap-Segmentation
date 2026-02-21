# Class Weight Experiment — Copilot Directive

## Objective

Run a systematic class weight comparison using **per-class Tversky loss (α=0.6, β=0.4)** on the CellMap segmentation task. This experiment is complementary to the loss function comparison running on Shenron (which found mild Tversky α=0.6 to be the best loss). Here we fix the loss function and vary the **class weighting strategy** to find the optimal way to balance learning across classes.

## Context

- **Repository:** `https://github.com/Shan-CU/CellMap-Segmentation.git` (branch: `main`)
- **Training code:** `experiments/loss_optimization/train_local.py` (already has normalization fix from commit `5f61b19`)
- **Loss code:** `experiments/loss_optimization/losses.py` (PerClassTverskyLoss class)
- **Config template:** `experiments/loss_optimization/config_shenron.py`
- **5 classes:** `nuc`, `mito_mem`, `er_mem`, `pm`, `golgi_mem`
- **Architecture:** UNet 2D (single Z-slice input, 256×256)
- **Training:** 60 epochs, 100 iterations/epoch, lr=1e-4, AMP enabled

## Critical: Known Bugs Already Fixed

The normalization fix is already in the repo. Do NOT modify `train_local.py`'s normalization code. The fix uses `_normalize_to_float32()` with explicit `/255.0` division and `random_validation=True`. These are correct.

## Step-by-Step Instructions

### 1. Clone and Setup Environment

```bash
git clone https://github.com/Shan-CU/CellMap-Segmentation.git
cd CellMap-Segmentation
git checkout main

# Create conda/micromamba environment
micromamba create -n cellmap python=3.11 -y
micromamba activate cellmap
pip install -e .
pip install tensorboard
```

Ensure the CellMap data is accessible. The `datasplit.csv` in `experiments/loss_optimization/` points to zarr files. If the data is at a different path on this machine, create a symlink or update the zarr paths in `datasplit.csv`.

### 2. Detect Hardware

Run this to determine GPU count, VRAM, CPU threads, and RAM:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nproc
free -h
```

Use these values to set batch size and num_workers in the config (Step 3). Guidelines:
- **Batch size:** 28 for 11GB GPUs, 48 for 24GB GPUs, 64 for 40GB+ GPUs (UNet 2D with AMP)
- **num_workers:** Use 1 per GPU for parallel runs. The CellMapDataSplit loads ~1000 zarr dataset handles into memory. Each forked worker duplicates this (~6GB per worker). On Shenron (126GB RAM), `num_workers=2` caused OOM with 4 parallel jobs. Be conservative.
- **Parallel jobs:** One loss per GPU, run sequentially on GPUs that have more than one loss assigned

### 3. Create Machine-Specific Config

Create `experiments/loss_optimization/config_<machine_name>.py` by copying `config_shenron.py` and modifying:

- `N_GPUS`, `GPU_MEMORY_GB`, `TOTAL_CPU_THREADS` 
- `REPO_ROOT`, `DATA_ROOT` paths
- `batch_size` in `MODEL_CONFIGS`
- Leave `LOSS_CONFIGS` as-is (we'll add new ones)

### 4. Compute Class Frequencies from Training Data

Create and run `experiments/loss_optimization/compute_class_frequencies.py`:

```python
#!/usr/bin/env python3
"""Compute per-class positive pixel frequencies from training data."""
import os, sys
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['SINGLE_GPU_MODE'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.dirname(__file__))

import torch
import json
import numpy as np
torch.set_num_threads(4)

# Import from whichever config file you created, or use config_shenron
from config_shenron import QUICK_TEST_CLASSES
from train_local import create_dataloaders

classes = QUICK_TEST_CLASSES
print(f"Computing class frequencies for: {classes}")

train_loader, _ = create_dataloaders(
    classes=classes, batch_size=8, iterations_per_epoch=100, input_shape=(1, 256, 256)
)

# Accumulate across many batches for stable estimates
total_pos = {c: 0 for c in classes}
total_valid = {c: 0 for c in classes}

n_batches = 0
for batch_idx, batch in enumerate(train_loader):
    if batch_idx >= 100:  # 100 batches × 8 = 800 samples
        break
    targets = batch['output']
    for i, c in enumerate(classes):
        t = targets[:, i]
        valid = ~t.isnan()
        total_valid[c] += valid.sum().item()
        total_pos[c] += (t[valid] > 0.5).sum().item()
    n_batches += 1
    if batch_idx % 20 == 0:
        print(f"  Batch {batch_idx}/{100}...")

print(f"\nProcessed {n_batches} batches")
print(f"\n{'Class':<12s} {'Pos Pixels':>12s} {'Valid Pixels':>12s} {'Frequency':>10s}")
print("-" * 50)

frequencies = {}
for c in classes:
    freq = total_pos[c] / max(total_valid[c], 1)
    frequencies[c] = freq
    print(f"{c:<12s} {total_pos[c]:>12d} {total_valid[c]:>12d} {freq:>10.6f}")

# Compute weight strategies
print("\n\n=== WEIGHT STRATEGIES ===\n")

# 1. Inverse frequency
median_freq = np.median(list(frequencies.values()))
inv_freq = {c: median_freq / max(f, 1e-8) for c, f in frequencies.items()}
# Clamp to [0.5, 5.0] to avoid extremes
inv_freq = {c: max(0.5, min(5.0, w)) for c, w in inv_freq.items()}
print("Inverse frequency:", {c: round(w, 3) for c, w in inv_freq.items()})

# 2. Sqrt inverse frequency
sqrt_inv = {c: np.sqrt(median_freq / max(f, 1e-8)) for c, f in frequencies.items()}
sqrt_inv = {c: max(0.5, min(3.0, w)) for c, w in sqrt_inv.items()}
print("Sqrt inverse freq:", {c: round(w, 3) for c, w in sqrt_inv.items()})

# 3. Log inverse frequency
log_inv = {c: np.log1p(1.0 / max(f, 1e-8)) / np.log1p(1.0 / median_freq) for c, f in frequencies.items()}
log_inv = {c: max(0.5, min(3.0, w)) for c, w in log_inv.items()}
print("Log inverse freq: ", {c: round(w, 3) for c, w in log_inv.items()})

# 4. Effective number (from "Class-Balanced Loss Based on Effective Number of Samples")
beta = 0.999
eff_num = {}
for c, f in frequencies.items():
    n = total_pos[c]
    effective_n = (1 - beta**n) / (1 - beta) if n > 0 else 1
    eff_num[c] = 1.0 / effective_n
# Normalize so max = 2.0
max_eff = max(eff_num.values())
eff_num = {c: max(0.5, min(3.0, w / max_eff * 2.0)) for c, w in eff_num.items()}
print("Effective number: ", {c: round(w, 3) for c, w in eff_num.items()})

# Save all strategies
results = {
    'frequencies': frequencies,
    'total_pos': total_pos,
    'total_valid': total_valid,
    'weight_strategies': {
        'uniform': {c: 1.0 for c in classes},
        'manual_current': {
            'nuc': 1.8, 'mito_mem': 1.3, 'er_mem': 1.6, 'pm': 1.6, 'golgi_mem': 1.1
        },
        'inverse_frequency': inv_freq,
        'sqrt_inverse_frequency': sqrt_inv,
        'log_inverse_frequency': log_inv,
        'effective_number': eff_num,
    }
}

with open('class_frequencies.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to class_frequencies.json")
```

Run it:
```bash
cd experiments/loss_optimization
python compute_class_frequencies.py
```

This produces `class_frequencies.json` containing all computed weight strategies.

### 5. Add Weight Experiment Loss Configs

Add these entries to the `LOSS_CONFIGS` dict in your config file. The weight values below are **placeholders** — replace them with the actual values from `class_frequencies.json` (Step 4).

```python
# --- Class Weight Experiment Configs ---
# All use per_class_tversky with α=0.6, β=0.4 (mild precision bias)
# Only the class_weights vary

'weight_uniform': {
    'type': 'per_class_tversky',
    'alpha': 0.6,
    'beta': 0.4,
    'class_weights': {'nuc': 1.0, 'mito_mem': 1.0, 'er_mem': 1.0, 'pm': 1.0, 'golgi_mem': 1.0},
    'description': 'Tversky mild (α=0.6) — uniform weights (baseline)',
},

'weight_manual': {
    'type': 'per_class_tversky',
    'alpha': 0.6,
    'beta': 0.4,
    'class_weights': {'nuc': 1.8, 'mito_mem': 1.3, 'er_mem': 1.6, 'pm': 1.6, 'golgi_mem': 1.1},
    'description': 'Tversky mild (α=0.6) — manually tuned weights',
},

'weight_inv_freq': {
    'type': 'per_class_tversky',
    'alpha': 0.6,
    'beta': 0.4,
    'class_weights': {},  # FILL FROM class_frequencies.json → inverse_frequency
    'description': 'Tversky mild (α=0.6) — inverse frequency weights',
},

'weight_sqrt_inv': {
    'type': 'per_class_tversky',
    'alpha': 0.6,
    'beta': 0.4,
    'class_weights': {},  # FILL FROM class_frequencies.json → sqrt_inverse_frequency
    'description': 'Tversky mild (α=0.6) — sqrt inverse frequency weights',
},

'weight_log_inv': {
    'type': 'per_class_tversky',
    'alpha': 0.6,
    'beta': 0.4,
    'class_weights': {},  # FILL FROM class_frequencies.json → log_inverse_frequency
    'description': 'Tversky mild (α=0.6) — log inverse frequency weights',
},

'weight_effective_num': {
    'type': 'per_class_tversky',
    'alpha': 0.6,
    'beta': 0.4,
    'class_weights': {},  # FILL FROM class_frequencies.json → effective_number
    'description': 'Tversky mild (α=0.6) — effective number weights',
},
```

### 6. Create the Parallel Runner Script

Create `experiments/loss_optimization/run_weight_experiment.sh`:

```bash
#!/bin/bash
# Class Weight Experiment: Compare 6 weighting strategies
# Loss: per_class_tversky (α=0.6, β=0.4) — best from Shenron loss comparison
# Variable: class_weights only
#
# Adapt GPU assignments based on your hardware.
# Example below is for 4 GPUs — adjust if you have fewer/more.

set -euo pipefail
cd "$(dirname "$0")"

# Activate environment (adjust for your setup)
eval "$(micromamba shell hook --shell bash)"
micromamba activate cellmap

# Thread controls
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

WORKERS_PER_GPU=1  # DO NOT increase — causes OOM from forked dataset copies
LOG_DIR="$(pwd)/logs_weight_exp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Class Weight Experiment"
echo "============================================================"
echo "  Loss: per_class_tversky (α=0.6, β=0.4)"
echo "  Variable: class weighting strategy"
echo "  Strategies: uniform, manual, inv_freq, sqrt_inv, log_inv, effective_num"
echo "  Log dir: $LOG_DIR"
echo "============================================================"
echo ""

# ---- ADAPT THIS SECTION TO YOUR GPU COUNT ----
# 6 strategies across N GPUs. Assign sequentially.
# For 4 GPUs: 2 GPUs run 2 strategies, 2 GPUs run 1 strategy
# For 2 GPUs: 3 strategies each
# For 1 GPU: all 6 sequential

N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Detected $N_GPUS GPUs"

LOSSES=(weight_uniform weight_manual weight_inv_freq weight_sqrt_inv weight_log_inv weight_effective_num)

if [ "$N_GPUS" -ge 4 ]; then
    # GPU 0: uniform + log_inv
    (
      CUDA_VISIBLE_DEVICES=0 python -u train_local.py --mode single_loss --loss weight_uniform --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu0_weight_uniform.log"
      CUDA_VISIBLE_DEVICES=0 python -u train_local.py --mode single_loss --loss weight_log_inv --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu0_weight_log_inv.log"
    ) &
    PID0=$!

    # GPU 1: manual + effective_num
    (
      CUDA_VISIBLE_DEVICES=1 python -u train_local.py --mode single_loss --loss weight_manual --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu1_weight_manual.log"
      CUDA_VISIBLE_DEVICES=1 python -u train_local.py --mode single_loss --loss weight_effective_num --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu1_weight_effective_num.log"
    ) &
    PID1=$!

    # GPU 2: inv_freq
    (
      CUDA_VISIBLE_DEVICES=2 python -u train_local.py --mode single_loss --loss weight_inv_freq --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu2_weight_inv_freq.log"
    ) &
    PID2=$!

    # GPU 3: sqrt_inv
    (
      CUDA_VISIBLE_DEVICES=3 python -u train_local.py --mode single_loss --loss weight_sqrt_inv --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu3_weight_sqrt_inv.log"
    ) &
    PID3=$!

    wait $PID0 $PID1 $PID2 $PID3

elif [ "$N_GPUS" -ge 2 ]; then
    # GPU 0: uniform, inv_freq, log_inv
    (
      for loss in weight_uniform weight_inv_freq weight_log_inv; do
        CUDA_VISIBLE_DEVICES=0 python -u train_local.py --mode single_loss --loss $loss --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu0_${loss}.log"
      done
    ) &
    PID0=$!

    # GPU 1: manual, sqrt_inv, effective_num
    (
      for loss in weight_manual weight_sqrt_inv weight_effective_num; do
        CUDA_VISIBLE_DEVICES=1 python -u train_local.py --mode single_loss --loss $loss --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu1_${loss}.log"
      done
    ) &
    PID1=$!

    wait $PID0 $PID1

else
    # Single GPU: run all sequentially
    for loss in "${LOSSES[@]}"; do
      CUDA_VISIBLE_DEVICES=0 python -u train_local.py --mode single_loss --loss $loss --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu0_${loss}.log"
    done
fi

echo ""
echo "============================================================"
echo "  CLASS WEIGHT EXPERIMENT COMPLETED at $(date)"
echo "============================================================"
```

Make it executable:
```bash
chmod +x run_weight_experiment.sh
```

### 7. Run the Experiment

```bash
cd experiments/loss_optimization
tmux new -s weight_exp './run_weight_experiment.sh 2>&1 | tee weight_experiment.log'
```

### 8. Monitor Progress

```bash
# Attach to tmux
tmux attach -t weight_exp

# Check all Val Dice scores
LOG_DIR=$(ls -td logs_weight_exp_* | head -1)
for f in "$LOG_DIR"/gpu*.log; do
  echo "=== $(basename $f) ==="
  grep "Val Dice" "$f"
  echo ""
done

# System health
free -h && nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

### 9. Expected Outcomes

| Strategy | Expected behavior | Why |
|---|---|---|
| **uniform** | Should match unweighted tversky_precision_mild from Shenron (~0.15+) | Same loss, weights are all 1.0 |
| **manual** | Slight improvement over uniform | Moderate boost to hard classes |
| **inverse_frequency** | Could help or hurt — depends on how extreme | Directly compensates for class imbalance |
| **sqrt_inverse_frequency** | Likely best data-driven option | Softer than raw inverse — less risk of instability |
| **log_inverse_frequency** | Similar to sqrt, very moderate | Most conservative data-driven approach |
| **effective_number** | Good if there's sample overlap | Accounts for data overlap, less aggressive |

### 10. Collect Results

After all runs complete, results JSONs are saved to `experiments/loss_optimization/results/`. Compare Val Dice across strategies:

```bash
python3 -c "
import json, glob
files = sorted(glob.glob('results/*weight_*_results.json'))
results = {}
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    results[d['loss_name']] = d

print(f\"{'Strategy':<30s} {'Best Dice':>10s} {'Best Epoch':>10s}\")
print('-' * 52)
for name in ['weight_uniform', 'weight_manual', 'weight_inv_freq',
             'weight_sqrt_inv', 'weight_log_inv', 'weight_effective_num']:
    if name in results:
        d = results[name]
        print(f\"{name:<30s} {d['best_dice']:>10.4f} {d.get('best_epoch', 'N/A'):>10}\")
"
```

## Summary

This experiment answers: **"Given Tversky α=0.6 as the loss function, what class weighting strategy maximizes Dice?"**

Combined with Shenron's result (best loss function), the final optimal config will be:
- **Loss:** per_class_tversky, α=0.6, β=0.4
- **Weights:** winner from this experiment
- **Architecture:** UNet 2D (or 2.5D if that experiment is also run)
