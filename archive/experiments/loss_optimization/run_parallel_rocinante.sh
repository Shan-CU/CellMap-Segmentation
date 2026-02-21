#!/bin/bash
# Run loss comparison experiments in parallel on Rocinante
#
# Hardware: 2× RTX 3090 (24GB VRAM each)
#
# Strategy:
#   - Run 3-4 losses per GPU sequentially
#   - Use single_gpu mode (no DDP) with 1-2 workers
#   - Batch size 32 per GPU (RTX 3090 has 24GB)
#   - Each loss takes ~1-2 hours
#   - Total time: ~3-6 hours (longest GPU)
#
# Usage:
#   ./run_parallel_rocinante.sh

set -euo pipefail
cd "$(dirname "$0")"

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

# Thread controls
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

WORKERS_PER_GPU=1
LOG_DIR="$(pwd)/logs_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Loss Comparison Experiment - Rocinante"
echo "============================================================"
echo "  GPUs: 2× RTX 3090 (24GB)"
echo "  Workers per GPU: $WORKERS_PER_GPU"
echo "  Losses: 7 total (split across 2 GPUs)"
echo "  Log dir: $LOG_DIR"
echo "  Expected time: ~4-6 hours"
echo "============================================================"
echo ""

# GPU 0: Run 4 losses sequentially
echo "[GPU 0] Starting 4 losses: baseline_bce → tversky_precision → tversky_precision_mild → per_class_weighted_focal"
(
  CUDA_VISIBLE_DEVICES=0 python -u train_local.py --mode single_loss --loss baseline_bce --num_workers $WORKERS_PER_GPU --single_gpu 2>&1 | tee "$LOG_DIR/gpu0_baseline_bce.log"
  echo "[GPU 0] baseline_bce DONE at $(date)"
  
  CUDA_VISIBLE_DEVICES=0 python -u train_local.py --mode single_loss --loss tversky_precision --num_workers $WORKERS_PER_GPU --single_gpu 2>&1 | tee "$LOG_DIR/gpu0_tversky_precision.log"
  echo "[GPU 0] tversky_precision DONE at $(date)"
  
  CUDA_VISIBLE_DEVICES=0 python -u train_local.py --mode single_loss --loss tversky_precision_mild --num_workers $WORKERS_PER_GPU --single_gpu 2>&1 | tee "$LOG_DIR/gpu0_tversky_precision_mild.log"
  echo "[GPU 0] tversky_precision_mild DONE at $(date)"
  
  CUDA_VISIBLE_DEVICES=0 python -u train_local.py --mode single_loss --loss per_class_weighted_focal --num_workers $WORKERS_PER_GPU --single_gpu 2>&1 | tee "$LOG_DIR/gpu0_per_class_weighted_focal.log"
  echo "[GPU 0] per_class_weighted_focal DONE at $(date)"
  echo "[GPU 0] ALL DONE at $(date)"
) &
PID0=$!

# GPU 1: Run 3 losses sequentially
echo "[GPU 1] Starting 3 losses: tversky_precision_strong → per_class_tversky_precision → per_class_tversky_precision_strong"
(
  CUDA_VISIBLE_DEVICES=1 python -u train_local.py --mode single_loss --loss tversky_precision_strong --num_workers $WORKERS_PER_GPU --single_gpu 2>&1 | tee "$LOG_DIR/gpu1_tversky_precision_strong.log"
  echo "[GPU 1] tversky_precision_strong DONE at $(date)"
  
  CUDA_VISIBLE_DEVICES=1 python -u train_local.py --mode single_loss --loss per_class_tversky_precision --num_workers $WORKERS_PER_GPU --single_gpu 2>&1 | tee "$LOG_DIR/gpu1_per_class_tversky_precision.log"
  echo "[GPU 1] per_class_tversky_precision DONE at $(date)"
  
  CUDA_VISIBLE_DEVICES=1 python -u train_local.py --mode single_loss --loss per_class_tversky_precision_strong --num_workers $WORKERS_PER_GPU --single_gpu 2>&1 | tee "$LOG_DIR/gpu1_per_class_tversky_precision_strong.log"
  echo "[GPU 1] per_class_tversky_precision_strong DONE at $(date)"
  echo "[GPU 1] ALL DONE at $(date)"
) &
PID1=$!

echo ""
echo "All jobs launched! PIDs: $PID0 $PID1"
echo "Started at: $(date)"
echo ""
echo "Monitor:"
echo "  GPU usage:  watch -n2 nvidia-smi"
echo "  Per-GPU logs: tail -f $LOG_DIR/gpu*.log"
echo "  Progress:   grep -h 'Epoch\|Val Dice\|Saved\|DONE' $LOG_DIR/gpu*.log"
echo "  Data logs:  ls -lh results/*_data_debug.log"
echo ""

# Wait for all to complete
wait $PID0 $PID1

echo ""
echo "============================================================"
echo "  ALL 7 LOSS EXPERIMENTS COMPLETED at $(date)"
echo "============================================================"
echo ""

# Print summary
echo "Results summary:"
python3 -c "
import json, glob
from pathlib import Path

files = sorted(glob.glob('results/*_results.json'))
# Get most recent result per loss
seen = {}
for f in files:
    try:
        with open(f) as fh:
            d = json.load(fh)
        loss_name = d['loss_name']
        if loss_name not in seen or Path(f).stat().st_mtime > Path(seen[loss_name]['file']).stat().st_mtime:
            seen[loss_name] = {**d, 'file': f}
    except:
        pass

print(f'{'Loss':<40s} {'Best Dice':>10s} {'Time (min)':>12s}')
print('-' * 64)

losses = ['baseline_bce', 'tversky_precision', 'tversky_precision_mild', 
          'tversky_precision_strong', 'per_class_tversky_precision',
          'per_class_tversky_precision_strong', 'per_class_weighted_focal']

for loss_name in losses:
    if loss_name in seen:
        d = seen[loss_name]
        dice = d.get('best_dice', 0)
        elapsed = d.get('elapsed_seconds', 0) / 60
        print(f'{loss_name:<40s} {dice:>10.4f} {elapsed:>12.1f}')
    else:
        print(f'{loss_name:<40s} {'N/A':>10s} {'N/A':>12s}')

print()
print('Best loss:', max(seen.items(), key=lambda x: x[1].get('best_dice', 0))[0] if seen else 'N/A')
" 2>/dev/null || echo "Could not parse results - check results/ directory"

echo ""
echo "Detailed results: ls -lh results/"
echo "Data debug logs: ls -lh results/*_data_debug.log"
echo "Model checkpoints: ls -lh checkpoints/"
echo ""
echo "Analyze specific log: python analyze_logs.py results/<run_name>_data_debug.log"
