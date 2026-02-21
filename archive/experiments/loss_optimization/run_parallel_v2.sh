#!/bin/bash
# Run 7 loss functions in parallel across 4 GPUs on Shenron (OPTIMIZED)
#
# Hardware: 4× RTX 2080 Ti (11GB), EPYC 7302 (32 threads), 126GB RAM
#
# Optimization strategy:
#   - 1 worker per GPU (4 GPUs × 1 worker = 4 of 32 threads)
#     NOTE: 4 workers OOMs (120GB), 2 workers OOMs (99GB+swap) — each worker
#     forks the full CellMapDataSplit (~1000 zarr datasets) into memory
#     1 worker + persistent_workers + prefetch_factor=2 gives pipeline overlap
#     without the massive memory duplication
#   - batch_size=28 per GPU (fits 11GB with AMP)
#   - Balanced distribution: max 2 sequential losses per GPU
#   - Persistent workers + prefetch for pipeline overlap
#
# Expected time: ~5-7 hours total (was 21+ hours before)
#   - Each loss: ~50-60 min on 1 GPU with 4 workers (was ~10 hours with 1 worker)
#   - Longest GPU runs 2 losses sequentially: ~2 hours
#   - Dataset loading cached after first run per GPU

set -euo pipefail
cd "$(dirname "$0")"

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate cellmap

# Thread controls — prevent OpenMP from fighting with dataloader workers
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

WORKERS_PER_GPU=1
LOG_DIR="$(pwd)/logs_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Loss Comparison Experiment (OPTIMIZED)"
echo "============================================================"
echo "  GPUs: 4× RTX 2080 Ti"
echo "  Workers per GPU: $WORKERS_PER_GPU"
echo "  Batch size: 28 per GPU"
echo "  Losses: 7 total (balanced across 4 GPUs)"
echo "  Log dir: $LOG_DIR"
echo "  Expected time: ~3-4 hours"
echo "============================================================"
echo ""

# GPU 0: baseline_bce + per_class_weighted_focal (2 losses)
echo "[GPU 0] Starting: baseline_bce → per_class_weighted_focal"
(
  CUDA_VISIBLE_DEVICES=0 python -u train_local.py --mode single_loss --loss baseline_bce --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu0_baseline_bce.log"
  echo "[GPU 0] baseline_bce DONE at $(date)"
  CUDA_VISIBLE_DEVICES=0 python -u train_local.py --mode single_loss --loss per_class_weighted_focal --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu0_per_class_weighted_focal.log"
  echo "[GPU 0] per_class_weighted_focal DONE at $(date)"
) &
PID0=$!

# GPU 1: tversky_precision + per_class_tversky_precision (2 losses)
echo "[GPU 1] Starting: tversky_precision → per_class_tversky_precision"
(
  CUDA_VISIBLE_DEVICES=1 python -u train_local.py --mode single_loss --loss tversky_precision --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu1_tversky_precision.log"
  echo "[GPU 1] tversky_precision DONE at $(date)"
  CUDA_VISIBLE_DEVICES=1 python -u train_local.py --mode single_loss --loss per_class_tversky_precision --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu1_per_class_tversky_precision.log"
  echo "[GPU 1] per_class_tversky_precision DONE at $(date)"
) &
PID1=$!

# GPU 2: tversky_precision_mild + per_class_tversky_precision_strong (2 losses)
echo "[GPU 2] Starting: tversky_precision_mild → per_class_tversky_precision_strong"
(
  CUDA_VISIBLE_DEVICES=2 python -u train_local.py --mode single_loss --loss tversky_precision_mild --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu2_tversky_precision_mild.log"
  echo "[GPU 2] tversky_precision_mild DONE at $(date)"
  CUDA_VISIBLE_DEVICES=2 python -u train_local.py --mode single_loss --loss per_class_tversky_precision_strong --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu2_per_class_tversky_precision_strong.log"
  echo "[GPU 2] per_class_tversky_precision_strong DONE at $(date)"
) &
PID2=$!

# GPU 3: tversky_precision_strong (1 loss — lightest load)
echo "[GPU 3] Starting: tversky_precision_strong"
(
  CUDA_VISIBLE_DEVICES=3 python -u train_local.py --mode single_loss --loss tversky_precision_strong --num_workers $WORKERS_PER_GPU 2>&1 | tee "$LOG_DIR/gpu3_tversky_precision_strong.log"
  echo "[GPU 3] tversky_precision_strong DONE at $(date)"
) &
PID3=$!

echo ""
echo "All jobs launched! PIDs: $PID0 $PID1 $PID2 $PID3"
echo "Started at: $(date)"
echo ""
echo "Monitor:"
echo "  GPU usage:  watch -n2 nvidia-smi"
echo "  Per-GPU logs: tail -f $LOG_DIR/gpu*.log"
echo "  Progress:   grep -h 'Epoch\|Val Dice\|Saved\|DONE' $LOG_DIR/gpu*.log"
echo ""

# Wait for all to complete
wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "============================================================"
echo "  ALL 7 LOSS EXPERIMENTS COMPLETED at $(date)"
echo "============================================================"
echo ""

# Print summary
echo "Results summary:"
python3 -c "
import json, glob
files = sorted(glob.glob('results/*_results.json'))
# Only show results from this run (most recent per loss)
seen = {}
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    seen[d['loss_name']] = d

print(f'{'Loss':<35s} {'Best Dice':>10s} {'Time (min)':>10s}')
print('-' * 57)
for loss_name in ['baseline_bce', 'tversky_precision', 'tversky_precision_mild', 
                   'tversky_precision_strong', 'per_class_tversky_precision',
                   'per_class_tversky_precision_strong', 'per_class_weighted_focal']:
    if loss_name in seen:
        d = seen[loss_name]
        print(f'{loss_name:<35s} {d[\"best_dice\"]:>10.4f} {d[\"elapsed_seconds\"]/60:>10.1f}')
    else:
        print(f'{loss_name:<35s} {'N/A':>10s} {'N/A':>10s}')
"
echo ""
echo "Check results in: results/ and checkpoints/"
