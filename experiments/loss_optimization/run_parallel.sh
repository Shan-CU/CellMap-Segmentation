#!/bin/bash
# Run 7 loss functions in parallel across 4 GPUs on Shenron
# Each GPU runs 1-2 losses sequentially, but all 4 GPUs run in parallel

cd "$(dirname "$0")"

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate cellmap

echo "🚀 Starting parallel loss comparison on 4 GPUs"
echo "   Each GPU: num_workers=1 (memory-safe), batch_size=28"
echo "   Expected time: ~6-7 hours total for all 7 losses"
echo "   Note: Initial dataset loading takes ~2-3 min per job"
echo ""

# GPU 0: baseline_bce + per_class_weighted_focal (2 losses)
echo "[GPU 0] Starting: baseline_bce, per_class_weighted_focal"
CUDA_VISIBLE_DEVICES=0 python -u train_local.py --mode single_loss --loss baseline_bce --num_workers 1 && \
CUDA_VISIBLE_DEVICES=0 python -u train_local.py --mode single_loss --loss per_class_weighted_focal --num_workers 1 &
PID0=$!

# GPU 1: tversky_precision + per_class_tversky_precision (2 losses)
echo "[GPU 1] Starting: tversky_precision, per_class_tversky_precision"
CUDA_VISIBLE_DEVICES=1 python -u train_local.py --mode single_loss --loss tversky_precision --num_workers 1 && \
CUDA_VISIBLE_DEVICES=1 python -u train_local.py --mode single_loss --loss per_class_tversky_precision --num_workers 1 &
PID1=$!

# GPU 2: tversky_precision_mild + per_class_tversky_precision_strong (2 losses)  
echo "[GPU 2] Starting: tversky_precision_mild, per_class_tversky_precision_strong"
CUDA_VISIBLE_DEVICES=2 python -u train_local.py --mode single_loss --loss tversky_precision_mild --num_workers 1 && \
CUDA_VISIBLE_DEVICES=2 python -u train_local.py --mode single_loss --loss per_class_tversky_precision_strong --num_workers 1 &
PID2=$!

# GPU 3: tversky_precision_strong (1 loss)
echo "[GPU 3] Starting: tversky_precision_strong"
CUDA_VISIBLE_DEVICES=3 python -u train_local.py --mode single_loss --loss tversky_precision_strong --num_workers 1 &
PID3=$!

echo ""
echo "All jobs launched! PIDs: $PID0 $PID1 $PID2 $PID3"
echo "Monitor with: tail -f runs/*/events.* or nvidia-smi"
echo ""

# Wait for all to complete
wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "✅ All 7 loss experiments completed!"
echo "Check results in: checkpoints/ and runs/"
