#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Experiment 3: Masking Strategy Comparison
#
# All use the same BalancedSoftmax Tversky (α=0.6, β=0.4, τ=1.0)
# -- best from class-weighting experiments.
# Only the NaN/unannotated pixel handling varies.
#
# 16 configs = 4 rounds of 4 (2 per GPU × 2 GPUs)
# ──────────────────────────────────────────────────────────────────────
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

set -u

export PYTHONPATH="${SCRIPT_DIR}/../..:${SCRIPT_DIR}/../../src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# All 16 masking strategy configs
STRATEGIES=(
    no_mask
    masksup_r0.3
    masksup_r0.5
    regional_g8
    regional_g16
    uncertainty_eu
    uncertainty_au
    box_class_mask
    box_class_mask_tight
    salient_mask
    salient_mask_aggressive
    entropy_mask
    entropy_mask_strict
    class_presence
    class_presence_strict
)

# NOTE: uncertainty_eu is more expensive (MC-dropout passes)
# It's listed alongside simpler strategies; if it's too slow,
# run it separately with ./run_masking_tuning.sh single uncertainty_eu

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  train              ★ Train all ${#STRATEGIES[@]} configs (4 parallel per round, 2 per GPU)
  train-sequential   Train all ${#STRATEGIES[@]} sequentially on one GPU
  single <name>      Train a single strategy on one GPU
  quick-test         Smoke test: all strategies, 5 epochs × 20 iter each
  resume             Resume training -- skip already-completed strategies
  summary            Print comparison table (Dice/Prec/Rec/IoU per class)
  visualize          Generate prediction overlays for all trained models
  evaluate-all       ★ Evaluate all models on ALL validation datasets
  tensorboard        Launch TensorBoard for this experiment
  status             Show which strategies are done / pending
  clean              Remove all checkpoints, results, and TensorBoard logs

Available strategies:
  ${STRATEGIES[*]}

Examples:
  $(basename "$0") train                     # full parallel run
  $(basename "$0") single entropy_mask       # just one strategy
  $(basename "$0") single no_mask --epochs 30  # override epochs
  $(basename "$0") quick-test                # fast sanity check
  $(basename "$0") resume                    # continue after interruption
  $(basename "$0") visualize --num_samples 30
  $(basename "$0") evaluate-all              # test on all datasets
  $(basename "$0") evaluate-all --datasets jrc_hela-2 jrc_jurkat-1
  $(basename "$0") status
EOF
    exit 1
}

[[ $# -lt 1 ]] && usage

CMD="$1"; shift

case "$CMD" in

    train)
        N_PER_GPU=2
        N_PARALLEL=$((N_PER_GPU * 2))  # 2 GPUs
        N_ROUNDS=$(( (${#STRATEGIES[@]} + N_PARALLEL - 1) / N_PARALLEL ))
        echo "═══════════════════════════════════════════════════════════"
        echo "  Masking Strategy Comparison"
        echo "    ${#STRATEGIES[@]} strategies, ${N_PARALLEL} parallel → ${N_ROUNDS} rounds"
        echo "═══════════════════════════════════════════════════════════"

        ROUND=0
        for ((i=0; i<${#STRATEGIES[@]}; i+=N_PARALLEL)); do
            ROUND=$((ROUND + 1))
            PIDS=()

            ROUND_STRATS=()
            for ((j=0; j<N_PARALLEL && i+j<${#STRATEGIES[@]}; j++)); do
                ROUND_STRATS+=("${STRATEGIES[$((i+j))]}")
            done

            echo ""
            echo "═══ Round ${ROUND}/${N_ROUNDS}: ${ROUND_STRATS[*]} ═══"

            for ((j=0; j<${#ROUND_STRATS[@]}; j++)); do
                if ((j < N_PER_GPU)); then
                    GPU_ID=0
                else
                    GPU_ID=1
                fi
                echo "    GPU ${GPU_ID} → ${ROUND_STRATS[$j]}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --mode single \
                    --strategy "${ROUND_STRATS[$j]}" --single_gpu "$@" &
                PIDS+=($!)
            done

            wait "${PIDS[@]}"
            echo "  Round ${ROUND} complete."
        done

        echo ""
        echo "✅ All ${#STRATEGIES[@]} masking strategies complete!"
        echo ""
        python train.py --mode summary
        ;;

    train-sequential)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Sequential masking strategy training (single GPU)"
        echo "═══════════════════════════════════════════════════════════"
        for name in "${STRATEGIES[@]}"; do
            echo ""
            echo ">>> Training: ${name}"
            python train.py --mode single --strategy "$name" --single_gpu "$@"
        done
        echo "✅ Done"
        python train.py --mode summary
        ;;

    single)
        STRAT_NAME="${1:-no_mask}"
        shift || true
        echo "═══════════════════════════════════════════════════════════"
        echo "  Single strategy: ${STRAT_NAME}"
        echo "═══════════════════════════════════════════════════════════"
        python train.py --mode single --strategy "$STRAT_NAME" --single_gpu "$@"
        ;;

    quick-test)
        N_PER_GPU=2
        N_PARALLEL=$((N_PER_GPU * 2))  # 2 GPUs
        N_ROUNDS=$(( (${#STRATEGIES[@]} + N_PARALLEL - 1) / N_PARALLEL ))
        echo "═══════════════════════════════════════════════════════════"
        echo "  Quick test: all ${#STRATEGIES[@]} strategies × 5 epochs"
        echo "    ${N_PARALLEL} parallel → ${N_ROUNDS} rounds"
        echo "═══════════════════════════════════════════════════════════"

        ROUND=0
        for ((i=0; i<${#STRATEGIES[@]}; i+=N_PARALLEL)); do
            ROUND=$((ROUND + 1))
            PIDS=()

            ROUND_STRATS=()
            for ((j=0; j<N_PARALLEL && i+j<${#STRATEGIES[@]}; j++)); do
                ROUND_STRATS+=("${STRATEGIES[$((i+j))]}")
            done

            echo ""
            echo "═══ Round ${ROUND}/${N_ROUNDS}: ${ROUND_STRATS[*]} ═══"

            for ((j=0; j<${#ROUND_STRATS[@]}; j++)); do
                if ((j < N_PER_GPU)); then
                    GPU_ID=0
                else
                    GPU_ID=1
                fi
                echo "    GPU ${GPU_ID} → ${ROUND_STRATS[$j]}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --mode single \
                    --strategy "${ROUND_STRATS[$j]}" --single_gpu --epochs 1 --iterations 20 "$@" &
                PIDS+=($!)
            done

            wait "${PIDS[@]}"
            echo "  Round ${ROUND} complete."
        done

        echo ""
        echo "✅ Quick test complete!"
        python train.py --mode summary
        ;;

    resume)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Resuming: skipping already-completed strategies"
        echo "═══════════════════════════════════════════════════════════"
        RESULTS_DIR="${SCRIPT_DIR}/results"
        for name in "${STRATEGIES[@]}"; do
            # Check if any result JSON exists for this strategy
            if ls "${RESULTS_DIR}"/mask_${name}_*_results.json 1>/dev/null 2>&1; then
                echo "  ✓ Skipping ${name} (already done)"
                continue
            fi
            echo ""
            echo ">>> Training: ${name}"
            python train.py --mode single --strategy "$name" --single_gpu "$@"
        done
        echo ""
        echo "✅ Resume complete"
        python train.py --mode summary
        ;;

    summary)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Summary table (Dice / Precision / Recall / IoU per class)"
        echo "═══════════════════════════════════════════════════════════"
        python train.py --mode summary
        ;;

    visualize)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Generating visualizations for all trained models"
        echo "═══════════════════════════════════════════════════════════"
        python visualize_all_models.py "$@"
        ;;

    evaluate-all)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Comprehensive evaluation: all models × all datasets"
        echo "═══════════════════════════════════════════════════════════"
        python evaluate_all_datasets.py "$@"
        ;;

    tensorboard)
        PORT="${1:-6007}"
        echo "═══════════════════════════════════════════════════════════"
        echo "  TensorBoard: http://localhost:${PORT}"
        echo "  Log dir: ${SCRIPT_DIR}/runs"
        echo "  Press Ctrl+C to stop"
        echo "═══════════════════════════════════════════════════════════"
        tensorboard --logdir="${SCRIPT_DIR}/runs" --port="${PORT}" --bind_all
        ;;

    status)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Masking Strategy Status"
        echo "═══════════════════════════════════════════════════════════"
        RESULTS_DIR="${SCRIPT_DIR}/results"
        CKPT_DIR="${SCRIPT_DIR}/checkpoints"
        DONE=0
        PENDING=0
        for name in "${STRATEGIES[@]}"; do
            if ls "${RESULTS_DIR}"/mask_${name}_*_results.json 1>/dev/null 2>&1; then
                # Extract best dice from the JSON
                DICE=$(python -c "
import json, glob
files = sorted(glob.glob('${RESULTS_DIR}/mask_${name}_*_results.json'))
if files:
    with open(files[-1]) as f:
        d = json.load(f)
    print(f\"Dice={d.get('best_dice',0):.4f}  Prec={d.get('best_precision',0):.4f}  Rec={d.get('best_recall',0):.4f}  IoU={d.get('best_iou',0):.4f}\")
" 2>/dev/null || echo "results found")
                echo "  ✅ ${name}: ${DICE}"
                DONE=$((DONE + 1))
            else
                echo "  ⬜ ${name}: pending"
                PENDING=$((PENDING + 1))
            fi
        done
        echo ""
        echo "  Done: ${DONE}/${#STRATEGIES[@]}  |  Pending: ${PENDING}"
        # Disk usage
        if [[ -d "${CKPT_DIR}" ]]; then
            CKPT_SIZE=$(du -sh "${CKPT_DIR}" 2>/dev/null | cut -f1)
            echo "  Checkpoints: ${CKPT_SIZE}"
        fi
        if [[ -d "${SCRIPT_DIR}/runs" ]]; then
            TB_SIZE=$(du -sh "${SCRIPT_DIR}/runs" 2>/dev/null | cut -f1)
            echo "  TensorBoard logs: ${TB_SIZE}"
        fi
        ;;

    clean)
        echo "═══════════════════════════════════════════════════════════"
        echo "  ⚠️  This will DELETE all checkpoints, results, and logs!"
        echo "═══════════════════════════════════════════════════════════"
        read -rp "  Are you sure? (yes/no): " CONFIRM
        if [[ "$CONFIRM" == "yes" ]]; then
            rm -rf "${SCRIPT_DIR}/checkpoints"/*
            rm -rf "${SCRIPT_DIR}/results"/*
            rm -rf "${SCRIPT_DIR}/runs"/*
            rm -rf "${SCRIPT_DIR}/visualizations"
            echo "  🗑️  Cleaned."
        else
            echo "  Cancelled."
        fi
        ;;

    *)
        echo "Unknown command: $CMD"
        usage
        ;;
esac
