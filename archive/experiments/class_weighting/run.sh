#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Class-Weighting Experiment Launcher
# Fixed loss: per-class Tversky (α=0.6, β=0.4)
# Variable: class weighting strategy
# ──────────────────────────────────────────────────────────────────────
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate environment (nounset temporarily disabled — conda scripts use unbound vars)
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

set -u

export PYTHONPATH="${SCRIPT_DIR}/../..:${SCRIPT_DIR}/../../src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# All 15 loss configs (order: static first, then dynamic)
ALL_LOSSES=(
    weight_uniform weight_manual weight_inv_freq weight_sqrt_inv
    weight_log_inv weight_effective_num
    cb_beta_0.99 cb_beta_0.999 cb_beta_0.9999
    balanced_softmax_tau_0.5 balanced_softmax_tau_1.0 balanced_softmax_tau_2.0
    seesaw_default seesaw_strong_mitigate seesaw_strong_compensate
)

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  compute-freqs        Scan data to compute class frequencies & weight strategies
  quick                Quick smoke-test (5 epochs, single GPU)
  compare              ★ RECOMMENDED: run all 15 configs, 4 at a time (2 per GPU)
  compare-sequential   Run all 15 on one GPU sequentially (slower, simpler)
  single <name>        Run a single config on one GPU
  summary              Print per-class dice comparison table from completed runs
  analyze              Generate plots and summary table
  tensorboard          Launch TensorBoard viewer

Available loss configs:
  ${ALL_LOSSES[*]}
EOF
    exit 1
}

[[ $# -lt 1 ]] && usage

CMD="$1"; shift

case "$CMD" in

    compute-freqs)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Computing class frequencies from training data"
        echo "═══════════════════════════════════════════════════════════"
        python compute_class_frequencies.py "$@"
        ;;

    quick)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Quick test (2 epochs, 4 configs in parallel, 2 per GPU)"
        echo "═══════════════════════════════════════════════════════════"
        QUICK_LOSSES=(weight_uniform cb_beta_0.999 balanced_softmax_tau_1.0 seesaw_default)
        PIDS=()
        echo ""
        echo "─── GPU 0: ${QUICK_LOSSES[0]}, ${QUICK_LOSSES[1]}  |  GPU 1: ${QUICK_LOSSES[2]}, ${QUICK_LOSSES[3]} ───"
        CUDA_VISIBLE_DEVICES=0 python train.py --mode single --loss "${QUICK_LOSSES[0]}" \
            --single_gpu --epochs 2 "$@" &
        PIDS+=($!)
        CUDA_VISIBLE_DEVICES=0 python train.py --mode single --loss "${QUICK_LOSSES[1]}" \
            --single_gpu --epochs 2 "$@" &
        PIDS+=($!)
        CUDA_VISIBLE_DEVICES=1 python train.py --mode single --loss "${QUICK_LOSSES[2]}" \
            --single_gpu --epochs 2 "$@" &
        PIDS+=($!)
        CUDA_VISIBLE_DEVICES=1 python train.py --mode single --loss "${QUICK_LOSSES[3]}" \
            --single_gpu --epochs 2 "$@" &
        PIDS+=($!)
        wait "${PIDS[@]}"
        echo "✅ Quick test complete"
        python train.py --mode summary
        ;;

    compare)
        N_PER_GPU=2
        N_PARALLEL=$((N_PER_GPU * 2))  # 2 GPUs
        N_ROUNDS=$(( (${#ALL_LOSSES[@]} + N_PARALLEL - 1) / N_PARALLEL ))
        echo "═══════════════════════════════════════════════════════════"
        echo "  ★ Parallel comparison: ${N_PARALLEL} configs at a time"
        echo "    (${N_PER_GPU} per GPU × 2 GPUs)"
        echo "    ${#ALL_LOSSES[@]} configs → ${N_ROUNDS} rounds"
        echo "═══════════════════════════════════════════════════════════"
        ROUND=0
        for ((i=0; i<${#ALL_LOSSES[@]}; i+=N_PARALLEL)); do
            ROUND=$((ROUND + 1))
            PIDS=()

            # Gather up to N_PARALLEL losses for this round
            ROUND_LOSSES=()
            for ((j=0; j<N_PARALLEL && i+j<${#ALL_LOSSES[@]}; j++)); do
                ROUND_LOSSES+=("${ALL_LOSSES[$((i+j))]}")
            done

            echo ""
            echo "═══ Round ${ROUND}/${N_ROUNDS}: ${ROUND_LOSSES[*]} ═══"

            # Launch: first N_PER_GPU on GPU 0, rest on GPU 1
            for ((j=0; j<${#ROUND_LOSSES[@]}; j++)); do
                if ((j < N_PER_GPU)); then
                    GPU_ID=0
                else
                    GPU_ID=1
                fi
                echo "    GPU ${GPU_ID} → ${ROUND_LOSSES[$j]}"
                CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --mode single \
                    --loss "${ROUND_LOSSES[$j]}" --single_gpu "$@" &
                PIDS+=($!)
            done

            # Wait for all in this round to finish
            wait "${PIDS[@]}"
        done
        echo ""
        echo "✅ All ${#ALL_LOSSES[@]} configs complete!"
        echo ""
        # Print final comparison table
        python train.py --mode summary
        ;;

    compare-sequential)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Sequential comparison (all 15 on one GPU)"
        echo "═══════════════════════════════════════════════════════════"
        python train.py --mode weighting_comparison --single_gpu "$@"
        ;;

    single)
        LOSS_NAME="${1:-weight_uniform}"
        shift || true
        echo "═══════════════════════════════════════════════════════════"
        echo "  Single config: ${LOSS_NAME}"
        echo "═══════════════════════════════════════════════════════════"
        python train.py --mode single --loss "$LOSS_NAME" --single_gpu "$@"
        ;;

    summary)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Per-class dice comparison table"
        echo "═══════════════════════════════════════════════════════════"
        python train.py --mode summary
        ;;

    analyze)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Generating analysis plots & summary"
        echo "═══════════════════════════════════════════════════════════"
        python analyze_results.py "$@"
        ;;

    tensorboard)
        echo "═══════════════════════════════════════════════════════════"
        echo "  TensorBoard — http://localhost:6008"
        echo "═══════════════════════════════════════════════════════════"
        tensorboard --logdir runs --port 6008 --bind_all
        ;;

    *)
        echo "Unknown command: $CMD"
        usage
        ;;
esac
