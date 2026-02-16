#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Experiment 2: Tversky α/β Tuning Launcher
#
# Trains 4 configs that sweep α (FP penalty) while keeping β HIGH
# to address low precision / high recall from Experiment 1.
#
# All use balanced_softmax_tau_1.0 (best weighting from Exp 1).
# Baseline was α=0.6, β=0.4. New configs raise both α and β.
#
# 4 configs = 1 round, 2 per GPU
#
# NOTE: No-mask experiments moved to experiments/masking_strategies/
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

# The 4 Tversky α/β tuning configs
TVERSKY_LOSSES=(
    tversky_a0.6_b0.6
    tversky_a0.7_b0.6
    tversky_a0.8_b0.6
    tversky_a0.8_b0.7
)

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  train            ★ Train all 4 Tversky α/β configs (4 parallel, 2 per GPU)
  train-sequential Train all 4 sequentially on one GPU
  single <name>    Train a single config
  evaluate         Run threshold sweep evaluation on new + old checkpoints
  summary          Print comparison table from completed runs

Available configs:
  ${TVERSKY_LOSSES[*]}
EOF
    exit 1
}

[[ $# -lt 1 ]] && usage

CMD="$1"; shift

case "$CMD" in

    train)
        N_PER_GPU=2
        N_PARALLEL=$((N_PER_GPU * 2))  # 2 GPUs
        N_ROUNDS=$(( (${#TVERSKY_LOSSES[@]} + N_PARALLEL - 1) / N_PARALLEL ))
        echo "═══════════════════════════════════════════════════════════"
        echo "  Tversky α/β Tuning: ${N_PARALLEL} configs at a time"
        echo "    ${#TVERSKY_LOSSES[@]} configs → ${N_ROUNDS} rounds"
        echo "═══════════════════════════════════════════════════════════"

        ROUND=0
        for ((i=0; i<${#TVERSKY_LOSSES[@]}; i+=N_PARALLEL)); do
            ROUND=$((ROUND + 1))
            PIDS=()

            ROUND_LOSSES=()
            for ((j=0; j<N_PARALLEL && i+j<${#TVERSKY_LOSSES[@]}; j++)); do
                ROUND_LOSSES+=("${TVERSKY_LOSSES[$((i+j))]}")
            done

        echo ""
            echo "═══ Round ${ROUND}/${N_ROUNDS}: ${ROUND_LOSSES[*]} ═══"

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

            wait "${PIDS[@]}"
        done

        echo ""
        echo "✅ All ${#TVERSKY_LOSSES[@]} Tversky α/β configs complete!"
        echo ""
        python train.py --mode summary
        ;;

    train-sequential)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Sequential Tversky α/β training (single GPU)"
        echo "═══════════════════════════════════════════════════════════"
        for loss_name in "${TVERSKY_LOSSES[@]}"; do
            echo ""
            echo ">>> Training: ${loss_name}"
            python train.py --mode single --loss "$loss_name" --single_gpu "$@"
        done
        echo "✅ Done"
        python train.py --mode summary
        ;;

    single)
        LOSS_NAME="${1:-tversky_a0.7_b0.6}"
        shift || true
        echo "═══════════════════════════════════════════════════════════"
        echo "  Single config: ${LOSS_NAME}"
        echo "═══════════════════════════════════════════════════════════"
        python train.py --mode single --loss "$LOSS_NAME" --single_gpu "$@"
        ;;

    evaluate)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Threshold sweep evaluation on ALL checkpoints"
        echo "═══════════════════════════════════════════════════════════"
        python evaluate_threshold_sweep.py "$@"
        ;;

    summary)
        echo "═══════════════════════════════════════════════════════════"
        echo "  Summary table"
        echo "═══════════════════════════════════════════════════════════"
        python train.py --mode summary
        ;;

    *)
        echo "Unknown command: $CMD"
        usage
        ;;
esac
