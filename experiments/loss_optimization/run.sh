#!/bin/bash
# Quick launch scripts for Shenron
# Make sure to activate environment first: source ~/miniforge3/bin/activate cellmap

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$1" in
    test)
        echo "Running setup test..."
        python test_setup.py
        ;;
    
    quick)
        echo "Running quick test (5 epochs, ~5 min)..."
        MODEL=${2:-unet_2d}
        echo "Model: $MODEL"
        python train_local.py --mode quick_test --model "$MODEL"
        ;;
    
    quick-ddp)
        echo "Running quick test with DDP on 4 GPUs..."
        MODEL=${2:-unet_2d}
        echo "Model: $MODEL"
        torchrun --nproc_per_node=4 --standalone train_local.py --mode quick_test --model "$MODEL"
        ;;
    
    compare-losses)
        echo "Running loss comparison (~1-2 hours)..."
        MODEL=${2:-unet_2d}
        echo "Model: $MODEL"
        torchrun --nproc_per_node=4 --standalone train_local.py --mode loss_comparison --model "$MODEL"
        ;;
    
    compare-models)
        echo "=============================================="
        echo "Running 2D vs 2.5D model comparison"
        echo "=============================================="
        echo ""
        echo "This will train:"
        echo "  1. UNet 2D (single slice input)"
        echo "  2. UNet 2.5D (5 adjacent slices as input)"
        echo ""
        echo "Each model uses all 4 GPUs. Estimated time: 2-4 hours total."
        echo ""
        LOSS=${2:-per_class_weighted}
        echo "Loss function: $LOSS"
        echo ""
        torchrun --nproc_per_node=4 --standalone train_local.py --mode model_comparison --loss "$LOSS"
        ;;
    
    full)
        MODEL=${2:-unet_2d}
        LOSS=${3:-per_class_weighted}
        echo "Running full training with model: $MODEL, loss: $LOSS (~8-12 hours)..."
        torchrun --nproc_per_node=4 --standalone train_local.py --mode full_train --model "$MODEL" --loss "$LOSS"
        ;;
    
    tensorboard)
        echo "Starting TensorBoard on port 6006..."
        tensorboard --logdir=runs --port=6006 --bind_all
        ;;
    
    *)
        echo "Usage: $0 {test|quick|quick-ddp|compare-losses|compare-models|full|tensorboard}"
        echo ""
        echo "Commands:"
        echo "  test              - Verify setup and imports"
        echo "  quick [model]     - Quick test (single GPU, 5 epochs)"
        echo "  quick-ddp [model] - Quick test with DDP (4 GPUs)"
        echo "  compare-losses [model] - Compare loss functions"
        echo "  compare-models [loss]  - Compare 2D vs 2.5D (RECOMMENDED)"
        echo "  full [model] [loss]    - Full training"
        echo "  tensorboard       - Start TensorBoard server"
        echo ""
        echo "Models: unet_2d, unet_25d"
        echo ""
        echo "Losses: baseline_bce, dice_bce, focal, combo, per_class_weighted"
        echo ""
        echo "Examples:"
        echo "  ./run.sh compare-models                    # 2D vs 2.5D with per_class_weighted loss"
        echo "  ./run.sh compare-models dice_bce           # 2D vs 2.5D with dice_bce loss"
        echo "  ./run.sh quick unet_25d                    # Quick 2.5D test"
        echo "  ./run.sh full unet_25d per_class_weighted  # Full 2.5D training"
        ;;
esac
