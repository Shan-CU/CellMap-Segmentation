#!/usr/bin/env python3
"""
Test script to verify the environment and run a minimal training loop.

Usage:
    python test_setup.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    errors = []
    
    # Core
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"    GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")
    
    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("  ✓ TensorBoard")
    except ImportError as e:
        errors.append(f"TensorBoard: {e}")
    
    # CellMap
    try:
        from cellmap_segmentation_challenge.models import UNet_2D
        print("  ✓ cellmap_segmentation_challenge")
    except ImportError as e:
        errors.append(f"cellmap_segmentation_challenge: {e}")
    
    # CellMap data
    try:
        from cellmap_data import CellMapDataLoader
        print("  ✓ cellmap_data")
    except ImportError as e:
        errors.append(f"cellmap_data: {e}")
    
    # Local modules
    try:
        from losses import get_loss_function, DiceBCELoss, FocalLoss
        print("  ✓ Local losses module")
    except ImportError as e:
        errors.append(f"Local losses: {e}")
    
    try:
        from config_shenron import get_config, QUICK_TEST_CLASSES
        print("  ✓ Local config module")
    except ImportError as e:
        errors.append(f"Local config: {e}")
    
    if errors:
        print("\n❌ Import errors:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_data_path():
    """Test data path."""
    print("\nTesting data paths...")
    
    data_path = Path("/volatile/cellmap/data")
    if data_path.exists():
        datasets = list(data_path.iterdir())
        print(f"  ✓ Data path exists: {data_path}")
        print(f"    Found {len(datasets)} datasets:")
        for d in datasets[:5]:
            print(f"      - {d.name}")
        if len(datasets) > 5:
            print(f"      ... and {len(datasets) - 5} more")
        return True
    else:
        print(f"  ❌ Data path not found: {data_path}")
        return False


def test_model_creation():
    """Test model creation for both 2D and 2.5D."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from cellmap_segmentation_challenge.models import UNet_2D
        
        # Test 2D model (1 input channel)
        model_2d = UNet_2D(1, 5)  # 1 input channel, 5 output classes
        x_2d = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            y_2d = model_2d(x_2d)
        
        print(f"  ✓ UNet 2D created")
        print(f"    Input shape: {x_2d.shape}")
        print(f"    Output shape: {y_2d.shape}")
        print(f"    Parameters: {sum(p.numel() for p in model_2d.parameters()):,}")
        
        # Test 2.5D model (5 input channels for adjacent slices)
        model_25d = UNet_2D(5, 5)  # 5 input channels, 5 output classes
        x_25d = torch.randn(2, 5, 256, 256)
        with torch.no_grad():
            y_25d = model_25d(x_25d)
        
        print(f"  ✓ UNet 2.5D created (5 input channels)")
        print(f"    Input shape: {x_25d.shape}")
        print(f"    Output shape: {y_25d.shape}")
        print(f"    Parameters: {sum(p.numel() for p in model_25d.parameters()):,}")
        
        # Note the parameter difference
        params_2d = sum(p.numel() for p in model_2d.parameters())
        params_25d = sum(p.numel() for p in model_25d.parameters())
        print(f"    Parameter increase from 2D to 2.5D: {params_25d - params_2d:,} ({(params_25d/params_2d - 1)*100:.1f}%)")
        
        # Test on GPU
        if torch.cuda.is_available():
            model_25d = model_25d.cuda()
            x_25d = x_25d.cuda()
            with torch.no_grad():
                y_25d = model_25d(x_25d)
            print(f"  ✓ GPU forward pass successful for 2.5D")
        
        return True
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        return False


def test_loss_functions():
    """Test loss functions."""
    print("\nTesting loss functions...")
    
    try:
        import torch
        from losses import (
            DiceLoss, DiceBCELoss, FocalLoss, TverskyLoss,
            ComboLoss, PerClassComboLoss, get_loss_function
        )
        
        # Create dummy data
        pred = torch.randn(2, 5, 64, 64)
        target = (torch.rand(2, 5, 64, 64) > 0.8).float()  # Sparse targets
        
        # Add some NaN (like real data)
        target[0, 2, :, :] = float('nan')
        
        losses = [
            ('DiceLoss', DiceLoss()),
            ('DiceBCELoss', DiceBCELoss()),
            ('FocalLoss', FocalLoss()),
            ('TverskyLoss', TverskyLoss()),
            ('ComboLoss', ComboLoss()),
        ]
        
        for name, loss_fn in losses:
            try:
                loss = loss_fn(pred, target)
                print(f"  ✓ {name}: {loss.item():.4f}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
        
        # Test PerClassComboLoss
        classes = ['a', 'b', 'c', 'd', 'e']
        loss_fn = PerClassComboLoss(classes)
        loss = loss_fn(pred, target)
        print(f"  ✓ PerClassComboLoss: {loss.item():.4f}")
        
        # Test factory function
        loss_fn = get_loss_function('combo')
        loss = loss_fn(pred, target)
        print(f"  ✓ get_loss_function('combo'): {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_memory():
    """Test GPU memory with realistic batch sizes for both 2D and 2.5D."""
    print("\nTesting GPU memory with realistic workloads...")
    
    try:
        import torch
        from cellmap_segmentation_challenge.models import UNet_2D
        
        if not torch.cuda.is_available():
            print("  ⚠ No GPU available, skipping memory test")
            return True
        
        device = torch.device('cuda:0')
        
        # Test 2D with batch size 12
        print("  Testing 2D UNet (batch=12)...")
        model = UNet_2D(1, 14).to(device)  # 14 classes
        batch_size = 12
        x = torch.randn(batch_size, 1, 256, 256, device=device)
        target = torch.rand(batch_size, 14, 256, 256, device=device)
        
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.cuda.amp.GradScaler()
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            y = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        mem_2d = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"    ✓ 2D batch=12: {mem_2d:.2f} GB")
        
        del model, x, target, optimizer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test 2.5D with batch size 8 (5 input channels)
        print("  Testing 2.5D UNet (batch=8, 5 input channels)...")
        model = UNet_2D(5, 14).to(device)  # 5 input channels, 14 classes
        batch_size = 8
        x = torch.randn(batch_size, 5, 256, 256, device=device)  # 5 slices
        target = torch.rand(batch_size, 14, 256, 256, device=device)
        
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.cuda.amp.GradScaler()
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            y = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        mem_25d = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"    ✓ 2.5D batch=8: {mem_25d:.2f} GB")
        
        print(f"\n  Memory summary (2080 Ti has 11GB):")
        print(f"    2D:   {mem_2d:.2f} GB ({11 - mem_2d:.2f} GB free)")
        print(f"    2.5D: {mem_25d:.2f} GB ({11 - mem_25d:.2f} GB free)")
        
        del model, x, target, optimizer
        torch.cuda.empty_cache()
        
        return True
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ❌ Out of memory: {e}")
        print("    Try reducing batch size")
        return False
    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ddp_spawn():
    """Test that DDP can be spawned."""
    print("\nTesting DDP compatibility...")
    
    try:
        import torch.distributed as dist
        import torch.multiprocessing as mp
        
        if not torch.cuda.is_available():
            print("  ⚠ No GPU available, skipping DDP test")
            return True
        
        n_gpus = torch.cuda.device_count()
        print(f"  ✓ Found {n_gpus} GPUs for DDP")
        
        if n_gpus < 2:
            print("  ⚠ Less than 2 GPUs, DDP would run in single-GPU mode")
        else:
            print(f"  ✓ DDP can use all {n_gpus} GPUs")
            print("    Run with: torchrun --nproc_per_node=4 train_local.py")
        
        return True
    except Exception as e:
        print(f"  ❌ DDP test failed: {e}")
        return False


def main():
    print("="*60)
    print("CellMap Loss Optimization - Setup Test")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Data paths", test_data_path()))
    results.append(("Model creation", test_model_creation()))
    results.append(("Loss functions", test_loss_functions()))
    results.append(("GPU memory", test_gpu_memory()))
    results.append(("DDP", test_ddp_spawn()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! Ready to train.")
        print("\nNext steps:")
        print("  1. cd /scratch/users/gest9386/CellMap-Segmentation/experiments/loss_optimization")
        print("  2. python train_local.py --mode quick_test")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix issues before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
