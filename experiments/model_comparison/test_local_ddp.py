#!/usr/bin/env python
"""
Local Test Script for Model Comparison DDP Training

Tests all models with DDP on local hardware (2x RTX 3090 24GB).
Runs quick validation to ensure everything works before cluster submission.

Hardware Detected:
- 2x NVIDIA GeForce RTX 3090 (24GB each)
- Intel Xeon Gold 5220R (96 threads)
- 251GB RAM

Usage:
    # Test single model (single GPU, quick)
    python test_local_ddp.py --model unet --dim 2d
    
    # Test with DDP (2 GPUs)
    torchrun --nproc_per_node=2 test_local_ddp.py --model swin --dim 3d
    
    # Test all models sequentially
    python test_local_ddp.py --all
    
    # Test all models with DDP
    ./test_local_ddp.py --all --ddp

Author: CellMap Segmentation Challenge
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_hardware():
    """Check and display hardware configuration."""
    print("=" * 60)
    print("HARDWARE CONFIGURATION")
    print("=" * 60)
    
    # GPU info
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"GPUs: {n_gpus}")
        total_vram = 0
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            total_vram += vram_gb
            print(f"  GPU {i}: {props.name} ({vram_gb:.1f} GB)")
        print(f"Total VRAM: {total_vram:.1f} GB")
    else:
        print("No CUDA GPUs available!")
        return False
    
    # CPU info
    print(f"CPU cores: {os.cpu_count()}")
    
    # RAM info
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"RAM: {ram_gb:.1f} GB")
    except ImportError:
        pass
    
    print("=" * 60)
    return True


def get_local_batch_sizes():
    """
    Get optimized batch sizes for local RTX 3090 (24GB VRAM each).
    
    With DDP, each GPU processes batch_size samples.
    Effective batch = batch_size * n_gpus * grad_accum
    
    RTX 3090 has less memory than A100 (24GB vs 80GB), so we use smaller batches.
    """
    return {
        # 2D Models - RTX 3090 optimized
        'unet_2d': {
            'batch_size': 16,      # ~8GB VRAM per GPU
            'grad_accum': 4,       # Effective: 16*2*4 = 128
            'iterations': 50,      # Quick test
            'epochs': 3,
        },
        'resnet_2d': {
            'batch_size': 12,      # ~10GB VRAM per GPU
            'grad_accum': 4,
            'iterations': 50,
            'epochs': 3,
        },
        'swin_2d': {
            'batch_size': 8,       # ~14GB VRAM per GPU (transformer)
            'grad_accum': 8,
            'iterations': 50,
            'epochs': 3,
        },
        'vit_2d': {
            'batch_size': 2,       # ViT 2D needs very small batch on RTX 3090
            'grad_accum': 16,      # Effective: 2*2*16 = 64
            'iterations': 50,
            'epochs': 3,
        },
        # 3D Models - Much smaller batches needed for RTX 3090
        # 3D models are VERY memory intensive due to volumetric data
        'unet_3d': {
            'batch_size': 1,       # Reduced for DDP overhead
            'grad_accum': 16,
            'iterations': 20,
            'epochs': 2,
        },
        'resnet_3d': {
            'batch_size': 1,       # Reduced for DDP overhead
            'grad_accum': 16,
            'iterations': 20,
            'epochs': 2,
        },
        'swin_3d': {
            'batch_size': 1,       # Minimum viable
            'grad_accum': 32,
            'iterations': 10,      # Fewer iterations for quick test
            'epochs': 2,
        },
        'vit_3d': {
            'batch_size': 1,       # Minimum viable
            'grad_accum': 32,
            'iterations': 10,      # Fewer iterations for quick test
            'epochs': 2,
        },
    }


def test_model_creation(model_name: str, dim: str):
    """Test that a model can be created and moved to GPU."""
    print(f"\n--- Testing model creation: {model_name} ({dim}) ---")
    
    try:
        from experiments.model_comparison.config_base import (
            get_model_config, get_input_shape, CLASSES
        )
        from experiments.model_comparison.train_comparison import create_model
        
        device = torch.device("cuda:0")
        model = create_model(model_name, dim, device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")
        
        # Test forward pass
        input_shape = get_input_shape(dim)
        batch_size = 1
        
        # 2D models expect (B, C, H, W), 3D models expect (B, C, D, H, W)
        if dim == '2d':
            # input_shape is (1, H, W) for 2D, we need (B, C, H, W)
            x = torch.randn(batch_size, 1, input_shape[1], input_shape[2]).to(device)
        else:
            # input_shape is (D, H, W) for 3D, we need (B, C, D, H, W)
            x = torch.randn(batch_size, 1, *input_shape).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected classes: {len(CLASSES)}")
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"  GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        del model, x, output
        torch.cuda.empty_cache()
        
        print(f"  ✓ Model creation successful")
        return True
        
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ddp_training(model_name: str, dim: str, quick: bool = True):
    """Test DDP training with a model."""
    config = get_local_batch_sizes()[f"{model_name}_{dim}"]
    
    if quick:
        epochs = 2
        iterations = 10
    else:
        epochs = config['epochs']
        iterations = config['iterations']
    
    print(f"\n--- Testing DDP training: {model_name} ({dim}) ---")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {epochs}")
    print(f"  Iterations: {iterations}")
    
    # Build command
    n_gpus = torch.cuda.device_count()
    
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--standalone",
        str(Path(__file__).parent / "train_comparison.py"),
        "--model", model_name,
        "--dim", dim,
        "--epochs", str(epochs),
        "--batch_size", str(config['batch_size']),
        "--debug",  # Uses fewer iterations
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    
    # Set up environment with CUDA memory optimizations for RTX 3090
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
        "CUDA_LAUNCH_BLOCKING": "0",
    }
    
    try:
        # Clear GPU memory before starting
        torch.cuda.empty_cache()
        
        start_time = time.time()
        # Use Popen to show live output with progress bars
        process = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).parent),
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
        )
        exit_code = process.wait(timeout=600)  # 10 minute timeout
        elapsed = time.time() - start_time
        
        # Clear GPU memory after test
        torch.cuda.empty_cache()
        
        if exit_code == 0:
            print(f"  ✓ DDP training successful ({elapsed:.1f}s)")
            return True
        else:
            print(f"  ✗ DDP training failed (exit code {exit_code})")
            return False
            
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"  ✗ Training timed out (>600s)")
        return False
        return False
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        return False


def test_memory_estimation(model_name: str, dim: str):
    """
    Estimate memory usage for a model with different batch sizes.
    Helps find optimal batch size for available VRAM.
    """
    print(f"\n--- Memory estimation: {model_name} ({dim}) ---")
    
    try:
        from experiments.model_comparison.train_comparison import create_model
        from experiments.model_comparison.config_base import get_input_shape
        
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(0)
        
        model = create_model(model_name, dim, device)
        input_shape = get_input_shape(dim)
        
        # Test different batch sizes
        test_batches = [1, 2, 4, 8] if dim == '3d' else [4, 8, 16, 32]
        
        print(f"  Input shape: {input_shape}")
        
        for batch_size in test_batches:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            
            try:
                x = torch.randn(batch_size, 1, *input_shape, device=device)
                
                # Forward pass
                with torch.cuda.amp.autocast():  # Mixed precision like training
                    output = model(x)
                    loss = output.mean()
                
                # Backward pass
                loss.backward()
                
                peak_mem = torch.cuda.max_memory_allocated(0) / (1024**3)
                print(f"  Batch {batch_size}: {peak_mem:.2f} GB peak memory")
                
                del x, output, loss
                model.zero_grad()
                
            except torch.cuda.OutOfMemoryError:
                print(f"  Batch {batch_size}: OOM")
                torch.cuda.empty_cache()
                break
        
        del model
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False


def run_all_tests(quick: bool = True, ddp: bool = False):
    """Run tests for all model combinations."""
    models = ['unet', 'resnet', 'swin', 'vit']
    dims = ['2d', '3d']
    
    results = {}
    
    print("\n" + "=" * 60)
    print("RUNNING ALL MODEL TESTS")
    print("=" * 60)
    
    for dim in dims:
        for model in models:
            key = f"{model}_{dim}"
            print(f"\n{'='*40}")
            print(f"Testing: {key}")
            print(f"{'='*40}")
            
            # Test model creation
            creation_ok = test_model_creation(model, dim)
            
            if creation_ok and ddp:
                # Test DDP training
                ddp_ok = test_ddp_training(model, dim, quick=quick)
                results[key] = 'PASS' if ddp_ok else 'FAIL'
            else:
                results[key] = 'PASS' if creation_ok else 'FAIL'
            
            # Clear memory between models
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v == 'PASS')
    total = len(results)
    
    for key, status in results.items():
        icon = "✓" if status == 'PASS' else "✗"
        print(f"  {icon} {key}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
    print("=" * 60)
    
    return passed == total


def main():
    parser = argparse.ArgumentParser(description='Test DDP training locally')
    parser.add_argument('--model', type=str, choices=['unet', 'resnet', 'swin', 'vit'],
                        help='Model to test')
    parser.add_argument('--dim', type=str, choices=['2d', '3d'],
                        help='Dimension to test')
    parser.add_argument('--all', action='store_true',
                        help='Test all models')
    parser.add_argument('--ddp', action='store_true',
                        help='Include DDP training tests')
    parser.add_argument('--memory', action='store_true',
                        help='Run memory estimation for optimal batch sizes')
    parser.add_argument('--quick', action='store_true', default=True,
                        help='Quick test with minimal iterations')
    
    args = parser.parse_args()
    
    # Check hardware first
    if not check_hardware():
        sys.exit(1)
    
    if args.memory:
        # Memory estimation mode
        if args.model and args.dim:
            test_memory_estimation(args.model, args.dim)
        else:
            for dim in ['2d', '3d']:
                for model in ['unet', 'resnet', 'swin', 'vit']:
                    test_memory_estimation(model, dim)
                    torch.cuda.empty_cache()
    elif args.all:
        # Test all models
        success = run_all_tests(quick=args.quick, ddp=args.ddp)
        sys.exit(0 if success else 1)
    elif args.model and args.dim:
        # Test specific model
        creation_ok = test_model_creation(args.model, args.dim)
        
        if creation_ok and args.ddp:
            ddp_ok = test_ddp_training(args.model, args.dim, quick=args.quick)
            sys.exit(0 if ddp_ok else 1)
        
        sys.exit(0 if creation_ok else 1)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python test_local_ddp.py --model unet --dim 2d")
        print("  python test_local_ddp.py --model swin --dim 3d --ddp")
        print("  python test_local_ddp.py --all")
        print("  python test_local_ddp.py --all --ddp")
        print("  python test_local_ddp.py --memory --model vit --dim 3d")


if __name__ == "__main__":
    main()
