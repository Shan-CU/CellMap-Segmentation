#!/usr/bin/env python3
"""
Local training script for loss optimization experiments on Shenron.

This script runs training experiments comparing different loss functions
using DDP across 4x RTX 2080 Ti GPUs.

Usage:
    # Quick test (5 min)
    python train_local.py --mode quick_test
    
    # Compare loss functions (1-2 hours)
    python train_local.py --mode loss_comparison
    
    # Full training with specific loss (8-12 hours)
    python train_local.py --mode full_train --loss per_class_weighted
    
    # Run with DDP (automatically used for multi-GPU)
    torchrun --nproc_per_node=4 train_local.py --mode quick_test
"""

# CRITICAL: Set thread limits BEFORE any imports to prevent thread explosion
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
# Set torch internal thread limits
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as T
from cellmap_data.transforms.augment import NaNtoNum, Binarize

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Auto-detect hostname and import appropriate config
import socket
hostname = socket.gethostname().lower()

if 'rocinante' in hostname:
    from config_rocinante import (
        LOSS_CONFIGS, QUICK_TEST_CLASSES, ALL_CLASSES,
        CHECKPOINT_DIR, TENSORBOARD_DIR, RESULTS_DIR,
        SPATIAL_TRANSFORMS_2D, DATALOADER_CONFIG, VALIDATION_CONFIG,
        CLASS_LOSS_WEIGHTS, USE_AMP, MAX_GRAD_NORM,
        MODEL_CONFIGS, get_config, get_model_config, ensure_dirs, get_device
    )
    print(f"[OK] Using config_rocinante (hostname: {hostname})")
else:
    from config_shenron import (
        LOSS_CONFIGS, QUICK_TEST_CLASSES, ALL_CLASSES,
        CHECKPOINT_DIR, TENSORBOARD_DIR, RESULTS_DIR,
        SPATIAL_TRANSFORMS_2D, DATALOADER_CONFIG, VALIDATION_CONFIG,
        CLASS_LOSS_WEIGHTS, USE_AMP, MAX_GRAD_NORM,
        MODEL_CONFIGS, get_config, get_model_config, ensure_dirs, get_device
    )
    print(f"[OK] Using config_shenron (hostname: {hostname})")

from losses import get_loss_function, PerClassComboLoss


def setup_ddp():
    """Initialize DDP if running with torchrun."""
    # Check for single GPU mode
    if os.environ.get('SINGLE_GPU_MODE') == '1':
        return 0, 0, 1
    
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        # Single GPU mode
        return 0, 0, 1


def cleanup_ddp():
    """Cleanup DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def log(msg):
    """Print only on main process."""
    if is_main_process():
        print(msg)


def setup_data_logger(log_dir: Path, run_name: str):
    """Setup detailed file logger for data pipeline debugging.
    
    Returns:
        logging.Logger: Configured logger that writes to file ONLY (no console output)
    """
    log_file = log_dir / f"{run_name}_data_debug.log"
    
    # Create logger
    data_logger = logging.getLogger(f'data_debug_{run_name}')
    data_logger.setLevel(logging.DEBUG)
    data_logger.handlers.clear()  # Clear any existing handlers
    data_logger.propagate = False  # CRITICAL: Don't propagate to root logger (prevents console output)
    
    # File handler ONLY - no console output
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    
    # Detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    data_logger.addHandler(fh)
    
    data_logger.info(f"="*80)
    data_logger.info(f"DATA PIPELINE DEBUG LOG: {run_name}")
    data_logger.info(f"="*80)
    
    return data_logger


def log_batch_details(data_logger, batch, phase, batch_idx, classes, detailed=False):
    """Log comprehensive batch statistics.
    
    Args:
        data_logger: Logger instance
        batch: Batch dict with 'input' and 'output'
        phase: 'train' or 'val'
        batch_idx: Batch index
        classes: List of class names
        detailed: If True, log extensive per-sample statistics
    """
    if data_logger is None:
        return
    
    data_logger.info(f"\n{'='*80}")
    data_logger.info(f"{phase.upper()} BATCH {batch_idx}")
    data_logger.info(f"{'='*80}")
    
    # Input statistics
    inputs = batch['input']
    data_logger.info(f"\nINPUT:")
    data_logger.info(f"  Shape: {inputs.shape}")
    data_logger.info(f"  Dtype: {inputs.dtype}")
    data_logger.info(f"  Device: {inputs.device}")
    data_logger.info(f"  Min: {inputs.min().item():.6f}")
    data_logger.info(f"  Max: {inputs.max().item():.6f}")
    data_logger.info(f"  Mean: {inputs.mean().item():.6f}")
    data_logger.info(f"  Std: {inputs.std().item():.6f}")
    data_logger.info(f"  Has NaN: {torch.isnan(inputs).any().item()}")
    data_logger.info(f"  Has Inf: {torch.isinf(inputs).any().item()}")
    
    if detailed:
        data_logger.info(f"  Per-sample stats:")
        for i in range(min(inputs.shape[0], 3)):  # First 3 samples
            sample = inputs[i]
            data_logger.info(f"    Sample {i}: min={sample.min().item():.6f}, "
                           f"max={sample.max().item():.6f}, mean={sample.mean().item():.6f}")
    
    # Target/Ground truth statistics
    targets = batch['output']
    data_logger.info(f"\nTARGET/GROUND TRUTH:")
    data_logger.info(f"  Shape: {targets.shape}")
    data_logger.info(f"  Dtype: {targets.dtype}")
    data_logger.info(f"  Device: {targets.device}")
    
    # Analyze per-class
    n_classes = targets.shape[1]
    for c_idx in range(n_classes):
        class_name = classes[c_idx] if c_idx < len(classes) else f"class_{c_idx}"
        class_data = targets[:, c_idx]
        
        # Check for NaN
        has_nan = torch.isnan(class_data).any().item()
        valid_data = class_data[~torch.isnan(class_data)]
        
        if valid_data.numel() > 0:
            data_logger.info(f"  {class_name}:")
            data_logger.info(f"    Has NaN: {has_nan} ({torch.isnan(class_data).sum().item()} / {class_data.numel()} pixels)")
            data_logger.info(f"    Valid pixels: {valid_data.numel()}")
            data_logger.info(f"    Min: {valid_data.min().item():.6f}")
            data_logger.info(f"    Max: {valid_data.max().item():.6f}")
            data_logger.info(f"    Mean: {valid_data.mean().item():.6f}")
            data_logger.info(f"    Unique values: {valid_data.unique().tolist()[:10]}...")  # First 10
            
            # Check if binary
            unique_vals = valid_data.unique()
            is_binary = len(unique_vals) <= 2 and all(v in [0, 1] for v in unique_vals.tolist())
            data_logger.info(f"    Is binary (0/1): {is_binary}")
            
            if is_binary or (valid_data.min() >= 0 and valid_data.max() <= 1):
                positive_ratio = (valid_data > 0.5).float().mean().item()
                data_logger.info(f"    Positive ratio (>0.5): {positive_ratio:.4f}")
        else:
            data_logger.info(f"  {class_name}: ALL NaN")
    
    # Batch metadata if available
    if 'metadata' in batch:
        data_logger.info(f"\nMETADATA:")
        for key, val in batch['metadata'].items():
            data_logger.info(f"  {key}: {val}")


def create_model(n_classes: int, input_channels: int = 1):
    """Create UNet 2D model.
    
    Args:
        n_classes: Number of output classes
        input_channels: Number of input channels (1 for 2D, 5 for 2.5D)
    """
    from cellmap_segmentation_challenge.models import UNet_2D
    return UNet_2D(input_channels, n_classes)


def create_dataloaders(classes, batch_size, iterations_per_epoch, input_shape=(1, 256, 256), data_logger=None):
    """Create train and validation dataloaders.
    
    Args:
        classes: List of class names to train on
        batch_size: Batch size per GPU
        iterations_per_epoch: Number of iterations per epoch
        input_shape: Input shape tuple (Z, H, W). Use (1, 256, 256) for 2D,
                     (5, 256, 256) for 2.5D with 5 adjacent slices.
        data_logger: Optional logger for data pipeline debugging
    """
    from cellmap_segmentation_challenge.utils.dataloader import get_dataloader
    
    if data_logger:
        data_logger.info(f"\n{'='*80}")
        data_logger.info("DATALOADER CREATION")
        data_logger.info(f"{'='*80}")
        data_logger.info(f"Classes: {classes}")
        data_logger.info(f"Batch size: {batch_size}")
        data_logger.info(f"Iterations per epoch: {iterations_per_epoch}")
        data_logger.info(f"Input shape: {input_shape}")
    
    # Check if datasplit exists, if not create it
    # Only rank 0 creates the datasplit to avoid memory bloat from multiple processes
    datasplit_path = Path(__file__).parent / "datasplit.csv"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if not datasplit_path.exists():
        if local_rank == 0:
            log("Creating datasplit.csv (rank 0 only)...")
            from cellmap_segmentation_challenge.utils.datasplit import make_datasplit_csv
            from config_shenron import DATA_ROOT
            # Format: {DATA_ROOT}/{dataset}/{dataset}.zarr/recon-1/{name}
            search_path = str(DATA_ROOT / "{dataset}/{dataset}.zarr/recon-1/{name}")
            make_datasplit_csv(
                classes=classes,
                csv_path=str(datasplit_path),
                validation_prob=0.15,
                force_all_classes=False,
            )
        # Other ranks wait for rank 0 to finish
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        else:
            import time
            while not datasplit_path.exists():
                time.sleep(1)
    
    # Log datasplit info
    if data_logger and datasplit_path.exists():
        import pandas as pd
        df = pd.read_csv(datasplit_path, header=None)
        data_logger.info(f"\nDATASPLIT INFO:")
        data_logger.info(f"  Path: {datasplit_path}")
        data_logger.info(f"  Total crops: {len(df)}")
        # First column is split type (train/validate)
        if len(df.columns) > 0:
            data_logger.info(f"  Train crops: {(df[0] == 'train').sum()}")
            data_logger.info(f"  Val crops: {(df[0] == 'validate').sum()}")
        data_logger.info(f"  CSV columns: {len(df.columns)}")
        data_logger.info(f"  First row sample: {df.iloc[0].tolist() if len(df) > 0 else 'N/A'}")
    
    # Input uses the provided shape (supports 2D and 2.5D)
    input_array_info = {"shape": input_shape, "scale": (8, 8, 8)}
    # Output is always single slice - we predict the center slice
    target_array_info = {"shape": (1, 256, 256), "scale": (8, 8, 8)}
    
    log(f"Input shape: {input_shape} ({'2.5D' if input_shape[0] > 1 else '2D'})")
    log(f"Output shape: (1, 256, 256)")
    
    # FIX: Use explicit /255.0 normalization instead of T.ToDtype(scale=True).
    # When spatial transforms (rotation) are applied, xarray.interp() returns float64
    # instead of uint8. T.ToDtype(scale=True) only scales int->float conversions,
    # so float64->float32 is NOT scaled, leaving training data in [0, 255] while
    # validation data (uint8 path) gets correctly scaled to [0, 1].
    # We clamp to [0, 1] after dividing to handle any interpolation overshoot.
    def _normalize_to_float32(x):
        x = x.float()
        if x.max() > 1.5:  # Raw uint8 range [0, 255]
            x = x / 255.0
        return x.clamp(0.0, 1.0)
    
    raw_value_transforms = T.Compose([
        T.Lambda(_normalize_to_float32),
        NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
    ])
    
    if data_logger:
        data_logger.info(f"\nTRANSFORMS:")
        data_logger.info(f"  Spatial transforms type: {type(SPATIAL_TRANSFORMS_2D).__name__}")
        data_logger.info(f"  Spatial transforms: {SPATIAL_TRANSFORMS_2D}")
        data_logger.info(f"  Value transforms type: {type(raw_value_transforms).__name__}")
        data_logger.info(f"  Value transforms: {raw_value_transforms}")
        # Document what the transforms do
        data_logger.info(f"  Value transform details:")
        data_logger.info(f"    - Lambda: Normalize to [0,1] if max>1.5, then clamp to [0,1]")
        data_logger.info(f"    - NaNtoNum: Convert NaN to 0, keep posinf/neginf as-is")
        data_logger.info(f"\nARRAY INFO:")
        data_logger.info(f"  Input: shape={input_array_info['shape']}, scale={input_array_info['scale']}")
        data_logger.info(f"  Target: shape={target_array_info['shape']}, scale={target_array_info['scale']}")
    
    # Override dataloader config for DDP - MUST use 0 workers to avoid fork bomb
    dataloader_kwargs = DATALOADER_CONFIG.copy()
    if torch.distributed.is_initialized():
        dataloader_kwargs['num_workers'] = 0
        dataloader_kwargs['persistent_workers'] = False
        log("DDP mode: forcing num_workers=0 to prevent fork bomb")
    
    # Store random_validation setting for logging
    random_validation = True
    
    if data_logger:
        data_logger.info(f"\nDATALOADER CONFIG:")
        for key, val in dataloader_kwargs.items():
            data_logger.info(f"  {key}: {val}")
        data_logger.info(f"  random_validation: {random_validation}")
        data_logger.info(f"  datasplit_path: {datasplit_path}")
        data_logger.info(f"  iterations_per_epoch: {iterations_per_epoch}")
    
    train_loader, val_loader = get_dataloader(
        datasplit_path=str(datasplit_path),
        classes=classes,
        batch_size=batch_size,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=SPATIAL_TRANSFORMS_2D,
        iterations_per_epoch=iterations_per_epoch,
        train_raw_value_transforms=raw_value_transforms,
        val_raw_value_transforms=raw_value_transforms,
        random_validation=random_validation,
        **dataloader_kwargs,
    )
    
    if data_logger:
        data_logger.info(f"\nDATALOADER CREATED:")
        data_logger.info(f"  Train loader: {len(train_loader)} batches/epoch")
        data_logger.info(f"  Val loader: {len(val_loader)} total batches")
        # Try to get dataset info
        if hasattr(train_loader, 'dataset'):
            train_ds = train_loader.dataset
            data_logger.info(f"  Train dataset type: {type(train_ds).__name__}")
            # Check for actual attributes
            if hasattr(train_ds, '__len__'):
                try:
                    data_logger.info(f"  Train dataset length: {len(train_ds)}")
                except:
                    data_logger.info(f"  Train dataset length: N/A")
        if hasattr(val_loader, 'dataset'):
            val_ds = val_loader.dataset
            data_logger.info(f"  Val dataset type: {type(val_ds).__name__}")
            if hasattr(val_ds, '__len__'):
                try:
                    data_logger.info(f"  Val dataset length: {len(val_ds)}")
                except:
                    data_logger.info(f"  Val dataset length: N/A")
        
        # Verify actual batch_size from loader
        data_logger.info(f"  Train loader batch_size: {train_loader.batch_size if hasattr(train_loader, 'batch_size') else 'N/A'}")
        data_logger.info(f"  Val loader batch_size: {val_loader.batch_size if hasattr(val_loader, 'batch_size') else 'N/A'}")
    
    return train_loader, val_loader


def compute_batch_counts(pred, target):
    """Compute raw TP, FP, FN counts per class for a single batch.
    
    Returns integer counts — no smoothing, no averaging.
    These get accumulated across all val batches, then Dice is computed once.
    Uses standard threshold=0.5.
    """
    pred_sigmoid = torch.sigmoid(pred)
    pred_binary = (pred_sigmoid > 0.5).float()
    
    # Handle NaN in target
    valid_mask = ~target.isnan()
    target_clean = target.nan_to_num(0)
    
    tp_list = []
    fp_list = []
    fn_list = []
    
    for c in range(pred.shape[1]):
        pred_c = pred_binary[:, c] * valid_mask[:, c]
        target_c = target_clean[:, c] * valid_mask[:, c]
        
        tp = (pred_c * target_c).sum().item()
        fp = (pred_c * (1 - target_c)).sum().item()
        fn = ((1 - pred_c) * target_c).sum().item()
        
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
    
    return {'tp': tp_list, 'fp': fp_list, 'fn': fn_list}


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, data_logger=None, classes=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Log first batch of first epoch in detail
        if epoch == 1 and batch_idx == 0 and data_logger:
            data_logger.info(f"\n{'#'*80}")
            data_logger.info(f"FIRST TRAINING BATCH (EPOCH 1)")
            data_logger.info(f"{'#'*80}")
            log_batch_details(data_logger, batch, 'train', batch_idx, classes, detailed=True)
        # Log periodic batches less verbosely
        elif batch_idx % 50 == 0 and data_logger:
            log_batch_details(data_logger, batch, 'train', batch_idx, classes, detailed=False)
        
        inputs = batch['input'].to(device)
        
        # FIX: Handle 2.5D inputs with extra dimension
        # Dataloader returns [B, 1, Z, H, W] but model expects [B, Z, H, W]
        if inputs.dim() == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)  # Remove channel dim → [B, Z, H, W]
        
        # Log shape transformation if it happened
        if epoch == 1 and batch_idx == 0 and data_logger:
            data_logger.info(f"\nAFTER PREPROCESSING (before model):")
            data_logger.info(f"  Input shape: {inputs.shape} (squeezed if needed)")
        
        # NOTE: Keep NaN values - loss functions handle them internally
        targets = batch['output'].to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            outputs = model(inputs)
            
            # Log model output on first batch
            if epoch == 1 and batch_idx == 0 and data_logger:
                data_logger.info(f"\nMODEL OUTPUT (logits):")
                data_logger.info(f"  Shape: {outputs.shape}")
                data_logger.info(f"  Dtype: {outputs.dtype}")
                data_logger.info(f"  Min: {outputs.min().item():.6f}")
                data_logger.info(f"  Max: {outputs.max().item():.6f}")
                data_logger.info(f"  Mean: {outputs.mean().item():.6f}")
                data_logger.info(f"  Std: {outputs.std().item():.6f}")
                
                # Show per-class output stats
                for c_idx in range(outputs.shape[1]):
                    class_name = classes[c_idx] if classes and c_idx < len(classes) else f"class_{c_idx}"
                    class_out = outputs[:, c_idx]
                    data_logger.info(f"  {class_name}: min={class_out.min().item():.4f}, "
                                   f"max={class_out.max().item():.4f}, mean={class_out.mean().item():.4f}")
            
            loss = criterion(outputs, targets)
            
            # Log loss calculation details on first batch
            if epoch == 1 and batch_idx == 0 and data_logger:
                data_logger.info(f"\nLOSS CALCULATION:")
                data_logger.info(f"  Loss function: {type(criterion).__name__}")
                data_logger.info(f"  Loss value: {loss.item():.6f}")
                data_logger.info(f"  Criterion config: {criterion if hasattr(criterion, '__dict__') else 'N/A'}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            if is_main_process():
                print(f"  WARNING: NaN/Inf loss at batch {batch_idx}, skipping")
            if data_logger:
                data_logger.warning(f"  NaN/Inf loss at batch {batch_idx}, skipping")
            optimizer.zero_grad()
            del inputs, targets, outputs, loss
            continue
        
        scaler.scale(loss).backward()
        
        if MAX_GRAD_NORM:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1
        
        if batch_idx % 10 == 0 and is_main_process():
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Release memory
        del inputs, targets, outputs, loss
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, val_loader, criterion, device, classes, data_logger=None, epoch=None):
    """Validate model with global micro-averaged Dice.
    
    Accumulates TP/FP/FN across ALL validation batches, then computes
    Dice once at the end. This avoids smoothing artifacts from per-batch
    averaging on rare classes.
    """
    model.eval()
    total_loss = 0
    n_batches = 0
    
    # Global accumulators
    global_tp = [0] * len(classes)
    global_fp = [0] * len(classes)
    global_fn = [0] * len(classes)
    
    if data_logger and epoch == 1:
        data_logger.info(f"\n{'#'*80}")
        data_logger.info(f"VALIDATION - EPOCH {epoch}")
        data_logger.info(f"{'#'*80}")
        data_logger.info(f"Validation batch limit: {VALIDATION_CONFIG['batch_limit']}")
    
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= VALIDATION_CONFIG['batch_limit']:
            break
        
        # Log first validation batch in detail
        if epoch == 1 and batch_idx == 0 and data_logger:
            log_batch_details(data_logger, batch, 'val', batch_idx, classes, detailed=True)
        
        inputs = batch['input'].to(device)
        
        # FIX: Handle 2.5D inputs with extra dimension
        if inputs.dim() == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)  # Remove channel dim → [B, Z, H, W]
        
        # NOTE: Keep NaN values - loss functions handle them internally
        targets = batch['output'].to(device)
        
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Print sigmoid stats on first batch
        if batch_idx == 0 and is_main_process():
            sigmoid_out = torch.sigmoid(outputs)
            print("  [Sigmoid stats per class]")
            for i, c in enumerate(classes):
                vals = sigmoid_out[:, i].flatten()
                print(f"    {c}: min={vals.min():.4f}, max={vals.max():.4f}, "
                      f"mean={vals.mean():.4f}, >0.5: {(vals > 0.5).float().mean()*100:.1f}%")
            
            # Also log to file
            if data_logger and epoch == 1:
                data_logger.info(f"\nVALIDATION MODEL OUTPUT (first batch):")
                data_logger.info(f"  Logits shape: {outputs.shape}")
                data_logger.info(f"  Sigmoid stats per class:")
                for i, c in enumerate(classes):
                    vals = sigmoid_out[:, i].flatten()
                    data_logger.info(f"    {c}: min={vals.min():.4f}, max={vals.max():.4f}, "
                                   f"mean={vals.mean():.4f}, >0.5: {(vals > 0.5).float().mean()*100:.1f}%")
        
        total_loss += loss.item()
        n_batches += 1
        
        # Accumulate TP/FP/FN counts
        counts = compute_batch_counts(outputs.detach(), targets.detach())
        for i in range(len(classes)):
            global_tp[i] += counts['tp'][i]
            global_fp[i] += counts['fp'][i]
            global_fn[i] += counts['fn'][i]
        
        # Release memory
        del inputs, targets, outputs, loss
    
    # Compute global micro-averaged Dice per class: 2*TP / (2*TP + FP + FN)
    per_class = {}
    for i, c in enumerate(classes):
        tp, fp, fn = global_tp[i], global_fp[i], global_fn[i]
        denom = 2 * tp + fp + fn
        dice = (2 * tp / denom) if denom > 0 else 0.0
        per_class[c] = {'dice': dice, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}
    
    mean_dice = sum(m['dice'] for m in per_class.values()) / len(classes)
    
    return {
        'loss': total_loss / max(n_batches, 1),
        'dice_mean': mean_dice,
        'per_class': per_class,
    }


def run_experiment(
    loss_name: str,
    config: dict,
    model_name: str = 'unet_2d',
    run_name: str = None,
):
    """Run a single training experiment.
    
    Args:
        loss_name: Name of loss function to use
        config: Training configuration dict
        model_name: Model to use ('unet_2d' or 'unet_25d')
        run_name: Optional name for this run
    """
    
    # Setup
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    ensure_dirs()
    
    # Get model configuration
    model_config = get_model_config(model_name)
    input_channels = model_config['input_channels']
    input_shape = model_config['input_shape']
    batch_size = model_config['batch_size']
    
    classes = config['classes']
    n_classes = len(classes)
    
    if run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{model_name}_{loss_name}_{timestamp}"
    
    log(f"\n{'='*60}")
    log(f"Starting experiment: {run_name}")
    log(f"Model: {model_name} ({model_config['description']})")
    log(f"Input shape: {input_shape} ({input_channels} channels)")
    log(f"Batch size: {batch_size}")
    log(f"Loss: {loss_name}")
    log(f"Classes: {classes}")
    log(f"{'='*60}\n")
    
    # Setup data logger (only on main process)
    data_logger = None
    if is_main_process():
        data_logger = setup_data_logger(RESULTS_DIR, run_name)
        data_logger.info(f"Experiment: {run_name}")
        data_logger.info(f"Model: {model_name}")
        data_logger.info(f"Loss: {loss_name}")
        data_logger.info(f"Classes: {classes}")
        data_logger.info(f"Config: {config}")
    
    # Create model with correct input channels
    model = create_model(n_classes, input_channels=input_channels)
    model = model.to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Create dataloaders with correct input shape
    train_loader, val_loader = create_dataloaders(
        classes=classes,
        batch_size=batch_size,
        iterations_per_epoch=config['iterations_per_epoch'],
        input_shape=input_shape,
        data_logger=data_logger,
    )
    
    # Create loss function
    loss_config = LOSS_CONFIGS.get(loss_name, LOSS_CONFIGS['per_class_weighted'])
    loss_type = loss_config.get('type', 'per_class_combo')
    loss_kwargs = {k: v for k, v in loss_config.items() if k not in ['type', 'description']}
    
    if 'class_weights' in loss_kwargs:
        # Filter weights to only include classes we're training
        loss_kwargs['class_weights'] = {
            c: loss_kwargs['class_weights'].get(c, 1.0) for c in classes
        }
    
    criterion = get_loss_function(loss_type, classes=classes, **loss_kwargs)
    criterion = criterion.to(device)
    
    log(f"Loss function: {loss_config.get('description', loss_name)}")
    
    # Optimizer and scheduler
    base_model = model.module if hasattr(model, 'module') else model
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4,
    )
    
    total_steps = config['epochs'] * config['iterations_per_epoch']
    
    # OneCycleLR requires at least 2 steps and reasonable pct_start
    # With pct_start=0.05, need at least 40 steps (2/0.05) to avoid division by zero
    # For short runs, use simpler constant LR
    if total_steps <= 40:
        log(f"Using constant LR (total_steps={total_steps} too small for OneCycleLR)")
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=total_steps
        )
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy='cos',
        )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    # TensorBoard
    writer = None
    if is_main_process():
        writer = SummaryWriter(TENSORBOARD_DIR / run_name)
    
    # Training loop
    best_dice = 0
    results = []
    
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        log(f"\nEpoch {epoch}/{config['epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            data_logger=data_logger, classes=classes
        )
        scheduler.step()
        
        log(f"  Train Loss: {train_loss:.4f}")
        if data_logger:
            data_logger.info(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}")
        
        # Validate
        if epoch % config.get('validate_every', 1) == 0:
            val_metrics = validate(model, val_loader, criterion, device, classes,
                                 data_logger=data_logger, epoch=epoch)
            
            log(f"  Val Loss: {val_metrics['loss']:.4f}")
            log(f"  Val Dice: {val_metrics['dice_mean']:.4f}")
            log(f"  Per-class (global micro-averaged):")
            for c in classes:
                m = val_metrics['per_class'][c]
                log(f"    {c}: Dice={m['dice']:.4f}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
                        # Log to file
            if data_logger:
                data_logger.info(f"\nEpoch {epoch} Validation Results:")
                data_logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
                data_logger.info(f"  Val Dice (mean): {val_metrics['dice_mean']:.4f}")
                data_logger.info(f"  Per-class results:")
                for c in classes:
                    m = val_metrics['per_class'][c]
                    data_logger.info(f"    {c}: Dice={m['dice']:.4f}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
                        # Log to file
            if data_logger:
                data_logger.info(f"\nEpoch {epoch} Validation Results:")
                data_logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
                data_logger.info(f"  Val Dice (mean): {val_metrics['dice_mean']:.4f}")
                data_logger.info(f"  Per-class results:")
                for c in classes:
                    m = val_metrics['per_class'][c]
                    data_logger.info(f"    {c}: Dice={m['dice']:.4f}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                writer.add_scalar('Dice/mean', val_metrics['dice_mean'], epoch)
                writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
                
                for c in classes:
                    writer.add_scalar(f'Dice/{c}', val_metrics['per_class'][c]['dice'], epoch)
                    writer.add_scalar(f'FP/{c}', val_metrics['per_class'][c]['fp'], epoch)
                    writer.add_scalar(f'FN/{c}', val_metrics['per_class'][c]['fn'], epoch)
            
            # Save best model
            if val_metrics['dice_mean'] > best_dice and is_main_process():
                best_dice = val_metrics['dice_mean']
                checkpoint_path = CHECKPOINT_DIR / f"{run_name}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': base_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice': best_dice,
                    'loss_name': loss_name,
                    'classes': classes,
                }, checkpoint_path)
                log(f"  Saved best model (Dice: {best_dice:.4f})")
            
            results.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'dice_mean': val_metrics['dice_mean'],
                'per_class': val_metrics['per_class'],
            })
        
        # Force garbage collection after each epoch
        torch.cuda.empty_cache()
        gc.collect()
    
    elapsed = time.time() - start_time
    log(f"\nTraining complete in {elapsed/60:.1f} minutes")
    log(f"Best Dice: {best_dice:.4f}")
    
    # Save results
    if is_main_process():
        results_path = RESULTS_DIR / f"{run_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'run_name': run_name,
                'loss_name': loss_name,
                'loss_config': loss_config,
                'train_config': config,
                'best_dice': best_dice,
                'elapsed_seconds': elapsed,
                'history': results,
            }, f, indent=2)
        log(f"Results saved to {results_path}")
    
    if writer:
        writer.close()
    
    cleanup_ddp()
    
    return best_dice, results


def run_loss_comparison(config: dict, model_name: str = 'unet_2d'):
    """Run comparison of all loss functions with a single model."""
    
    losses_to_test = [
        # Reference
        'baseline_bce',                    # 1. Anchor (scored 0.028)
        # Proven winner
        'tversky_precision',               # 2. α=0.7 β=0.3 (scored 0.370!)
        # Explore α/β space around the winner
        'tversky_precision_mild',          # 3. α=0.6 β=0.4 — less precision
        'tversky_precision_strong',        # 4. α=0.8 β=0.2 — more precision
        # Does per-class weighting improve the winner?
        'per_class_tversky_precision',     # 5. α=0.7 + class weights
        'per_class_tversky_precision_strong', # 6. α=0.8 + class weights
        # Does per-class weighted Dice+BCE+Focal beat Tversky?
        'per_class_weighted_focal',        # 7. BCE+Dice+Focal + class weights
    ]
    
    results = {}
    
    for loss_name in losses_to_test:
        log(f"\n{'#'*60}")
        log(f"Testing loss: {loss_name} with model: {model_name}")
        log(f"{'#'*60}")
        
        best_dice, history = run_experiment(loss_name, config, model_name=model_name)
        results[loss_name] = {
            'best_dice': best_dice,
            'description': LOSS_CONFIGS[loss_name]['description'],
        }
    
    # Summary
    log(f"\n{'='*60}")
    log("LOSS COMPARISON RESULTS")
    log(f"{'='*60}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_dice'], reverse=True)
    for i, (name, res) in enumerate(sorted_results, 1):
        log(f"{i}. {name}: Dice={res['best_dice']:.4f} - {res['description']}")
    
    return results


def run_model_comparison(config: dict, loss_name: str = 'per_class_weighted'):
    """Run comparison of 2D vs 2.5D models.
    
    This trains 2D first, then 2.5D, with all resources dedicated to each.
    This is the recommended way to compare architectures.
    
    Args:
        config: Training configuration
        loss_name: Loss function to use for both models
    """
    
    models_to_test = ['unet_2d', 'unet_25d']
    results = {}
    
    for model_name in models_to_test:
        model_config = get_model_config(model_name)
        
        log(f"\n{'#'*60}")
        log(f"Training model: {model_name}")
        log(f"Description: {model_config['description']}")
        log(f"Input shape: {model_config['input_shape']}")
        log(f"Batch size: {model_config['batch_size']}")
        log(f"Loss: {loss_name}")
        log(f"{'#'*60}")
        
        best_dice, history = run_experiment(
            loss_name=loss_name,
            config=config,
            model_name=model_name,
        )
        
        # Get final per-class results
        final_per_class = history[-1]['per_class'] if history else {}
        
        results[model_name] = {
            'best_dice': best_dice,
            'description': model_config['description'],
            'input_shape': model_config['input_shape'],
            'per_class': final_per_class,
        }
        
        # Clear GPU memory between runs
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        log(f"\nCleared GPU memory before next model\n")
    
    # Summary comparison
    log(f"\n{'='*60}")
    log("2D vs 2.5D MODEL COMPARISON RESULTS")
    log(f"{'='*60}")
    
    for model_name, res in results.items():
        log(f"\n{model_name}: Best Dice = {res['best_dice']:.4f}")
        log(f"  {res['description']}")
        log(f"  Input: {res['input_shape']}")
        if res['per_class']:
            log("  Per-class Dice:")
            for class_name, metrics in res['per_class'].items():
                dice = metrics.get('dice', 0)
                status = "🟢" if dice > 0.4 else ("🟡" if dice > 0.2 else "🔴")
                log(f"    {status} {class_name}: {dice:.4f}")
    
    # Improvement analysis
    if 'unet_2d' in results and 'unet_25d' in results:
        dice_2d = results['unet_2d']['best_dice']
        dice_25d = results['unet_25d']['best_dice']
        improvement = (dice_25d - dice_2d) / dice_2d * 100 if dice_2d > 0 else 0
        
        log(f"\n{'='*60}")
        log("SUMMARY")
        log(f"{'='*60}")
        log(f"2D UNet:   {dice_2d:.4f}")
        log(f"2.5D UNet: {dice_25d:.4f}")
        log(f"Improvement: {improvement:+.1f}%")
        
        if dice_25d > dice_2d:
            log("\n✅ 2.5D provides better results! Use 2.5D for cluster training.")
        else:
            log("\n⚠️ 2D performed better. Consider other improvements.")
        
        # Per-class improvement for hard classes
        log("\nPer-class improvement (focus on hard classes):")
        for cls in ['nuc', 'endo_mem', 'er_mem', 'pm']:
            if cls in results['unet_2d'].get('per_class', {}) and cls in results['unet_25d'].get('per_class', {}):
                d2d = results['unet_2d']['per_class'][cls].get('dice', 0)
                d25d = results['unet_25d']['per_class'][cls].get('dice', 0)
                imp = (d25d - d2d) / d2d * 100 if d2d > 0 else 0
                log(f"  {cls}: {d2d:.4f} → {d25d:.4f} ({imp:+.1f}%)")
    
    # Save comparison results
    if is_main_process():
        results_path = RESULTS_DIR / "model_comparison_results.json"
        # Convert tuples to lists for JSON serialization
        serializable_results = {}
        for k, v in results.items():
            serializable_results[k] = {
                **v,
                'input_shape': list(v['input_shape']) if isinstance(v['input_shape'], tuple) else v['input_shape']
            }
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        log(f"\nResults saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Loss optimization and model comparison experiments')
    parser.add_argument('--mode', type=str, default='quick_test',
                        choices=['quick_test', 'loss_comparison', 'model_comparison', 'full_train', 'single_loss'],
                        help='Training mode (use single_loss with --loss for parallel GPU runs)')
    parser.add_argument('--model', type=str, default='unet_2d',
                        choices=['unet_2d', 'unet_25d'],
                        help='Model architecture (for single experiment)')
    parser.add_argument('--loss', type=str, default='per_class_weighted',
                        choices=list(LOSS_CONFIGS.keys()),
                        help='Loss function')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (note: model_comparison uses per-model batch sizes)')
    parser.add_argument('--single_gpu', action='store_true',
                        help='Run in single-GPU mode (no DDP, enables num_workers)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of dataloader workers (default=1 for parallel runs)')
    
    args = parser.parse_args()
    
    # Handle single GPU mode - set environment to skip DDP
    if args.single_gpu or args.mode == 'single_loss':
        os.environ['SINGLE_GPU_MODE'] = '1'
        # Enable workers in single GPU mode (but limited for parallel runs)
        import config_shenron
        num_workers = args.num_workers
        config_shenron.DATALOADER_CONFIG['num_workers'] = num_workers
        if num_workers > 0:
            config_shenron.DATALOADER_CONFIG['persistent_workers'] = True
            config_shenron.DATALOADER_CONFIG['prefetch_factor'] = 2
        print(f"🚀 Single GPU mode: num_workers={num_workers}, no DDP")
    
    # Get config
    config = get_config(args.mode if args.mode != 'single_loss' else 'loss_comparison')
    
    # Apply overrides
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    # Run based on mode
    if args.mode == 'single_loss':
        # Single loss on single GPU - for parallel runs
        run_experiment(args.loss, config, model_name=args.model)
    elif args.mode == 'loss_comparison':
        run_loss_comparison(config, model_name=args.model)
    elif args.mode == 'model_comparison':
        run_model_comparison(config, loss_name=args.loss)
    else:
        run_experiment(args.loss, config, model_name=args.model)


if __name__ == '__main__':
    main()
