#!/usr/bin/env python3
"""
Single-GPU training script for 7-loss benchmark on Windows RTX 4090.

No DDP, no torchrun — just simple single-GPU training with AMP.
Mirrors train_local.py logic but removes all distributed code.

Usage:
    # Quick test (5 min)
    python train_4090.py --mode quick_test

    # Run the 7-loss benchmark (~2.5-3.5 hours)
    python train_4090.py --mode loss_comparison

    # Single loss experiment
    python train_4090.py --loss tversky_precision

    # Full training with best loss
    python train_4090.py --mode full_train --loss tversky_precision
"""

# Thread limits BEFORE any other imports (4 threads avoids contention with 8 workers)
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

# ── RTX 4090 (Ada Lovelace) performance flags ──
# TF32: 2× faster matmul/conv with negligible precision loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Benchmark: auto-tune convolution algorithms for fixed input sizes
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# TF32 precision for float32 GEMMs
torch.set_float32_matmul_precision('high')

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from config_4090 import (
    LOSS_CONFIGS, QUICK_TEST_CLASSES, ALL_CLASSES,
    CHECKPOINT_DIR, TENSORBOARD_DIR, RESULTS_DIR,
    SPATIAL_TRANSFORMS_2D, DATALOADER_CONFIG, VALIDATION_CONFIG,
    CLASS_LOSS_WEIGHTS, USE_AMP, MAX_GRAD_NORM,
    MODEL_CONFIGS, get_config, get_model_config, ensure_dirs, get_device
)
from losses import get_loss_function, PerClassComboLoss


def log(msg):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def create_model(n_classes: int, input_channels: int = 1):
    """Create UNet 2D model with channels_last and torch.compile.

    Args:
        n_classes: Number of output classes
        input_channels: Number of input channels (1 for 2D, 5 for 2.5D)
    """
    from cellmap_segmentation_challenge.models import UNet_2D
    model = UNet_2D(input_channels, n_classes)

    # channels_last memory format: 10-30% faster conv2d on Ada Lovelace
    model = model.to(memory_format=torch.channels_last)

    # torch.compile: kernel fusion + graph optimization (PyTorch 2.x)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            log("torch.compile applied (reduce-overhead mode)")
        except Exception as e:
            log(f"torch.compile skipped: {e}")

    return model


def create_dataloaders(classes, batch_size, iterations_per_epoch, input_shape=(1, 256, 256)):
    """Create train and validation dataloaders.

    Args:
        classes: List of class names to train on
        batch_size: Batch size (total, single GPU)
        iterations_per_epoch: Number of iterations per epoch
        input_shape: Input shape tuple (Z, H, W)
    """
    from cellmap_segmentation_challenge.utils.dataloader import get_dataloader

    datasplit_path = Path(__file__).parent / "datasplit.csv"

    if not datasplit_path.exists():
        log("Creating datasplit.csv...")
        from cellmap_segmentation_challenge.utils.datasplit import make_datasplit_csv
        from config_4090 import DATA_ROOT
        search_path = str(DATA_ROOT / "{dataset}/{dataset}.zarr/recon-1/{name}")
        make_datasplit_csv(
            classes=classes,
            csv_path=str(datasplit_path),
            search_path=search_path,
            validation_prob=0.15,
            force_all_classes=False,
        )

    input_array_info = {"shape": input_shape, "scale": (8, 8, 8)}
    target_array_info = {"shape": (1, 256, 256), "scale": (8, 8, 8)}

    log(f"Input shape: {input_shape} ({'2.5D' if input_shape[0] > 1 else '2D'})")
    log(f"Output shape: (1, 256, 256)")
    log(f"Batch size: {batch_size}")

    train_loader, val_loader = get_dataloader(
        datasplit_path=str(datasplit_path),
        classes=classes,
        batch_size=batch_size,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=SPATIAL_TRANSFORMS_2D,
        iterations_per_epoch=iterations_per_epoch,
        **DATALOADER_CONFIG,
    )

    return train_loader, val_loader


def compute_metrics(pred, target):
    """Compute Dice and IoU metrics per class."""
    pred_binary = (torch.sigmoid(pred) > 0.5).float()

    valid_mask = ~target.isnan()
    target_clean = target.nan_to_num(0)

    dice_scores = []
    iou_scores = []

    for c in range(pred.shape[1]):
        pred_c = pred_binary[:, c] * valid_mask[:, c]
        target_c = target_clean[:, c] * valid_mask[:, c]

        intersection = (pred_c * target_c).sum()
        pred_sum = pred_c.sum()
        target_sum = target_c.sum()
        union = pred_sum + target_sum - intersection

        dice = (2 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
        iou = (intersection + 1e-6) / (union + 1e-6)

        dice_scores.append(dice.item())
        iou_scores.append(iou.item())

    return {
        'dice_per_class': dice_scores,
        'iou_per_class': iou_scores,
        'dice_mean': sum(dice_scores) / len(dice_scores),
        'iou_mean': sum(iou_scores) / len(iou_scores),
    }


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch):
    """Train for one epoch with per-step scheduler updates."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['input'].to(device, non_blocking=True)

        # Handle 2.5D inputs: dataloader returns [B, 1, Z, H, W], model expects [B, Z, H, W]
        if inputs.dim() == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)

        # channels_last for faster conv2d on Ada Lovelace
        inputs = inputs.to(memory_format=torch.channels_last)
        targets = batch['output'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARNING: NaN/Inf loss at batch {batch_idx}, skipping")
            optimizer.zero_grad(set_to_none=True)
            del inputs, targets, outputs, loss
            continue

        scaler.scale(loss).backward()

        if MAX_GRAD_NORM:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        del inputs, targets, outputs, loss

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, val_loader, criterion, device, classes):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_metrics = {c: {'dice': [], 'iou': []} for c in classes}
    n_batches = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= VALIDATION_CONFIG['batch_limit']:
            break

        inputs = batch['input'].to(device, non_blocking=True)

        # Handle 2.5D inputs
        if inputs.dim() == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)

        inputs = inputs.to(memory_format=torch.channels_last)
        targets = batch['output'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item()
        n_batches += 1

        metrics = compute_metrics(outputs.detach(), targets.detach())
        for i, c in enumerate(classes):
            val = metrics['dice_per_class'][i]
            all_metrics[c]['dice'].append(val.item() if torch.is_tensor(val) else val)
            val = metrics['iou_per_class'][i]
            all_metrics[c]['iou'].append(val.item() if torch.is_tensor(val) else val)

        del inputs, targets, outputs, loss

    # Average metrics
    avg_metrics = {}
    for c in classes:
        if all_metrics[c]['dice']:
            avg_metrics[c] = {
                'dice': sum(all_metrics[c]['dice']) / len(all_metrics[c]['dice']),
                'iou': sum(all_metrics[c]['iou']) / len(all_metrics[c]['iou']),
            }
        else:
            avg_metrics[c] = {'dice': 0, 'iou': 0}

    mean_dice = sum(m['dice'] for m in avg_metrics.values()) / len(classes)
    mean_iou = sum(m['iou'] for m in avg_metrics.values()) / len(classes)

    return {
        'loss': total_loss / max(n_batches, 1),
        'dice_mean': mean_dice,
        'iou_mean': mean_iou,
        'per_class': avg_metrics,
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
    device = get_device()
    ensure_dirs()

    model_config = get_model_config(model_name)
    input_channels = model_config['input_channels']
    input_shape = model_config['input_shape']
    batch_size = config.get('batch_size', model_config['batch_size'])

    classes = config['classes']
    n_classes = len(classes)

    if run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{model_name}_{loss_name}_{timestamp}"

    log(f"\n{'='*60}")
    log(f"Starting experiment: {run_name}")
    log(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    log(f"Model: {model_name} ({model_config['description']})")
    log(f"Input shape: {input_shape} ({input_channels} channels)")
    log(f"Batch size: {batch_size}")
    log(f"Loss: {loss_name}")
    log(f"Classes: {classes}")
    log(f"Epochs: {config['epochs']}, Iters/epoch: {config['iterations_per_epoch']}")
    log(f"{'='*60}\n")

    # Create model (single GPU — no DDP)
    model = create_model(n_classes, input_channels=input_channels)
    model = model.to(device)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        classes=classes,
        batch_size=batch_size,
        iterations_per_epoch=config['iterations_per_epoch'],
        input_shape=input_shape,
    )

    # Create loss function
    loss_config = LOSS_CONFIGS.get(loss_name, LOSS_CONFIGS['baseline_bce'])
    loss_type = loss_config.get('type', 'bce')
    loss_kwargs = {k: v for k, v in loss_config.items() if k not in ['type', 'description']}

    if 'class_weights' in loss_kwargs:
        loss_kwargs['class_weights'] = {
            c: loss_kwargs['class_weights'].get(c, 1.0) for c in classes
        }

    criterion = get_loss_function(loss_type, classes=classes, **loss_kwargs)
    criterion = criterion.to(device)

    log(f"Loss function: {loss_config.get('description', loss_name)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4,
    )

    # OneCycleLR — stepped per iteration inside train_epoch
    total_steps = config['epochs'] * config['iterations_per_epoch']
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
    writer = SummaryWriter(TENSORBOARD_DIR / run_name)

    # Training loop
    best_dice = 0
    results = []

    start_time = time.time()

    for epoch in range(1, config['epochs'] + 1):
        log(f"\nEpoch {epoch}/{config['epochs']}")

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )

        log(f"  Train Loss: {train_loss:.4f}")

        # Validate
        if epoch % config.get('validate_every', 1) == 0:
            val_metrics = validate(model, val_loader, criterion, device, classes)

            log(f"  Val Loss: {val_metrics['loss']:.4f}")
            log(f"  Val Dice: {val_metrics['dice_mean']:.4f}")
            log(f"  Per-class Dice:")
            for c in classes:
                log(f"    {c}: {val_metrics['per_class'][c]['dice']:.4f}")

            # TensorBoard logging
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Dice/mean', val_metrics['dice_mean'], epoch)
            writer.add_scalar('IoU/mean', val_metrics['iou_mean'], epoch)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

            for c in classes:
                writer.add_scalar(f'Dice/{c}', val_metrics['per_class'][c]['dice'], epoch)

            # Save best model
            if val_metrics['dice_mean'] > best_dice:
                best_dice = val_metrics['dice_mean']
                checkpoint_path = CHECKPOINT_DIR / f"{run_name}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
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

    elapsed = time.time() - start_time
    log(f"\nTraining complete in {elapsed/60:.1f} minutes")
    log(f"Best Dice: {best_dice:.4f}")

    # Save results JSON
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

    writer.close()

    return best_dice, results


def run_loss_comparison(config: dict, model_name: str = 'unet_2d'):
    """Run comparison of all 7 loss functions."""

    losses_to_test = [
        # Reference
        'baseline_bce',                       # 1. Anchor
        # Proven winner
        'tversky_precision',                  # 2. α=0.7 β=0.3
        # Explore α/β space
        'tversky_precision_mild',             # 3. α=0.6 β=0.4
        'tversky_precision_strong',           # 4. α=0.8 β=0.2
        # Per-class weighting + Tversky
        'per_class_tversky_precision',        # 5. α=0.7 + class weights
        'per_class_tversky_precision_strong', # 6. α=0.8 + class weights
        # Combo with class weights
        'per_class_weighted_focal',           # 7. BCE+Dice+Focal + class weights
    ]

    results = {}

    for loss_name in losses_to_test:
        log(f"\n{'#'*60}")
        log(f"Testing loss: {loss_name} with model: {model_name}")
        log(f"{'#'*60}")

        try:
            best_dice, history = run_experiment(loss_name, config, model_name=model_name)
            results[loss_name] = {
                'best_dice': best_dice,
                'description': LOSS_CONFIGS[loss_name]['description'],
            }
        except Exception as e:
            log(f"ERROR training {loss_name}: {e}")
            import traceback
            traceback.print_exc()
            results[loss_name] = {'best_dice': 0, 'error': str(e)}

        # Clear GPU memory between runs
        torch.cuda.empty_cache()
        gc.collect()

    # Summary
    log(f"\n{'='*60}")
    log("LOSS COMPARISON RESULTS")
    log(f"{'='*60}")

    sorted_results = sorted(results.items(), key=lambda x: x[1].get('best_dice', 0), reverse=True)
    for i, (name, res) in enumerate(sorted_results, 1):
        desc = res.get('description', '')
        dice = res.get('best_dice', 0)
        log(f"{i}. {name}: Dice={dice:.4f} - {desc}")

    # Save summary
    summary_path = RESULTS_DIR / f"loss_comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nSummary saved to {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='7-Loss Benchmark on RTX 4090 (single GPU)')
    parser.add_argument('--mode', type=str, default='quick_test',
                        choices=['quick_test', 'loss_comparison', 'model_comparison', 'full_train'],
                        help='Training mode')
    parser.add_argument('--model', type=str, default='unet_2d',
                        choices=['unet_2d', 'unet_25d'],
                        help='Model architecture')
    parser.add_argument('--loss', type=str, default='tversky_precision',
                        choices=list(LOSS_CONFIGS.keys()),
                        help='Loss function (for quick_test / full_train modes)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')

    args = parser.parse_args()

    config = get_config(args.mode)

    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size

    log(f"Mode: {args.mode}")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"Config: {config}")

    if args.mode == 'loss_comparison':
        run_loss_comparison(config, model_name=args.model)
    else:
        run_experiment(args.loss, config, model_name=args.model)


if __name__ == '__main__':
    main()
