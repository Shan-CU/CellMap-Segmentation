#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for masking strategy experiments.

Fixes the loss function to BalancedSoftmax Tversky (alpha=0.6, beta=0.4,
tau=1.0) -- best from class-weighting experiments -- and compares
different NaN/unannotated pixel masking strategies.

Re-uses model, data, and DDP infrastructure from class_weighting.

Usage:
    # Quick smoke test (5 epochs)
    python train.py --mode quick_test

    # Run all masking configs sequentially
    python train.py --mode comparison

    # Single strategy on single GPU (for parallel launches)
    python train.py --mode single --strategy no_mask --single_gpu

    # Summary table
    python train.py --mode summary
"""

# -- Thread limits (BEFORE any other imports) --
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
import torch.distributed as dist
import torchvision.transforms.v2 as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from cellmap_data.transforms.augment import NaNtoNum

# -- Path setup --
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
# IMPORTANT: this experiment's dir FIRST so its config.py wins
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(1, str(REPO_ROOT))
sys.path.insert(2, str(REPO_ROOT / "src"))
# class_weighting AFTER this dir -- only for shared utilities if needed
sys.path.insert(3, str(SCRIPT_DIR.parent / "class_weighting"))

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = True

# -- Config --
from config import (
    MASKING_CONFIGS, QUICK_TEST_CLASSES,
    CHECKPOINT_DIR, TENSORBOARD_DIR, RESULTS_DIR, DATASPLIT_CSV,
    SPATIAL_TRANSFORMS_2D, DATALOADER_CONFIG, VALIDATION_CONFIG,
    MODEL_CONFIG, USE_AMP, MAX_GRAD_NORM,
    TVERSKY_ALPHA, TVERSKY_BETA, BALANCED_SOFTMAX_TAU,
    DATA_ROOT, EXPERIMENT_DIR,
    get_config, ensure_dirs,
)
from masking_losses import get_masking_loss


# ======================================================================
# DDP helpers  (identical to class_weighting/train.py)
# ======================================================================

def setup_ddp():
    if os.environ.get('SINGLE_GPU_MODE') == '1':
        torch.cuda.set_device(0)
        return 0, 0, 1
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    torch.cuda.set_device(0)
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def log(msg):
    if is_main_process():
        print(msg)


# ======================================================================
# Model  (same UNet_2D as class_weighting)
# ======================================================================

def create_model(n_classes: int, input_channels: int = 1):
    from cellmap_segmentation_challenge.models import UNet_2D
    return UNet_2D(input_channels, n_classes)


# ======================================================================
# Data  (re-uses class_weighting datasplit for reproducibility)
# ======================================================================

def create_dataloaders(classes, batch_size, iterations_per_epoch,
                       input_shape=(1, 256, 256)):
    from cellmap_segmentation_challenge.utils.dataloader import get_dataloader

    datasplit_path = DATASPLIT_CSV
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not datasplit_path.exists():
        if local_rank == 0:
            log("Creating datasplit.csv (rank 0 only)...")
            from cellmap_segmentation_challenge.utils.datasplit import make_datasplit_csv
            make_datasplit_csv(
                classes=classes,
                csv_path=str(datasplit_path),
                validation_prob=0.15,
                force_all_classes=False,
            )
        if dist.is_initialized():
            dist.barrier()
        else:
            import time as _t
            while not datasplit_path.exists():
                _t.sleep(0.5)

    input_array_info = {"shape": input_shape, "scale": (8, 8, 8)}
    target_array_info = {"shape": (1, 256, 256), "scale": (8, 8, 8)}

    def _normalize_to_float32(x):
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        return x.clamp(0.0, 1.0)

    raw_value_transforms = T.Compose([
        T.Lambda(_normalize_to_float32),
        NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
    ])

    dl_kwargs = DATALOADER_CONFIG.copy()
    if dist.is_initialized():
        dl_kwargs['num_workers'] = 0
        dl_kwargs['persistent_workers'] = False
        log("DDP mode: forcing num_workers=0")
    else:
        log(f"DataLoader: num_workers={dl_kwargs.get('num_workers', 0)}, "
            f"prefetch={dl_kwargs.get('prefetch_factor', 2)}")

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
        random_validation=True,
        **dl_kwargs,
    )
    return train_loader, val_loader


# ======================================================================
# Metrics  (identical to class_weighting -- NaN-masked validation dice)
# ======================================================================

def compute_batch_counts(pred, target):
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    valid_mask = ~target.isnan()
    target_clean = target.nan_to_num(0)

    tp_list, fp_list, fn_list = [], [], []
    for c in range(pred.shape[1]):
        p = pred_binary[:, c] * valid_mask[:, c]
        t = target_clean[:, c] * valid_mask[:, c]
        tp_list.append((p * t).sum().item())
        fp_list.append((p * (1 - t)).sum().item())
        fn_list.append(((1 - p) * t).sum().item())
    return {'tp': tp_list, 'fp': fp_list, 'fn': fn_list}


# ======================================================================
# Foreground mask  (exclude black padding from loss & metrics)
# ======================================================================

FG_THRESHOLD = 0.01  # pixels below this in the normalized [0,1] image are padding

def apply_foreground_mask(inputs, targets):
    """Set target to NaN wherever the raw image is black (padding).

    CellMap zarr crops often have black padding regions where no EM data
    exists.  The model predicts organelles there (free FP), inflating
    false-positive counts and hurting precision/Dice.

    By injecting NaN into targets at those pixels, ALL existing masking
    strategies automatically ignore them — no per-strategy changes needed.

    Args:
        inputs:  (B, C_in, H, W) normalized raw image, typically C_in=1.
        targets: (B, C_out, H, W) target with existing NaN for unannotated.

    Returns:
        targets with additional NaN at padding pixels (modified in-place).
    """
    # inputs may be (B, 1, H, W) or (B, H, W) — handle both
    if inputs.dim() == 4:
        fg_mask = (inputs.abs().amax(dim=1, keepdim=True) > FG_THRESHOLD)  # (B, 1, H, W)
    else:
        fg_mask = (inputs.abs() > FG_THRESHOLD).unsqueeze(1)  # (B, 1, H, W)

    # Expand to match target channels: (B, 1, H, W) broadcasts to (B, C, H, W)
    # Set target to NaN where fg_mask is False (padding region)
    targets = targets.clone()
    targets[~fg_mask.expand_as(targets)] = float('nan')
    return targets


# ======================================================================
# Train / Validate
# ======================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, scheduler,
                device, epoch, classes=None):
    model.train()
    total_loss = 0
    n_batches = 0

    # Tell EU loss what epoch we're on (for warmup)
    if hasattr(criterion, 'set_epoch'):
        criterion.set_epoch(epoch)

    for batch_idx, batch in enumerate(loader):
        inputs = batch['input'].to(device)
        if inputs.dim() == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        targets = batch['output'].to(device)

        # Mask out black padding regions (NaN injected into targets)
        targets = apply_foreground_mask(inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            if is_main_process():
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

        if batch_idx % 25 == 0 and is_main_process():
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Batch {batch_idx}/{len(loader)}, "
                  f"Loss: {loss.item():.4f}, LR: {lr_now:.2e}")

        del inputs, targets, outputs, loss

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, classes):
    model.eval()
    total_loss = 0
    n_batches = 0

    global_tp = [0] * len(classes)
    global_fp = [0] * len(classes)
    global_fn = [0] * len(classes)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= VALIDATION_CONFIG['batch_limit']:
            break

        inputs = batch['input'].to(device)
        if inputs.dim() == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        targets = batch['output'].to(device)

        # Mask out black padding regions
        targets = apply_foreground_mask(inputs, targets)

        with torch.amp.autocast('cuda', enabled=USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if batch_idx == 0 and is_main_process():
            sigmoid_out = torch.sigmoid(outputs)
            print("  [Sigmoid stats per class]")
            for i, c in enumerate(classes):
                vals = sigmoid_out[:, i].flatten()
                print(f"    {c}: min={vals.min():.4f}, max={vals.max():.4f}, "
                      f"mean={vals.mean():.4f}, >0.5: {(vals > 0.5).float().mean()*100:.1f}%")

        total_loss += loss.item()
        n_batches += 1

        counts = compute_batch_counts(outputs.detach(), targets.detach())
        for i in range(len(classes)):
            global_tp[i] += counts['tp'][i]
            global_fp[i] += counts['fp'][i]
            global_fn[i] += counts['fn'][i]

        del inputs, targets, outputs, loss

    per_class = {}
    for i, c in enumerate(classes):
        tp, fp, fn = global_tp[i], global_fp[i], global_fn[i]
        denom = 2 * tp + fp + fn
        dice = (2 * tp / denom) if denom > 0 else 0.0
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec  = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        iou_denom = tp + fp + fn
        iou  = (tp / iou_denom) if iou_denom > 0 else 0.0
        per_class[c] = {
            'dice': dice, 'precision': prec, 'recall': rec, 'iou': iou,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        }

    mean_dice = sum(m['dice'] for m in per_class.values()) / len(classes)
    mean_prec = sum(m['precision'] for m in per_class.values()) / len(classes)
    mean_rec  = sum(m['recall'] for m in per_class.values()) / len(classes)
    mean_iou  = sum(m['iou'] for m in per_class.values()) / len(classes)
    return {
        'loss': total_loss / max(n_batches, 1),
        'dice_mean': mean_dice,
        'precision_mean': mean_prec,
        'recall_mean': mean_rec,
        'iou_mean': mean_iou,
        'per_class': per_class,
    }


# ======================================================================
# Experiment runner
# ======================================================================

def run_experiment(strategy_name: str, config: dict, run_name: str = None):
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    ensure_dirs()

    classes = config['classes']
    n_classes = len(classes)
    batch_size = MODEL_CONFIG['batch_size']
    input_shape = MODEL_CONFIG['input_shape']
    input_channels = MODEL_CONFIG['input_channels']

    if run_name is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"mask_{strategy_name}_{ts}"

    log(f"\n{'='*60}")
    log(f"Experiment: {run_name}")
    log(f"Masking strategy: {strategy_name}")
    log(f"Classes: {classes}")
    log(f"Batch: {batch_size}  Input: {input_shape}")
    log(f"{'='*60}\n")

    # -- Model --
    model = create_model(n_classes, input_channels).to(device)
    try:
        model = torch.compile(model, mode='reduce-overhead')
        log("  torch.compile enabled (reduce-overhead)")
    except Exception as e:
        log(f"  torch.compile unavailable: {e}")
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # -- Data --
    train_loader, val_loader = create_dataloaders(
        classes=classes,
        batch_size=batch_size,
        iterations_per_epoch=config['iterations_per_epoch'],
        input_shape=input_shape,
    )

    # -- Loss (masking strategy) --
    strat_cfg = MASKING_CONFIGS[strategy_name]
    strat_kwargs = {k: v for k, v in strat_cfg.items()
                    if k not in ('strategy', 'description')}
    criterion = get_masking_loss(
        strategy=strat_cfg['strategy'],
        alpha=TVERSKY_ALPHA,
        beta=TVERSKY_BETA,
        tau=BALANCED_SOFTMAX_TAU,
        **strat_kwargs,
    ).to(device)
    log(f"Strategy: {strat_cfg.get('description', strategy_name)}")

    # -- Optimizer / Scheduler --
    base_model = model.module if hasattr(model, 'module') else model
    optimizer = torch.optim.AdamW(base_model.parameters(),
                                  lr=config['learning_rate'],
                                  weight_decay=1e-4)
    total_steps = config['epochs'] * config['iterations_per_epoch']
    if total_steps <= 40:
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=total_steps)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['learning_rate'],
            total_steps=total_steps, pct_start=0.05,
            anneal_strategy='cos')

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    writer = None
    if is_main_process():
        writer = SummaryWriter(TENSORBOARD_DIR / run_name)

    # -- Training loop --
    best_dice = 0
    results = []
    start_time = time.time()

    for epoch in range(1, config['epochs'] + 1):
        log(f"\nEpoch {epoch}/{config['epochs']}")

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            scheduler, device, epoch, classes=classes)
        log(f"  Train Loss: {train_loss:.4f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if epoch % config.get('validate_every', 1) == 0:
            val_res = validate(model, val_loader, criterion, device, classes)
            log(f"  Val Loss: {val_res['loss']:.4f}  "
                f"Dice: {val_res['dice_mean']:.4f}  "
                f"Prec: {val_res['precision_mean']:.4f}  "
                f"Rec: {val_res['recall_mean']:.4f}  "
                f"IoU: {val_res['iou_mean']:.4f}")
            for c in classes:
                pc = val_res['per_class'][c]
                log(f"    {c}: dice={pc['dice']:.4f} "
                    f"prec={pc['precision']:.4f} rec={pc['recall']:.4f} "
                    f"iou={pc['iou']:.4f}")

            if is_main_process():
                if writer:
                    writer.add_scalar('loss/train', train_loss, epoch)
                    writer.add_scalar('loss/val', val_res['loss'], epoch)
                    writer.add_scalar('dice/mean', val_res['dice_mean'], epoch)
                    writer.add_scalar('precision/mean', val_res['precision_mean'], epoch)
                    writer.add_scalar('recall/mean', val_res['recall_mean'], epoch)
                    writer.add_scalar('iou/mean', val_res['iou_mean'], epoch)
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                    for c in classes:
                        writer.add_scalar(f'dice/{c}',
                                          val_res['per_class'][c]['dice'], epoch)
                        writer.add_scalar(f'precision/{c}',
                                          val_res['per_class'][c]['precision'], epoch)
                        writer.add_scalar(f'recall/{c}',
                                          val_res['per_class'][c]['recall'], epoch)
                        writer.add_scalar(f'iou/{c}',
                                          val_res['per_class'][c]['iou'], epoch)

                results.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_res['loss'],
                    'dice_mean': val_res['dice_mean'],
                    'precision_mean': val_res['precision_mean'],
                    'recall_mean': val_res['recall_mean'],
                    'iou_mean': val_res['iou_mean'],
                    'per_class': val_res['per_class'],
                })

                if val_res['dice_mean'] > best_dice:
                    best_dice = val_res['dice_mean']
                    ckpt = CHECKPOINT_DIR / f"{run_name}_best.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': base_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_dice': best_dice,
                        'strategy_name': strategy_name,
                    }, ckpt)
                    log(f"  ★ New best Dice={best_dice:.4f} -> {ckpt.name}")

    elapsed = time.time() - start_time
    log(f"\nDone in {elapsed/60:.1f} min -- Best Dice: {best_dice:.4f}")

    if is_main_process():
        # Find best epoch's precision/recall for the saved result
        best_epoch_data = {}
        for r in results:
            if abs(r.get('dice_mean', 0) - best_dice) < 1e-6:
                best_epoch_data = r
                break

        rpath = RESULTS_DIR / f"{run_name}_results.json"
        with open(rpath, 'w') as f:
            json.dump({
                'strategy_name': strategy_name,
                'description': strat_cfg.get('description', ''),
                'best_dice': best_dice,
                'best_precision': best_epoch_data.get('precision_mean', 0),
                'best_recall': best_epoch_data.get('recall_mean', 0),
                'best_iou': best_epoch_data.get('iou_mean', 0),
                'best_per_class': best_epoch_data.get('per_class', {}),
                'elapsed_min': elapsed / 60,
                'config': {k: str(v) if isinstance(v, Path) else v
                           for k, v in config.items()},
                'history': results,
            }, f, indent=2)
        log(f"Results -> {rpath}")

    if writer:
        writer.close()

    del model, optimizer, scheduler, scaler, criterion
    del train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    cleanup_ddp()
    return best_dice, results


# ======================================================================
# Comparison runner
# ======================================================================

def run_comparison(config: dict, resume: bool = True):
    """Run all masking strategies sequentially, skipping completed."""
    strategies = list(MASKING_CONFIGS.keys())
    all_results = {}

    if resume:
        for rfile in RESULTS_DIR.glob("mask_*_results.json"):
            try:
                with open(rfile) as f:
                    data = json.load(f)
                name = data.get('strategy_name')
                if name and name in MASKING_CONFIGS:
                    all_results[name] = {
                        'best_dice': data.get('best_dice', 0),
                        'best_precision': data.get('best_precision', 0),
                        'best_recall': data.get('best_recall', 0),
                        'best_iou': data.get('best_iou', 0),
                        'description': MASKING_CONFIGS[name]['description'],
                    }
            except (json.JSONDecodeError, KeyError):
                pass
        if all_results:
            log(f"Resuming: {len(all_results)} strategies done, "
                f"{len(strategies) - len(all_results)} remaining")

    for name in strategies:
        if name in all_results:
            log(f"\n  -> Skipping {name} (already done, "
                f"Dice={all_results[name]['best_dice']:.4f})")
            continue

        log(f"\n{'#'*60}")
        log(f"Testing: {name} ({len(all_results)+1}/{len(strategies)})")
        log(f"{'#'*60}")
        best_dice, _ = run_experiment(name, config)
        all_results[name] = {
            'best_dice': best_dice,
            'description': MASKING_CONFIGS[name]['description'],
        }

    if is_main_process():
        summary_path = RESULTS_DIR / "masking_comparison_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    print_summary_table()
    return all_results


def print_summary_table():
    """Load all result JSONs and print a comprehensive comparison table
    with Dice / Precision / Recall / IoU per class and overall."""
    ensure_dirs()
    result_files = sorted(RESULTS_DIR.glob("mask_*_results.json"))
    if not result_files:
        print("No result files found.")
        return

    # Deduplicate by strategy name, keeping best dice for each
    best_by_strategy = {}
    classes = None
    
    for rfile in result_files:
        try:
            with open(rfile) as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue
        name = data.get('strategy_name', rfile.stem)
        best_dice = data.get('best_dice', 0)
        elapsed = data.get('elapsed_min', 0)

        # Try to get per-class from the top-level best_per_class first
        per_class = data.get('best_per_class', {})

        # Fallback: find best epoch from history
        if not per_class:
            history = data.get('history', [])
            best_epoch = None
            for h in history:
                if abs(h.get('dice_mean', 0) - best_dice) < 1e-6:
                    best_epoch = h
                    break
            if best_epoch is None and history:
                best_epoch = max(history, key=lambda h: h.get('dice_mean', 0))
            if best_epoch and 'per_class' in best_epoch:
                per_class = best_epoch['per_class']

        mean_prec = data.get('best_precision', 0)
        mean_rec = data.get('best_recall', 0)
        mean_iou = data.get('best_iou', 0)

        # Recompute means from per-class if available
        if per_class:
            if classes is None:
                classes = list(per_class.keys())
            nc = len(per_class)
            mean_prec = sum(v.get('precision', 0) if isinstance(v, dict) else 0
                            for v in per_class.values()) / max(nc, 1)
            mean_rec = sum(v.get('recall', 0) if isinstance(v, dict) else 0
                           for v in per_class.values()) / max(nc, 1)
            mean_iou = sum(v.get('iou', 0) if isinstance(v, dict) else 0
                           for v in per_class.values()) / max(nc, 1)

        row_data = {
            'name': name,
            'dice': best_dice,
            'prec': mean_prec,
            'rec': mean_rec,
            'iou': mean_iou,
            'per_class': per_class,
            'elapsed': elapsed,
        }
        
        # Keep only the best result per strategy
        if name not in best_by_strategy or best_dice > best_by_strategy[name]['dice']:
            best_by_strategy[name] = row_data

    rows = list(best_by_strategy.values())

    if not rows:
        print("No valid results to display.")
        return

    if classes is None:
        classes = []

    rows.sort(key=lambda r: r['dice'], reverse=True)

    # ── Table 1: Overall summary ─────────────────────────────────────
    name_w = 28
    col_w = 8
    print(f"\n{'='*120}")
    print("MASKING STRATEGY COMPARISON — Overall Metrics (best epoch by Dice)")
    print(f"{'='*120}")

    hdr = (f"{'#':<4} {'Strategy':<{name_w}} "
           f"{'Dice':>{col_w}} {'Prec':>{col_w}} {'Rec':>{col_w}} {'IoU':>{col_w}} "
           f"{'Time(m)':>{col_w}}")
    print(hdr)
    print('-' * len(hdr))

    for i, r in enumerate(rows, 1):
        print(f"{i:<4} {r['name']:<{name_w}} "
              f"{r['dice']:>{col_w}.4f} {r['prec']:>{col_w}.4f} "
              f"{r['rec']:>{col_w}.4f} {r['iou']:>{col_w}.4f} "
              f"{r['elapsed']:>{col_w}.1f}")

    # ── Table 2: Per-class Dice ──────────────────────────────────────
    if classes:
        for metric_name, metric_key in [('Dice', 'dice'), ('Precision', 'precision'),
                                         ('Recall', 'recall'), ('IoU', 'iou')]:
            print(f"\n{'─'*120}")
            print(f"Per-class {metric_name}")
            print(f"{'─'*120}")

            hdr2 = f"{'#':<4} {'Strategy':<{name_w}}"
            for c in classes:
                hdr2 += f" {c[:10]:>{col_w+2}}"
            hdr2 += f" {'Mean':>{col_w+2}}"
            print(hdr2)
            print('-' * len(hdr2))

            for i, r in enumerate(rows, 1):
                line = f"{i:<4} {r['name']:<{name_w}}"
                vals = []
                for c in classes:
                    pc = r['per_class'].get(c, {})
                    v = pc.get(metric_key, 0) if isinstance(pc, dict) else 0
                    vals.append(v)
                    line += f" {v:>{col_w+2}.4f}"
                mean_v = sum(vals) / max(len(vals), 1)
                line += f" {mean_v:>{col_w+2}.4f}"
                print(line)

            # Best per class for this metric
            print(f"  Best per class:")
            for c in classes:
                best_r = max(rows, key=lambda r: r['per_class'].get(c, {}).get(metric_key, 0)
                              if isinstance(r['per_class'].get(c), dict) else 0)
                v = best_r['per_class'].get(c, {}).get(metric_key, 0) if isinstance(best_r['per_class'].get(c), dict) else 0
                print(f"    {c:<20s}: {v:.4f}  ({best_r['name']})")

    # ── Overall best ─────────────────────────────────────────────────
    best = rows[0]
    print(f"\n{'='*120}")
    print(f"★ Best overall: {best['name']}  "
          f"Dice={best['dice']:.4f}  Prec={best['prec']:.4f}  "
          f"Rec={best['rec']:.4f}  IoU={best['iou']:.4f}")
    print(f"{'='*120}\n")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Masking strategy experiments (BalSoftmax Tversky)')
    parser.add_argument('--mode', default='quick_test',
                        choices=['quick_test', 'comparison', 'single', 'summary'],
                        help='Experiment mode')
    parser.add_argument('--strategy', default='no_mask',
                        choices=list(MASKING_CONFIGS.keys()),
                        help='Strategy name (for single mode)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Override iterations_per_epoch')
    parser.add_argument('--single_gpu', action='store_true',
                        help='Force single-GPU (no DDP)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Override num_workers')
    parser.add_argument('--no_resume', action='store_true',
                        help='Do not skip already-completed strategies')

    args = parser.parse_args()

    if args.single_gpu or args.mode == 'single':
        os.environ['SINGLE_GPU_MODE'] = '1'
        if args.num_workers is not None:
            DATALOADER_CONFIG['num_workers'] = args.num_workers
            if args.num_workers == 0:
                DATALOADER_CONFIG['persistent_workers'] = False
        print(f"Single GPU mode")

    if args.mode == 'quick_test':
        config = get_config('quick_test')
    else:
        config = get_config('full')

    if args.epochs:
        config['epochs'] = args.epochs
    if args.iterations:
        config['iterations_per_epoch'] = args.iterations

    if args.mode == 'summary':
        ensure_dirs()
        print_summary_table()
        return
    elif args.mode == 'single':
        run_experiment(args.strategy, config)
    elif args.mode == 'comparison':
        run_comparison(config, resume=not args.no_resume)
    else:
        # quick_test: one from each family
        for name in ['no_mask', 'masksup_r0.3', 'regional_g8',
                      'entropy_mask', 'class_presence']:
            log(f"\n>>> Quick test: {name}")
            run_experiment(name, config)


if __name__ == '__main__':
    main()
