#!/usr/bin/env python3
"""
Training script for class-weighting experiments.

Fixes the loss function to per-class Tversky (α=0.6, β=0.4) and compares
class weighting strategies: uniform, manual, inverse-frequency, sqrt-inverse,
log-inverse, effective-number, Class-Balanced, Balanced Softmax, and Seesaw.

Usage:
    # Quick smoke test (5 epochs, ~5 min)
    python train.py --mode quick_test

    # Run all 15 weighting configs sequentially
    python train.py --mode weighting_comparison

    # Single loss on single GPU (for parallel launches)
    python train.py --mode single --loss cb_beta_0.999

    # DDP (multi-GPU)
    torchrun --nproc_per_node=2 train.py --mode weighting_comparison
"""

# ── Thread limits (BEFORE any other imports) ──────────────────────────
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
import socket
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision.transforms.v2 as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from cellmap_data.transforms.augment import NaNtoNum

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))        # repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "loss_optimization"))  # shared utils
sys.path.insert(0, str(Path(__file__).parent))                       # this experiment

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = True   # fixed input size → free speedup

# ── Config import ─────────────────────────────────────────────────────
from config import (
    LOSS_CONFIGS, QUICK_TEST_CLASSES,
    CHECKPOINT_DIR, TENSORBOARD_DIR, RESULTS_DIR,
    SPATIAL_TRANSFORMS_2D, DATALOADER_CONFIG, VALIDATION_CONFIG,
    MODEL_CONFIG, USE_AMP, MAX_GRAD_NORM, ESTIMATED_VOXEL_COUNTS,
    DATA_ROOT, EXPERIMENT_DIR,
    get_config, ensure_dirs, get_device,
)
from losses_class_weighting import get_weighting_loss


# ======================================================================
# DDP helpers
# ======================================================================

def setup_ddp():
    if os.environ.get('SINGLE_GPU_MODE') == '1':
        # Single-GPU mode: use cuda:0 (which maps to whatever
        # physical GPU CUDA_VISIBLE_DEVICES points to)
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
# Model
# ======================================================================

def create_model(n_classes: int, input_channels: int = 1):
    from cellmap_segmentation_challenge.models import UNet_2D
    return UNet_2D(input_channels, n_classes)


# ======================================================================
# Data
# ======================================================================

def create_dataloaders(classes, batch_size, iterations_per_epoch,
                       input_shape=(1, 256, 256)):
    from cellmap_segmentation_challenge.utils.dataloader import get_dataloader

    datasplit_path = Path(__file__).parent / "datasplit.csv"
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
        # DDP with this dataloader requires num_workers=0
        dl_kwargs['num_workers'] = 0
        dl_kwargs['persistent_workers'] = False
        log("DDP mode: forcing num_workers=0")
    else:
        # Single-GPU: use configured workers for throughput
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
# Metrics
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
# Train / Validate
# ======================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, scheduler,
                device, epoch, classes=None):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        inputs = batch['input'].to(device)
        if inputs.dim() == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        targets = batch['output'].to(device)

        optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
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
        scheduler.step()  # OneCycleLR steps per BATCH, not per epoch

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
        per_class[c] = {'dice': dice, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}

    mean_dice = sum(m['dice'] for m in per_class.values()) / len(classes)
    return {
        'loss': total_loss / max(n_batches, 1),
        'dice_mean': mean_dice,
        'per_class': per_class,
    }


# ======================================================================
# Experiment runner
# ======================================================================

def run_experiment(loss_name: str, config: dict, run_name: str = None):
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
        run_name = f"cw_{loss_name}_{ts}"

    log(f"\n{'='*60}")
    log(f"Experiment: {run_name}")
    log(f"Loss config: {loss_name}")
    log(f"Classes: {classes}")
    log(f"Batch: {batch_size}  Input: {input_shape}")
    log(f"{'='*60}\n")

    # ── Model ─────────────────────────────────────────────────────
    model = create_model(n_classes, input_channels).to(device)
    # torch.compile for Ampere+ GPUs (3090 = Ampere)
    try:
        model = torch.compile(model, mode='reduce-overhead')
        log("  torch.compile enabled (reduce-overhead)")
    except Exception as e:
        log(f"  torch.compile unavailable: {e}")
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(
        classes=classes,
        batch_size=batch_size,
        iterations_per_epoch=config['iterations_per_epoch'],
        input_shape=input_shape,
    )

    # ── Loss ──────────────────────────────────────────────────────
    loss_cfg = LOSS_CONFIGS[loss_name]
    loss_kwargs = {k: v for k, v in loss_cfg.items()
                   if k not in ('type', 'description')}
    criterion = get_weighting_loss(
        loss_type=loss_cfg['type'],
        classes=classes,
        class_voxel_counts=ESTIMATED_VOXEL_COUNTS,
        **loss_kwargs,
    ).to(device)
    log(f"Loss: {loss_cfg.get('description', loss_name)}")

    # ── Optimizer / Scheduler ─────────────────────────────────────
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

    # ── Training loop ─────────────────────────────────────────────
    best_dice = 0
    results = []
    start_time = time.time()

    for epoch in range(1, config['epochs'] + 1):
        log(f"\nEpoch {epoch}/{config['epochs']}")

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            scheduler, device, epoch, classes=classes)
        # NOTE: scheduler.step() is called per-batch inside train_epoch
        log(f"  Train Loss: {train_loss:.4f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if epoch % config.get('validate_every', 1) == 0:
            val_res = validate(model, val_loader, criterion, device, classes)
            log(f"  Val Loss: {val_res['loss']:.4f}  Dice: {val_res['dice_mean']:.4f}")
            for c in classes:
                d = val_res['per_class'][c]['dice']
                log(f"    {c}: {d:.4f}")

            if is_main_process():
                if writer:
                    writer.add_scalar('loss/train', train_loss, epoch)
                    writer.add_scalar('loss/val', val_res['loss'], epoch)
                    writer.add_scalar('dice/mean', val_res['dice_mean'], epoch)
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                    for c in classes:
                        writer.add_scalar(f'dice/{c}',
                                          val_res['per_class'][c]['dice'], epoch)

                results.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_res['loss'],
                    'dice_mean': val_res['dice_mean'],
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
                        'loss_name': loss_name,
                    }, ckpt)
                    log(f"  ★ New best Dice={best_dice:.4f} → {ckpt.name}")

    elapsed = time.time() - start_time
    log(f"\nDone in {elapsed/60:.1f} min — Best Dice: {best_dice:.4f}")

    if is_main_process():
        rpath = RESULTS_DIR / f"{run_name}_results.json"
        with open(rpath, 'w') as f:
            json.dump({
                'loss_name': loss_name,
                'description': loss_cfg.get('description', ''),
                'best_dice': best_dice,
                'elapsed_min': elapsed / 60,
                'config': {k: str(v) if isinstance(v, Path) else v
                           for k, v in config.items()},
                'history': results,
            }, f, indent=2)
        log(f"Results → {rpath}")

    if writer:
        writer.close()

    # Heavy cleanup between experiments (not every epoch)
    del model, optimizer, scheduler, scaler, criterion
    del train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    cleanup_ddp()
    return best_dice, results


# ======================================================================
# Comparison runner
# ======================================================================

def run_weighting_comparison(config: dict, resume: bool = True):
    """Run all configured losses sequentially, skipping completed ones."""
    losses_to_test = list(LOSS_CONFIGS.keys())

    all_results = {}

    # Load any previously completed results (for resume / fault tolerance)
    if resume:
        for rfile in RESULTS_DIR.glob("cw_*_results.json"):
            try:
                with open(rfile) as f:
                    data = json.load(f)
                name = data.get('loss_name')
                if name and name in LOSS_CONFIGS:
                    all_results[name] = {
                        'best_dice': data.get('best_dice', 0),
                        'description': LOSS_CONFIGS[name]['description'],
                    }
            except (json.JSONDecodeError, KeyError):
                pass
        if all_results:
            log(f"Resuming: {len(all_results)} configs already done, "
                f"{len(losses_to_test) - len(all_results)} remaining")

    for loss_name in losses_to_test:
        if loss_name in all_results:
            log(f"\n  ⏭ Skipping {loss_name} (already completed, "
                f"Dice={all_results[loss_name]['best_dice']:.4f})")
            continue

        log(f"\n{'#'*60}")
        log(f"Testing: {loss_name} "
            f"({len(all_results)+1}/{len(losses_to_test)})")
        log(f"{'#'*60}")
        best_dice, _ = run_experiment(loss_name, config)
        all_results[loss_name] = {
            'best_dice': best_dice,
            'description': LOSS_CONFIGS[loss_name]['description'],
        }

    if is_main_process():
        summary_path = RESULTS_DIR / "comparison_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    print_summary_table()
    return all_results


def print_summary_table():
    """Load all result JSONs and print a formatted per-class comparison table."""
    ensure_dirs()
    result_files = sorted(RESULTS_DIR.glob("cw_*_results.json"))
    if not result_files:
        print("No result files found.")
        return

    # Collect per-config best-epoch per-class dice
    rows = []  # (loss_name, mean_dice, {class: dice}, elapsed_min)
    classes = None
    for rfile in result_files:
        try:
            with open(rfile) as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue
        loss_name = data.get('loss_name', rfile.stem)
        best_dice = data.get('best_dice', 0)
        elapsed = data.get('elapsed_min', 0)

        # Find the epoch with the best mean dice
        history = data.get('history', [])
        best_epoch = None
        for h in history:
            if abs(h.get('dice_mean', 0) - best_dice) < 1e-6:
                best_epoch = h
                break
        if best_epoch is None and history:
            best_epoch = max(history, key=lambda h: h.get('dice_mean', 0))

        per_class = {}
        if best_epoch and 'per_class' in best_epoch:
            if classes is None:
                classes = list(best_epoch['per_class'].keys())
            for c, v in best_epoch['per_class'].items():
                per_class[c] = v.get('dice', 0) if isinstance(v, dict) else v

        rows.append((loss_name, best_dice, per_class, elapsed))

    if not rows:
        print("No valid results to display.")
        return

    if classes is None:
        classes = []

    # Sort by mean dice descending
    rows.sort(key=lambda r: r[1], reverse=True)

    # ── Print table ───────────────────────────────────────────────
    col_w = 10  # width for each dice column
    name_w = 32
    print(f"\n{'='*100}")
    print("CLASS WEIGHTING COMPARISON — PER-CLASS DICE (best epoch)")
    print(f"{'='*100}")

    # Header
    hdr = f"{'Rank':<5} {'Loss Config':<{name_w}} {'Mean':>{col_w}}"
    for c in classes:
        hdr += f" {c:>{col_w}}"
    hdr += f" {'Time(m)':>{col_w}}"
    print(hdr)
    print('─' * len(hdr))

    # Rows
    for i, (name, mean_d, pc, elapsed) in enumerate(rows, 1):
        row = f"{i:<5} {name:<{name_w}} {mean_d:>{col_w}.4f}"
        for c in classes:
            d = pc.get(c, 0)
            row += f" {d:>{col_w}.4f}"
        row += f" {elapsed:>{col_w}.1f}"
        print(row)

    # ── Best per class ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Best per class:")
    for c in classes:
        best_row = max(rows, key=lambda r: r[2].get(c, 0))
        d = best_row[2].get(c, 0)
        print(f"  {c:<20s}: {d:.4f}  ({best_row[0]})")

    best_overall = rows[0]
    print(f"\n★ Best overall: {best_overall[0]}  "
          f"Mean Dice = {best_overall[1]:.4f}")
    print(f"{'='*100}\n")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Class-weighting experiments (Tversky α=0.6, β=0.4)')
    parser.add_argument('--mode', default='quick_test',
                        choices=['quick_test', 'weighting_comparison', 'single',
                                 'summary'],
                        help='Experiment mode')
    parser.add_argument('--loss', default='weight_uniform',
                        choices=list(LOSS_CONFIGS.keys()),
                        help='Loss config name (for single mode)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs')
    parser.add_argument('--single_gpu', action='store_true',
                        help='Force single-GPU (no DDP)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Override num_workers')
    parser.add_argument('--no_resume', action='store_true',
                        help='Do not skip already-completed configs')

    args = parser.parse_args()

    if args.single_gpu or args.mode == 'single':
        os.environ['SINGLE_GPU_MODE'] = '1'
        if args.num_workers is not None:
            DATALOADER_CONFIG['num_workers'] = args.num_workers
            if args.num_workers == 0:
                DATALOADER_CONFIG['persistent_workers'] = False
        print(f"🚀 Single GPU mode")

    mode = args.mode if args.mode != 'single' else 'weighting_comparison'
    config = get_config(mode)

    if args.epochs:
        config['epochs'] = args.epochs

    if args.mode == 'summary':
        ensure_dirs()
        print_summary_table()
        return
    elif args.mode == 'single':
        run_experiment(args.loss, config)
    elif args.mode == 'weighting_comparison':
        run_weighting_comparison(config, resume=not args.no_resume)
    else:
        # quick_test — one from each family
        for loss_name in ['weight_uniform', 'weight_manual',
                          'cb_beta_0.999', 'balanced_softmax_tau_1.0',
                          'seesaw_default']:
            log(f"\n>>> Quick test: {loss_name}")
            run_experiment(loss_name, config)


if __name__ == '__main__':
    main()
