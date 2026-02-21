#!/usr/bin/env python3
"""
Training script for class-weighted model comparison.

Trains UNet 2D, ResNet 2D, Swin 2D, and ViT 2D with the best class-weighting
strategy from the class_weighting experiment (Balanced Softmax Tversky τ=1.0).

Uses the same data pipeline as class_weighting/train.py:
  - NaN-safe float32 normalization (no Binarize on targets)
  - NaNtoNum on raw values
  - Same spatial transforms

Adds feature extraction, metrics tracking, and rich visualization from
the model_comparison experiment.

Usage:
    # Single model on single GPU
    python train.py --model unet

    # With DDP (2 A100s on Blanca)
    torchrun --standalone --nproc_per_node=2 train.py --model swin

    # Print summary table
    python train.py --summary
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
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.v2 as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from cellmap_data.transforms.augment import NaNtoNum

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent))          # class_weighting/ (losses)
sys.path.insert(0, str(SCRIPT_DIR))                  # this dir (config)
sys.path.insert(0, str(REPO_ROOT / "experiments" / "model_comparison"))  # feature_extractor, metrics_tracker

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ── Imports from this experiment's config ─────────────────────────────
from config import (
    CLASSES, N_CLASSES, BATCH_SIZE, INPUT_SHAPE, INPUT_CHANNELS, SCALE,
    MODEL_REGISTRY, LOSS_CONFIG, ESTIMATED_VOXEL_COUNTS,
    CHECKPOINT_DIR, TENSORBOARD_DIR, RESULTS_DIR, VISUALIZATIONS_DIR,
    METRICS_DIR, FEATURES_DIR,
    SPATIAL_TRANSFORMS_2D, DATALOADER_CONFIG, VALIDATION_CONFIG,
    TRAINING_CONFIG, USE_AMP, MAX_GRAD_NORM,
    VISUALIZATION_SEED, VIS_EVERY, CHECKPOINT_EVERY, VIS_SAMPLES,
    ensure_dirs, get_device,
)

# Loss from class_weighting experiment
from losses_class_weighting import get_weighting_loss

# Feature extraction & metrics from model_comparison experiment
from feature_extractor import FeatureExtractor, visualize_feature_maps
from metrics_tracker import MetricsTracker


# ======================================================================
# DDP helpers
# ======================================================================

def setup_ddp():
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


def is_main():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def log(msg):
    if is_main():
        print(msg, flush=True)


# ======================================================================
# Model factory
# ======================================================================

def create_model(model_name: str):
    """Instantiate model from MODEL_REGISTRY."""
    from cellmap_segmentation_challenge.models import (
        UNet_2D, ResNet, SwinTransformer, ViTVNet2D,
    )

    reg = MODEL_REGISTRY[model_name]
    cls_name = reg['class']
    cfg = reg['config']

    if cls_name == 'UNet_2D':
        return UNet_2D(**cfg)
    elif cls_name == 'ResNet':
        return ResNet(**cfg)
    elif cls_name == 'SwinTransformer':
        return SwinTransformer(**cfg)
    elif cls_name == 'ViTVNet2D':
        return ViTVNet2D(config=cfg, in_channels=INPUT_CHANNELS, num_classes=N_CLASSES)
    else:
        raise ValueError(f"Unknown model class: {cls_name}")


# ======================================================================
# Data  (same pipeline as class_weighting/train.py)
# ======================================================================

def create_dataloaders(batch_size, iterations_per_epoch):
    from cellmap_segmentation_challenge.utils.dataloader import get_dataloader

    datasplit_path = SCRIPT_DIR / "datasplit.csv"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not datasplit_path.exists():
        if local_rank == 0:
            log("Creating datasplit.csv …")
            from cellmap_segmentation_challenge.utils.datasplit import make_datasplit_csv
            make_datasplit_csv(
                classes=CLASSES,
                csv_path=str(datasplit_path),
                validation_prob=0.15,
                force_all_classes=False,
            )
        if dist.is_initialized():
            dist.barrier()
        else:
            while not datasplit_path.exists():
                time.sleep(0.5)

    input_array_info  = {"shape": INPUT_SHAPE, "scale": SCALE}
    target_array_info = {"shape": INPUT_SHAPE, "scale": SCALE}

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
        dl_kwargs.pop('prefetch_factor', None)
        log("DDP mode: forcing num_workers=0")
    else:
        if dl_kwargs.get('prefetch_factor') is None:
            dl_kwargs.pop('prefetch_factor', None)
        log(f"DataLoader: num_workers={dl_kwargs.get('num_workers', 0)}")

    train_loader, val_loader = get_dataloader(
        datasplit_path=str(datasplit_path),
        classes=CLASSES,
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
# Metrics helpers
# ======================================================================

def compute_batch_counts(pred, target):
    """Compute TP/FP/FN per class (NaN-safe)."""
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
# Visualization
# ======================================================================

def get_fixed_samples(val_loader, num_samples=VIS_SAMPLES, seed=VISUALIZATION_SEED):
    """Get/load fixed validation samples shared across all models."""
    save_path = VISUALIZATIONS_DIR / "fixed_samples_2d.pt"

    if save_path.exists():
        log(f"Loading fixed samples from {save_path}")
        return torch.load(save_path, weights_only=True)

    log(f"Generating {num_samples} fixed visualization samples (seed={seed}) …")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    samples = {'inputs': [], 'targets': []}
    val_loader.refresh()
    loader_iter = iter(val_loader.loader)
    collected = 0

    for batch in loader_iter:
        if collected >= num_samples:
            break
        inp = batch['input']
        tgt = batch['output']
        samples['inputs'].append(inp[0:1])
        samples['targets'].append(tgt[0:1])
        collected += 1

    samples['inputs']  = torch.cat(samples['inputs'],  dim=0)
    samples['targets'] = torch.cat(samples['targets'], dim=0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, save_path)
    log(f"Saved {collected} fixed samples → {save_path}")
    return samples


def save_visualization(model, fixed_samples, epoch, model_name, device,
                       feature_extractor=None, writer=None):
    """Save prediction visualizations and feature maps."""
    model.eval()
    with torch.no_grad():
        inputs  = fixed_samples['inputs'].to(device)
        targets = fixed_samples['targets'].to(device)

        # Handle extra dim from dataloader
        if inputs.dim() == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)

        outputs = model(inputs)
        preds   = torch.sigmoid(outputs)

        # Feature maps
        features = None
        if feature_extractor is not None:
            try:
                features = feature_extractor(inputs)
            except Exception as e:
                log(f"  Feature extraction failed: {e}")

    for idx in range(inputs.shape[0]):
        fig = _comparison_figure(inputs[idx], targets[idx], preds[idx])
        fpath = VISUALIZATIONS_DIR / f"{model_name}_e{epoch:04d}_s{idx}.png"
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        if writer:
            writer.add_figure(f'samples/sample_{idx}', fig, epoch)
        plt.close(fig)

    if features is not None:
        try:
            feat_fig = visualize_feature_maps(features, batch_idx=0, max_channels=8)
            fpath = VISUALIZATIONS_DIR / f"{model_name}_e{epoch:04d}_features.png"
            feat_fig.savefig(fpath, dpi=150, bbox_inches='tight')
            if writer:
                writer.add_figure('features/feature_maps', feat_fig, epoch)
            plt.close(feat_fig)
        except Exception as e:
            log(f"  Feature map viz failed: {e}")


def _comparison_figure(inp, target, pred):
    """Create comparison figure for one sample: input, GT, pred per class."""
    n_classes = target.shape[0]
    fig_h = 2.5 * (n_classes + 1)
    fig, axes = plt.subplots(n_classes + 1, 3, figsize=(12, fig_h))

    # Input (row 0)
    if inp.dim() == 3:
        img = inp[0].cpu().numpy()
    else:
        img = inp.cpu().numpy()
    img = img.astype(np.float32)

    for col in range(3):
        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].set_title(['Input', 'Ground Truth', 'Prediction'][col],
                               fontsize=12, fontweight='bold')
        axes[0, col].axis('off')
    axes[0, 0].set_ylabel('Raw', fontsize=10, fontweight='bold')

    for i, cls in enumerate(CLASSES[:n_classes]):
        row = i + 1
        axes[row, 0].imshow(img, cmap='gray', alpha=0.3)
        axes[row, 0].set_ylabel(cls, fontsize=9, fontweight='bold')
        axes[row, 0].axis('off')

        gt = np.nan_to_num(target[i].cpu().numpy(), nan=0).astype(np.float32)
        axes[row, 1].imshow(gt, cmap='hot', vmin=0, vmax=1)
        if gt.sum() == 0:
            axes[row, 1].text(0.5, 0.5, 'No GT', transform=axes[row, 1].transAxes,
                             ha='center', va='center', fontsize=8, color='white',
                             bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        axes[row, 1].axis('off')

        p = pred[i].cpu().numpy().astype(np.float32)
        axes[row, 2].imshow(p, cmap='hot', vmin=0, vmax=1)
        axes[row, 2].text(0.02, 0.98, f'max:{p.max():.2f}',
                         transform=axes[row, 2].transAxes, ha='left', va='top',
                         fontsize=7, color='white',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.3))
        axes[row, 2].axis('off')

    plt.tight_layout()
    return fig


# ======================================================================
# Train / Validate
# ======================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, scheduler,
                device, epoch, writer=None, n_iter_start=0):
    model.train()
    total_loss = 0
    n_batches = 0
    n_iter = n_iter_start

    for batch_idx, batch in enumerate(loader):
        inputs  = batch['input'].to(device)
        if inputs.dim() == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        targets = batch['output'].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            if is_main():
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
        scheduler.step()  # OneCycleLR: step per batch

        total_loss += loss.item()
        n_batches += 1
        n_iter += 1

        if writer and batch_idx % 10 == 0:
            writer.add_scalar('train/loss_iter', loss.item(), n_iter)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], n_iter)

        if batch_idx % 50 == 0 and is_main():
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Batch {batch_idx}/{len(loader)}, "
                  f"Loss: {loss.item():.4f}, LR: {lr_now:.2e}")

        del inputs, targets, outputs, loss

    return total_loss / max(n_batches, 1), n_iter


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0

    global_tp = [0.0] * N_CLASSES
    global_fp = [0.0] * N_CLASSES
    global_fn = [0.0] * N_CLASSES

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= VALIDATION_CONFIG['batch_limit']:
            break

        inputs  = batch['input'].to(device)
        if inputs.dim() == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        targets = batch['output'].to(device)

        with torch.amp.autocast('cuda', enabled=USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Sigmoid stats on first batch
        if batch_idx == 0 and is_main():
            sigmoid_out = torch.sigmoid(outputs)
            print("  [Sigmoid stats per class]")
            for i, c in enumerate(CLASSES):
                vals = sigmoid_out[:, i].flatten()
                print(f"    {c}: min={vals.min():.4f}, max={vals.max():.4f}, "
                      f"mean={vals.mean():.4f}, >0.5: {(vals > 0.5).float().mean()*100:.1f}%")

        total_loss += loss.item()
        n_batches += 1

        counts = compute_batch_counts(outputs.detach(), targets.detach())
        for i in range(N_CLASSES):
            global_tp[i] += counts['tp'][i]
            global_fp[i] += counts['fp'][i]
            global_fn[i] += counts['fn'][i]

        del inputs, targets, outputs, loss

    per_class = {}
    for i, c in enumerate(CLASSES):
        tp, fp, fn = global_tp[i], global_fp[i], global_fn[i]
        denom = 2 * tp + fp + fn
        dice = (2 * tp / denom) if denom > 0 else 0.0
        iou  = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec  = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        per_class[c] = {
            'dice': dice, 'iou': iou, 'precision': prec, 'recall': rec,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        }

    mean_dice = sum(m['dice'] for m in per_class.values()) / N_CLASSES
    mean_iou  = sum(m['iou']  for m in per_class.values()) / N_CLASSES
    return {
        'loss': total_loss / max(n_batches, 1),
        'dice_mean': mean_dice,
        'iou_mean': mean_iou,
        'per_class': per_class,
    }


# ======================================================================
# Main training loop
# ======================================================================

def run_training(model_name: str, resume: bool = False, no_compile: bool = False):
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    ensure_dirs()

    reg = MODEL_REGISTRY[model_name]
    lr = reg['lr']
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"mc_{model_name}_{ts}"

    log(f"\n{'='*70}")
    log(f"  MODEL COMPARISON — {model_name.upper()}")
    log(f"  Loss: {LOSS_CONFIG['description']}")
    log(f"  Batch: {BATCH_SIZE}  Input: {INPUT_SHAPE}  LR: {lr}")
    log(f"  Epochs: {TRAINING_CONFIG['epochs']}  Iter/epoch: {TRAINING_CONFIG['iterations_per_epoch']}")
    log(f"  Device: {device}  World size: {world_size}")
    log(f"{'='*70}\n")

    # ── Model ─────────────────────────────────────────────────────
    model = create_model(model_name).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # torch.compile
    if not no_compile:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            log("torch.compile enabled (reduce-overhead)")
        except Exception as e:
            log(f"torch.compile failed: {e}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=(reg['type'] == 'transformer'))

    base_model = model.module if hasattr(model, 'module') else model

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        iterations_per_epoch=TRAINING_CONFIG['iterations_per_epoch'],
    )

    # ── Loss ──────────────────────────────────────────────────────
    loss_kwargs = {k: v for k, v in LOSS_CONFIG.items()
                   if k not in ('type', 'description')}
    criterion = get_weighting_loss(
        loss_type=LOSS_CONFIG['type'],
        classes=CLASSES,
        class_voxel_counts=ESTIMATED_VOXEL_COUNTS,
        **loss_kwargs,
    ).to(device)

    # ── Optimizer / Scheduler ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        base_model.parameters(), lr=lr,
        weight_decay=TRAINING_CONFIG['weight_decay'],
        betas=TRAINING_CONFIG['betas'],
    )
    total_steps = TRAINING_CONFIG['epochs'] * TRAINING_CONFIG['iterations_per_epoch']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.05, anneal_strategy='cos',
    )
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    # ── Metrics tracker ───────────────────────────────────────────
    metrics_tracker = MetricsTracker(
        classes=CLASSES,
        save_path=str(METRICS_DIR),
        model_name=model_name,
    )

    # ── Feature extractor ─────────────────────────────────────────
    feature_extractor = None
    if is_main():
        try:
            feature_extractor = FeatureExtractor(model)
            log(f"Feature extractor: {len(feature_extractor.layer_names)} layers")
        except Exception as e:
            log(f"Feature extractor init failed: {e}")

    # ── TensorBoard ───────────────────────────────────────────────
    writer = None
    if is_main():
        writer = SummaryWriter(str(TENSORBOARD_DIR / run_name))

    # ── Fixed visualization samples ───────────────────────────────
    fixed_samples = None
    if is_main():
        fixed_samples = get_fixed_samples(val_loader)
        log(f"Fixed samples shape: {fixed_samples['inputs'].shape}")

    # ── Resume from checkpoint ────────────────────────────────────
    start_epoch = 1
    best_dice = 0.0
    n_iter = 0

    if resume:
        ckpt_path = CHECKPOINT_DIR / f"mc_{model_name}_latest.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            base_model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_dice = ckpt.get('best_dice', 0.0)
            n_iter = ckpt.get('n_iter', 0)
            log(f"Resumed from epoch {ckpt['epoch']}, best_dice={best_dice:.4f}")

    # ── Training loop ─────────────────────────────────────────────
    results = []
    start_time = time.time()

    for epoch in range(start_epoch, TRAINING_CONFIG['epochs'] + 1):
        log(f"\nEpoch {epoch}/{TRAINING_CONFIG['epochs']}")
        metrics_tracker.start_epoch()

        train_loss, n_iter = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            scheduler, device, epoch, writer=writer, n_iter_start=n_iter,
        )
        log(f"  Train Loss: {train_loss:.4f}  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ── Validation ────────────────────────────────────────────
        if epoch % TRAINING_CONFIG.get('validate_every', 1) == 0:
            val_res = validate(model, val_loader, criterion, device)
            log(f"  Val Loss: {val_res['loss']:.4f}  Dice: {val_res['dice_mean']:.4f}  "
                f"IoU: {val_res['iou_mean']:.4f}")
            for c in CLASSES:
                d = val_res['per_class'][c]
                log(f"    {c}: dice={d['dice']:.4f}  iou={d['iou']:.4f}  "
                    f"prec={d['precision']:.4f}  rec={d['recall']:.4f}")

            if is_main():
                # TensorBoard
                if writer:
                    writer.add_scalar('loss/train', train_loss, epoch)
                    writer.add_scalar('loss/val', val_res['loss'], epoch)
                    writer.add_scalar('dice/mean', val_res['dice_mean'], epoch)
                    writer.add_scalar('iou/mean', val_res['iou_mean'], epoch)
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                    for c in CLASSES:
                        d = val_res['per_class'][c]
                        writer.add_scalar(f'dice/{c}', d['dice'], epoch)
                        writer.add_scalar(f'iou/{c}', d['iou'], epoch)
                        writer.add_scalar(f'precision/{c}', d['precision'], epoch)
                        writer.add_scalar(f'recall/{c}', d['recall'], epoch)

                # Track results
                results.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_res['loss'],
                    'dice_mean': val_res['dice_mean'],
                    'iou_mean': val_res['iou_mean'],
                    'per_class': val_res['per_class'],
                })

                # Best checkpoint
                if val_res['dice_mean'] > best_dice:
                    best_dice = val_res['dice_mean']
                    ckpt = CHECKPOINT_DIR / f"mc_{model_name}_best.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': base_model.state_dict(),
                        'best_dice': best_dice,
                        'model_name': model_name,
                    }, ckpt)
                    log(f"  ★ New best Dice={best_dice:.4f} → {ckpt.name}")

                # Periodic checkpoint
                if epoch % CHECKPOINT_EVERY == 0:
                    ckpt = CHECKPOINT_DIR / f"mc_{model_name}_epoch{epoch:04d}.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': base_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_dice': best_dice,
                        'n_iter': n_iter,
                    }, ckpt)

                # Latest checkpoint (for resume)
                latest_ckpt = CHECKPOINT_DIR / f"mc_{model_name}_latest.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': base_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_dice': best_dice,
                    'n_iter': n_iter,
                }, latest_ckpt)

                # Visualizations
                if epoch % VIS_EVERY == 0 or epoch == 1:
                    save_visualization(
                        model, fixed_samples, epoch, model_name, device,
                        feature_extractor=feature_extractor, writer=writer,
                    )

        # End-of-epoch metrics
        metrics_tracker.end_epoch(epoch, learning_rate=optimizer.param_groups[0]['lr'])
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    log(f"\n{'='*70}")
    log(f"  {model_name.upper()} DONE — {elapsed/60:.1f} min — Best Dice: {best_dice:.4f}")
    log(f"{'='*70}\n")

    # ── Save results ──────────────────────────────────────────────
    if is_main():
        metrics_tracker.save()

        rpath = RESULTS_DIR / f"mc_{model_name}_results.json"
        with open(rpath, 'w') as f:
            json.dump({
                'model_name': model_name,
                'model_type': reg['type'],
                'model_class': reg['class'],
                'total_params': total_params,
                'trainable_params': trainable_params,
                'loss': LOSS_CONFIG['description'],
                'batch_size': BATCH_SIZE,
                'learning_rate': lr,
                'best_dice': best_dice,
                'elapsed_min': elapsed / 60,
                'epochs': TRAINING_CONFIG['epochs'],
                'iterations_per_epoch': TRAINING_CONFIG['iterations_per_epoch'],
                'classes': CLASSES,
                'history': results,
            }, f, indent=2)
        log(f"Results → {rpath}")

        # Save features for best model
        if feature_extractor:
            try:
                from feature_extractor import extract_and_save_features
                feat_path = FEATURES_DIR / f"{model_name}_features.npz"
                # Just save the feature extractor info
                log(f"Feature extractor layers: {feature_extractor.layer_names}")
            except Exception:
                pass
            feature_extractor.remove_hooks()

    if writer:
        writer.close()

    del model, optimizer, scheduler, scaler, criterion
    del train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    cleanup_ddp()

    return best_dice, results


# ======================================================================
# Summary table
# ======================================================================

def print_summary_table():
    """Load all result JSONs and print a formatted comparison table."""
    ensure_dirs()
    result_files = sorted(RESULTS_DIR.glob("mc_*_results.json"))
    if not result_files:
        print("No result files found in", RESULTS_DIR)
        return

    rows = []
    classes = None
    for rfile in result_files:
        try:
            with open(rfile) as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue

        model_name = data.get('model_name', rfile.stem)
        model_type = data.get('model_type', '?')
        params = data.get('trainable_params', 0)
        best_dice = data.get('best_dice', 0)
        elapsed = data.get('elapsed_min', 0)

        history = data.get('history', [])
        best_epoch = None
        for h in history:
            if abs(h.get('dice_mean', 0) - best_dice) < 1e-6:
                best_epoch = h
                break
        if best_epoch is None and history:
            best_epoch = max(history, key=lambda h: h.get('dice_mean', 0))

        per_class = {}
        iou_mean = 0
        if best_epoch and 'per_class' in best_epoch:
            if classes is None:
                classes = list(best_epoch['per_class'].keys())
            for c, v in best_epoch['per_class'].items():
                per_class[c] = v if isinstance(v, dict) else {'dice': v}
            iou_mean = best_epoch.get('iou_mean', 0)

        rows.append((model_name, model_type, params, best_dice, iou_mean, per_class, elapsed))

    if not rows:
        print("No valid results to display.")
        return
    if classes is None:
        classes = CLASSES

    rows.sort(key=lambda r: r[3], reverse=True)

    # Header
    col_w = 10
    name_w = 12
    print(f"\n{'='*120}")
    print("MODEL COMPARISON — Balanced Softmax Tversky τ=1.0 — PER-CLASS DICE (best epoch)")
    print(f"{'='*120}")

    hdr = f"{'Rank':<5} {'Model':<{name_w}} {'Type':<6} {'Params':>10} {'MeanDice':>{col_w}} {'MeanIoU':>{col_w}}"
    for c in classes:
        hdr += f" {c:>{col_w}}"
    hdr += f" {'Time(m)':>{col_w}}"
    print(hdr)
    print('─' * len(hdr))

    for i, (name, mtype, params, md, miou, pc, elapsed) in enumerate(rows, 1):
        row = (f"{i:<5} {name:<{name_w}} {mtype:<6} {params:>10,} "
               f"{md:>{col_w}.4f} {miou:>{col_w}.4f}")
        for c in classes:
            d = pc.get(c, {}).get('dice', 0) if isinstance(pc.get(c), dict) else pc.get(c, 0)
            row += f" {d:>{col_w}.4f}"
        row += f" {elapsed:>{col_w}.1f}"
        print(row)

    # Best per class
    print(f"\n{'─'*60}")
    print("Best per class:")
    for c in classes:
        best_row = max(rows, key=lambda r: (r[5].get(c, {}).get('dice', 0)
                                            if isinstance(r[5].get(c), dict) else 0))
        d = best_row[5].get(c, {}).get('dice', 0) if isinstance(best_row[5].get(c), dict) else 0
        print(f"  {c:<20s}: {d:.4f}  ({best_row[0]})")

    best_overall = rows[0]
    print(f"\n★ Best overall: {best_overall[0]}  Mean Dice = {best_overall[3]:.4f}")
    print(f"{'='*120}\n")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Model comparison with Balanced Softmax Tversky τ=1.0')
    parser.add_argument('--model', choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture')
    parser.add_argument('--summary', action='store_true',
                        help='Print comparison summary table')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile()')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Override num_workers')
    args = parser.parse_args()

    if args.summary:
        print_summary_table()
        return

    if args.model is None:
        parser.error("--model is required (unless using --summary)")

    if args.epochs:
        TRAINING_CONFIG['epochs'] = args.epochs
    if args.batch_size:
        global BATCH_SIZE
        BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        DATALOADER_CONFIG['num_workers'] = args.num_workers
        if args.num_workers == 0:
            DATALOADER_CONFIG['persistent_workers'] = False
            DATALOADER_CONFIG.pop('prefetch_factor', None)

    run_training(args.model, resume=args.resume, no_compile=args.no_compile)


if __name__ == '__main__':
    main()
