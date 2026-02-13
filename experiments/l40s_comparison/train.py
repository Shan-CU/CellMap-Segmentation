#!/usr/bin/env python
"""
L40S Model Comparison Training Script

Trains 8 segmentation models (UNet/ResNet/Swin/ViT × 2D/3D) on the CellMap
14-class FIB-SEM segmentation task using the unified Balanced Softmax Partial
Tversky loss.

Key differences from experiments/model_comparison/train_comparison.py:
  - Uses BalancedSoftmaxPartialTverskyLoss (winner of 3 experiments)
  - Partial annotation masking for 14-class training
  - Early stopping (patience=20)
  - Online class frequency logging to TensorBoard
  - L40S-optimized batch sizes and GPU allocation

Usage:
    # Single model
    torchrun --standalone --nproc_per_node=2 train.py --model unet --dim 2d

    # With SLURM (see slurm/ directory)
    sbatch slurm/train_unet_2d.sbatch
"""

import argparse
import ctypes
import gc
import multiprocessing
import os
import platform
import random
import sys
import time
from pathlib import Path

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from cellmap_data.transforms.augment import NaNtoNum, Binarize
from tensorboardX import SummaryWriter
from tqdm import tqdm
from upath import UPath
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CUDA Optimizations
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cellmap_segmentation_challenge.models import (
    UNet_2D,
    UNet_3D,
    ResNet,
    SwinTransformer,
    ViTVNet,
    SwinTransformer3D,
    ViTVNet2D,
)
from cellmap_segmentation_challenge.utils import (
    get_dataloader,
    make_datasplit_csv,
)
from cellmap_segmentation_challenge.utils.ddp import (
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    get_world_size,
    reduce_value,
    sync_across_processes,
)

# Local imports
from config import (
    CLASSES,
    METRICS_PATH,
    CHECKPOINTS_PATH,
    TENSORBOARD_PATH,
    VISUALIZATIONS_PATH,
    MODEL_REGISTRY,
    get_model_config,
    get_input_shape,
    get_spatial_transforms,
    get_gradient_accumulation,
    get_num_workers,
    ensure_dirs_exist,
    MAX_GRAD_NORM,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BETAS,
    ITERATIONS_PER_EPOCH_2D,
    ITERATIONS_PER_EPOCH_3D,
    INPUT_SCALE_2D,
    INPUT_SCALE_3D,
    VALIDATION_PROB,
    EPOCHS_2D,
    EPOCHS_3D,
    LEARNING_RATE_OVERRIDE,
    TVERSKY_ALPHA,
    TVERSKY_BETA,
    BALANCED_SOFTMAX_TAU,
    FREQ_UPDATE_INTERVAL,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    VISUALIZATION_SEED,
)
from losses import (
    BalancedSoftmaxPartialTverskyLoss,
    DiceLoss,
    get_loss_function,
)


# ============================================================
# TensorStore / glibc Memory Fragmentation Workaround
# ============================================================
# TensorStore's cross-thread alloc/free pattern (encode on worker threads,
# free on IO threads) causes severe glibc malloc arena fragmentation.  RSS
# grows linearly (~17 GB/min for Swin 3D on 32-core nodes) even though
# in-use heap stays flat — the memory is "free" inside glibc arenas but
# never returned to the OS.
#
# Root cause: google/tensorstore#223 — confirmed by maintainers (@jbms,
# @laramiel, @daniel-wer).  The growth correlates with CPU core count
# because glibc creates up to 8×N_CORES arenas by default.
#
# Fix (primary): Set MALLOC_ARENA_MAX=1 in the environment (done in all
# sbatch files).  This limits glibc to a single arena, eliminating
# fragmentation entirely.
#
# Fix (secondary): Call gc.collect() + malloc_trim(0) periodically to
# nudge glibc into releasing any remaining free pages.
#
# NOTE: The default cache_pool total_bytes_limit is 0 (= caching
# disabled).  Do NOT set a non-zero limit — that would *enable* an LRU
# cache and increase memory use, not decrease it.
#
# References:
#   - google/tensorstore#223            (root cause discussion)
#   - google/tensorstore#235            (duplicate, merged into #223)
#   - janelia-cellmap/cellmap-segmentation-challenge#183 (downstream report)


def force_memory_release():
    """Force release of C++ heap memory back to the OS (secondary measure)."""
    gc.collect()
    if platform.system() == "Linux":
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except OSError:
            pass


# ============================================================
# Model Creation
# ============================================================


def create_model(model_name: str, dimension: str, device: torch.device) -> nn.Module:
    """Create model from registry."""
    config = get_model_config(model_name, dimension)
    model_class_name = config["class"]
    model_kwargs = config["config"]

    # Map class name to actual class
    model_classes = {
        "UNet_2D": UNet_2D,
        "UNet_3D": UNet_3D,
        "ResNet": ResNet,
        "SwinTransformer": SwinTransformer,
        "SwinTransformer3D": SwinTransformer3D,
        "ViTVNet": ViTVNet,
        "ViTVNet2D": ViTVNet2D,
    }

    model_cls = model_classes[model_class_name]
    model = model_cls(**model_kwargs)
    model = model.to(device)

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Created {model_class_name}: {total_params:,} params ({trainable:,} trainable)")

    return model


# ============================================================
# Visualization
# ============================================================


def _extract_tensor(value):
    """Recursively extract a tensor from nested dicts/lists returned by cellmap_data."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        # Nested dict — grab the first value and recurse
        return _extract_tensor(next(iter(value.values())))
    if isinstance(value, (list, tuple)):
        tensors = [_extract_tensor(v) for v in value]
        return torch.stack(tensors)
    raise TypeError(f"Cannot extract tensor from {type(value)}")


def get_fixed_samples(val_loader, num_samples=5, save_path=None):
    """Get or load fixed samples for visualization."""
    if save_path is not None and Path(save_path).exists():
        data = torch.load(save_path, weights_only=True)
        print(f"Loaded {data['inputs'].shape[0]} fixed samples from {save_path}")
        return data

    # Use the same keys the training loop uses
    input_keys = list(val_loader.dataset.input_arrays.keys())
    target_keys = list(val_loader.dataset.target_arrays.keys())

    inputs_list = []
    targets_list = []

    for batch in val_loader.loader:
        # Extract inputs — same logic as training loop
        if len(input_keys) > 1:
            inp = torch.cat([_extract_tensor(batch[k]) for k in input_keys], dim=1)
        else:
            inp = _extract_tensor(batch[input_keys[0]])

        # Extract targets
        if len(target_keys) > 1:
            tgt = torch.cat([_extract_tensor(batch[k]) for k in target_keys], dim=1)
        else:
            tgt = _extract_tensor(batch[target_keys[0]])

        inputs_list.append(inp)
        targets_list.append(tgt)

        if sum(x.shape[0] for x in inputs_list) >= num_samples:
            break

    data = {
        "inputs": torch.cat(inputs_list)[:num_samples],
        "targets": torch.cat(targets_list)[:num_samples],
    }

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, save_path)

    return data


def save_epoch_visualization(
    model, fixed_samples, epoch, model_name, dimension, device, save_path, writer=None
):
    """Save visualization of model predictions on fixed samples."""
    if fixed_samples is None:
        return

    model.eval()
    with torch.no_grad():
        inputs = fixed_samples["inputs"].to(device)
        targets = fixed_samples["targets"]
        outputs = model(inputs).cpu()
        predictions = torch.sigmoid(outputs)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for idx in range(min(3, inputs.shape[0])):
        fig = create_comparison_figure(
            inputs[idx].cpu(), targets[idx], predictions[idx], dimension
        )
        fig_path = save_path / f"{model_name}_{dimension}_epoch{epoch:04d}_sample{idx}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        if writer is not None:
            writer.add_figure(f"samples/sample_{idx}", fig, epoch)
        plt.close(fig)


def create_comparison_figure(input_tensor, target_tensor, prediction_tensor, dimension):
    """Create input/GT/prediction comparison figure for ALL 14 classes."""
    if dimension == "3d" and input_tensor.dim() == 4:
        mid = input_tensor.shape[1] // 2
        input_2d = input_tensor[:, mid, :, :]
        target_2d = target_tensor[:, mid, :, :]
        pred_2d = prediction_tensor[:, mid, :, :]
    else:
        input_2d = input_tensor
        target_2d = target_tensor
        pred_2d = prediction_tensor

    n_classes = target_2d.shape[0]
    fig_height = 2.5 * (n_classes + 1)
    fig, axes = plt.subplots(n_classes + 1, 3, figsize=(12, fig_height))

    if input_2d.dim() == 3:
        input_img = input_2d[0].cpu().numpy().astype(np.float32)
    else:
        input_img = input_2d.cpu().numpy().astype(np.float32)

    for col in range(3):
        axes[0, col].imshow(input_img, cmap="gray")
        axes[0, col].set_title(["Input", "Ground Truth", "Prediction"][col], fontsize=12, fontweight="bold")
        axes[0, col].axis("off")
    axes[0, 0].set_ylabel("Raw", fontsize=10, fontweight="bold")

    for i, class_name in enumerate(CLASSES[:n_classes]):
        row = i + 1
        axes[row, 0].imshow(input_img, cmap="gray", alpha=0.3)
        axes[row, 0].set_ylabel(class_name, fontsize=9, fontweight="bold")
        axes[row, 0].axis("off")

        gt = np.nan_to_num(target_2d[i].cpu().numpy(), nan=0).astype(np.float32)
        axes[row, 1].imshow(gt, cmap="hot", vmin=0, vmax=1)
        if gt.sum() == 0:
            axes[row, 1].text(
                0.5, 0.5, "No GT", transform=axes[row, 1].transAxes,
                ha="center", va="center", fontsize=8, color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
            )
        axes[row, 1].axis("off")

        pred = pred_2d[i].cpu().numpy().astype(np.float32)
        axes[row, 2].imshow(pred, cmap="hot", vmin=0, vmax=1)
        axes[row, 2].axis("off")

    plt.tight_layout()
    return fig


# ============================================================
# Metrics (lightweight — no separate metrics_tracker import needed)
# ============================================================


def compute_per_class_dice(predictions, targets):
    """Compute per-class Dice scores (NaN-safe)."""
    with torch.no_grad():
        probs = torch.sigmoid(predictions)
        preds_binary = (probs > 0.5).float()

        mask = ~torch.isnan(targets)
        targets_clean = targets.nan_to_num(0)

        dice_scores = {}
        for c, name in enumerate(CLASSES):
            if c >= predictions.shape[1]:
                break
            pred_c = preds_binary[:, c].flatten()
            tgt_c = targets_clean[:, c].flatten()
            m_c = mask[:, c].flatten()

            pred_c = pred_c[m_c]
            tgt_c = tgt_c[m_c]

            if len(pred_c) == 0:
                dice_scores[name] = float("nan")
                continue

            tp = (pred_c * tgt_c).sum()
            union = pred_c.sum() + tgt_c.sum()

            if union == 0:
                dice_scores[name] = 1.0 if tp == 0 else 0.0
            else:
                dice_scores[name] = (2 * tp / union).item()

        return dice_scores


# ============================================================
# Early Stopping
# ============================================================


class EarlyStopping:
    """Early stopping tracker."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ============================================================
# Argument Parsing
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="L40S Model Comparison Training")
    parser.add_argument("--model", type=str, required=True, choices=["unet", "resnet", "swin", "vit"])
    parser.add_argument("--dim", type=str, required=True, choices=["2d", "3d"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--iterations_per_epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vis_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--no_early_stopping", action="store_true")
    # Loss overrides
    parser.add_argument("--tau", type=float, default=BALANCED_SOFTMAX_TAU)
    parser.add_argument("--tversky_alpha", type=float, default=TVERSKY_ALPHA)
    parser.add_argument("--tversky_beta", type=float, default=TVERSKY_BETA)
    return parser.parse_args()


# ============================================================
# Main Training Function
# ============================================================


def train_model(args):
    """Main training loop."""

    # DDP setup
    _is_worker = multiprocessing.parent_process() is not None
    if not _is_worker:
        local_rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            try:
                torch.cuda.memory._set_allocator_settings("expandable_segments:False")
            except Exception:
                pass
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device("cpu")

    use_ddp = world_size > 1

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_dirs_exist()

    # Model config
    model_config = get_model_config(args.model, args.dim)
    model_full_name = f"{args.model}_{args.dim}"

    # Training params (with config defaults)
    batch_size = args.batch_size or model_config["batch_size"]
    epochs = args.epochs or (EPOCHS_2D if args.dim == "2d" else EPOCHS_3D)
    iterations_per_epoch = args.iterations_per_epoch or (
        ITERATIONS_PER_EPOCH_2D if args.dim == "2d" else ITERATIONS_PER_EPOCH_3D
    )
    grad_accum = args.gradient_accumulation or get_gradient_accumulation(args.model, args.dim)
    num_workers = args.num_workers if args.num_workers is not None else get_num_workers(args.model, args.dim)

    # Learning rate
    lr = args.lr
    if lr is None:
        lr = LEARNING_RATE_OVERRIDE.get(model_full_name, LEARNING_RATE)

    if args.debug:
        epochs = 3
        iterations_per_epoch = 10

    if is_main_process():
        n_gpus = world_size
        eff_batch = batch_size * grad_accum * n_gpus
        print(f"\n{'='*60}")
        print(f"L40S MODEL COMPARISON — {model_full_name.upper()}")
        print(f"{'='*60}")
        print(f"Model type     : {model_config['type']}")
        print(f"Batch size     : {batch_size} (per GPU)")
        print(f"Grad accum     : {grad_accum}")
        print(f"Effective batch: {eff_batch}")
        print(f"Epochs         : {epochs}")
        print(f"Iters/epoch    : {iterations_per_epoch}")
        print(f"Learning rate  : {lr}")
        print(f"Num workers    : {num_workers}")
        print(f"GPUs           : {n_gpus}")
        print(f"Device         : {device}")
        use_amp = args.amp and not args.no_amp
        use_compile = args.compile and not args.no_compile
        print(f"AMP (BFloat16) : {'Enabled' if use_amp else 'Disabled'}")
        print(f"torch.compile  : {'Enabled' if use_compile else 'Disabled'}")
        print(f"Early stopping : {'Disabled' if args.no_early_stopping else f'patience={EARLY_STOPPING_PATIENCE}'}")
        print(f"Loss           : BalancedSoftmaxTversky(τ={args.tau}, α={args.tversky_alpha}, β={args.tversky_beta})")
        print(f"{'='*60}\n")

    # ============================================================
    # Create Model
    # ============================================================
    model = create_model(args.model, args.dim, device)

    # torch.compile
    use_compile = args.compile and not args.no_compile
    if use_compile and hasattr(torch, "compile"):
        if is_main_process():
            print("Compiling model with torch.compile(mode='default')...")
        try:
            model = torch.compile(model, mode="default")
            if is_main_process():
                print("  Model compiled successfully")
        except Exception as e:
            if is_main_process():
                print(f"  Warning: torch.compile() failed: {e}")

    # DDP
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=model_config["type"] == "transformer",
        )

    _model = model.module if hasattr(model, "module") else model

    # ============================================================
    # Optimizer & Scheduler
    # ============================================================
    optimizer = torch.optim.AdamW(
        _model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, betas=BETAS
    )

    total_steps = (epochs * iterations_per_epoch) // grad_accum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1000,
    )

    # ============================================================
    # Data Loaders
    # ============================================================
    input_shape = get_input_shape(args.dim)
    input_scale = INPUT_SCALE_2D if args.dim == "2d" else INPUT_SCALE_3D
    spatial_transforms = get_spatial_transforms(args.dim)

    input_array_info = {"shape": input_shape, "scale": input_scale}
    target_array_info = {"shape": input_shape, "scale": input_scale}

    datasplit_path = "datasplit.csv"
    if not os.path.exists(datasplit_path):
        make_datasplit_csv(
            classes=CLASSES,
            scale=input_scale,
            csv_path=datasplit_path,
            validation_prob=VALIDATION_PROB,
        )

    train_raw_transforms = T.Compose([
        T.ToDtype(torch.float, scale=True),
        NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
    ])
    target_transforms = T.Compose([T.ToDtype(torch.float), Binarize()])

    dataloader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": args.pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
        dataloader_kwargs["persistent_workers"] = args.persistent_workers

    train_loader, val_loader = get_dataloader(
        datasplit_path=datasplit_path,
        classes=CLASSES,
        batch_size=batch_size,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=spatial_transforms,
        iterations_per_epoch=iterations_per_epoch,
        random_validation=True,
        device=device,
        weighted_sampler=True,
        train_raw_value_transforms=train_raw_transforms,
        val_raw_value_transforms=train_raw_transforms,
        target_value_transforms=target_transforms,
        **dataloader_kwargs,
    )

    # Fixed samples for visualization
    if is_main_process():
        fixed_samples_path = VISUALIZATIONS_PATH / f"fixed_samples_{args.dim}.pt"
        fixed_samples = get_fixed_samples(val_loader, num_samples=5, save_path=fixed_samples_path)
        print(f"Fixed samples shape: {fixed_samples['inputs'].shape}")
    else:
        fixed_samples = None

    # ============================================================
    # Loss Function (Balanced Softmax Partial Tversky)
    # ============================================================
    criterion = BalancedSoftmaxPartialTverskyLoss(
        classes=CLASSES,
        alpha=args.tversky_alpha,
        beta=args.tversky_beta,
        tau=args.tau,
        freq_update_interval=FREQ_UPDATE_INTERVAL,
    ).to(device)

    # Dice loss for validation metric
    dice_criterion = DiceLoss()

    if is_main_process():
        print(f"\nLoss: BalancedSoftmaxPartialTversky(τ={args.tau}, α={args.tversky_alpha}, β={args.tversky_beta})")
        print(f"  Partial annotation masking: ENABLED (NaN detection)")
        print(f"  Online class frequency update: every {FREQ_UPDATE_INTERVAL} batches\n")

    # TensorBoard
    writer = None
    if is_main_process():
        log_path = str(TENSORBOARD_PATH / model_full_name)
        writer = SummaryWriter(log_path)

    # Early stopping
    early_stopper = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
    ) if not args.no_early_stopping else None

    # AMP
    use_amp = args.amp and not args.no_amp and torch.cuda.is_available()
    if use_amp:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            scaler = None
            if is_main_process():
                print("AMP: BFloat16 (no scaler needed)")
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")
            if is_main_process():
                print("AMP: Float16 + GradScaler")
    else:
        amp_dtype = torch.float32
        scaler = None

    # ============================================================
    # Resume from checkpoint
    # ============================================================
    start_epoch = 1
    best_val_dice = 0.0
    n_iter = 0

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            if is_main_process():
                print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                _model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val_dice = ckpt.get("best_val_dice", 0.0)
                n_iter = ckpt.get("n_iter", 0)
            else:
                # Plain state dict
                _model.load_state_dict(ckpt)
            if is_main_process():
                print(f"  Resumed at epoch {start_epoch}, best_val_dice={best_val_dice:.4f}")

    # ============================================================
    # Training Loop
    # ============================================================
    input_keys = list(train_loader.dataset.input_arrays.keys())
    target_keys = list(train_loader.dataset.target_arrays.keys())

    if is_main_process():
        print(f"\nStarting training for {epochs} epochs (from epoch {start_epoch})...")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loader.refresh()
        loader = iter(train_loader.loader)

        epoch_losses = []
        epoch_bar = tqdm(
            range(iterations_per_epoch),
            desc=f"Epoch {epoch}/{epochs}",
            disable=not is_main_process(),
        )

        optimizer.zero_grad()

        for iter_idx in epoch_bar:
            batch = next(loader)
            n_iter += 1

            # Get inputs and targets
            if len(input_keys) > 1:
                inputs = {key: batch[key].to(device) for key in input_keys}
            else:
                inputs = batch[input_keys[0]].to(device)

            if len(target_keys) > 1:
                targets = {key: batch[key].to(device) for key in target_keys}
            else:
                targets = batch[target_keys[0]].to(device)

            # Forward + loss with AMP
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets) / grad_accum

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (iter_idx + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                if MAX_GRAD_NORM is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), MAX_GRAD_NORM
                    )
                else:
                    grad_norm = None

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                try:
                    scheduler.step()
                except Exception:
                    pass

                optimizer.zero_grad()

                # Log to TensorBoard
                if writer is not None:
                    if grad_norm is not None:
                        writer.add_scalar("train/grad_norm", float(grad_norm), n_iter)
                    try:
                        current_lr = scheduler.get_last_lr()[0]
                    except Exception:
                        current_lr = lr
                    writer.add_scalar("train/lr", current_lr, n_iter)

            loss_value = loss.item() * grad_accum
            epoch_losses.append(loss_value)
            epoch_bar.set_postfix({"loss": f"{loss_value:.4f}"})

            if writer is not None:
                writer.add_scalar("train/loss", loss_value, n_iter)

            # Periodic memory trim for 3D models (most severe leak)
            if args.dim == "3d" and (iter_idx + 1) % 50 == 0:
                force_memory_release()

        # ============================================================
        # Validation
        # ============================================================
        if use_ddp:
            sync_across_processes()

        model.eval()
        val_loader.refresh()

        val_losses = []
        all_val_dice = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader.loader):
                if batch_idx >= 10:
                    break

                if len(input_keys) > 1:
                    val_inputs = {key: batch[key].to(device) for key in input_keys}
                else:
                    val_inputs = batch[input_keys[0]].to(device)

                if len(target_keys) > 1:
                    val_targets = {key: batch[key].to(device) for key in target_keys}
                else:
                    val_targets = batch[target_keys[0]].to(device)

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                val_losses.append(val_loss.item())

                # Per-class Dice
                dice_scores = compute_per_class_dice(val_outputs.cpu(), val_targets.cpu())
                all_val_dice.append(dice_scores)

        # Aggregate metrics
        avg_train_loss = np.mean(epoch_losses)
        avg_val_loss = np.mean(val_losses) if val_losses else 0

        # Mean Dice across classes and batches
        mean_dice_per_class = {}
        for name in CLASSES:
            scores = [d[name] for d in all_val_dice if name in d and not np.isnan(d[name])]
            mean_dice_per_class[name] = np.mean(scores) if scores else 0.0
        val_dice_mean = np.mean(list(mean_dice_per_class.values()))

        if use_ddp:
            val_dice_mean_t = reduce_value(val_dice_mean, op="mean")
            if isinstance(val_dice_mean_t, torch.Tensor):
                val_dice_mean = val_dice_mean_t.item()
            else:
                val_dice_mean = float(val_dice_mean_t)

        if is_main_process():
            # TensorBoard logging
            if writer is not None:
                writer.add_scalar("val/loss", avg_val_loss, n_iter)
                writer.add_scalar("val/dice_mean", val_dice_mean, n_iter)
                writer.add_scalar("train/epoch_loss", avg_train_loss, n_iter)

                # Per-class Dice
                for name, score in mean_dice_per_class.items():
                    writer.add_scalar(f"val_dice/{name}", score, n_iter)

                # Log class frequency adjustments from loss
                logit_adj = criterion.get_logit_adjustments()
                for name, adj in logit_adj.items():
                    writer.add_scalar(f"loss/logit_adj/{name}", adj, n_iter)

                class_freqs = criterion.get_class_frequencies()
                for name, freq in class_freqs.items():
                    writer.add_scalar(f"loss/class_freq/{name}", freq, n_iter)

            # Print summary
            print(
                f"\nEpoch {epoch}: train_loss={avg_train_loss:.4f}, "
                f"val_loss={avg_val_loss:.4f}, val_dice={val_dice_mean:.4f}"
            )

            # Per-class Dice table (every 10 epochs or first/last)
            if epoch % 10 == 0 or epoch == start_epoch or epoch == epochs:
                print("  Per-class Dice:")
                for name, score in sorted(mean_dice_per_class.items(), key=lambda x: x[1]):
                    print(f"    {name:12s}: {score:.4f}")

            # Save best model
            if val_dice_mean > best_val_dice:
                best_val_dice = val_dice_mean
                save_path = CHECKPOINTS_PATH / f"{model_full_name}_best.pth"
                torch.save(
                    {
                        "model_state_dict": _model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_val_dice": best_val_dice,
                        "n_iter": n_iter,
                        "args": vars(args),
                    },
                    save_path,
                )
                print(f"  -> New best model saved (dice={val_dice_mean:.4f})")

            # Periodic checkpoint
            if epoch % args.checkpoint_every == 0:
                save_path = CHECKPOINTS_PATH / f"{model_full_name}_epoch{epoch:04d}.pth"
                torch.save(
                    {
                        "model_state_dict": _model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_val_dice": best_val_dice,
                        "n_iter": n_iter,
                    },
                    save_path,
                )

            # Visualizations
            if epoch % args.vis_every == 0 or epoch == start_epoch:
                save_epoch_visualization(
                    model, fixed_samples, epoch, model_full_name,
                    args.dim, device, VISUALIZATIONS_PATH, writer=writer,
                )

        # Early stopping
        if early_stopper is not None:
            if early_stopper.step(val_dice_mean):
                if is_main_process():
                    print(
                        f"\nEarly stopping triggered at epoch {epoch} "
                        f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)"
                    )
                break

        torch.cuda.empty_cache()
        force_memory_release()

    # ============================================================
    # Finalize
    # ============================================================
    if is_main_process():
        # Save final checkpoint
        save_path = CHECKPOINTS_PATH / f"{model_full_name}_final.pth"
        torch.save(
            {
                "model_state_dict": _model.state_dict(),
                "epoch": epoch,
                "best_val_dice": best_val_dice,
                "n_iter": n_iter,
                "args": vars(args),
            },
            save_path,
        )

        if writer is not None:
            writer.close()

        print(f"\n{'='*60}")
        print(f"Training Complete: {model_full_name}")
        print(f"{'='*60}")
        print(f"Total epochs: {epoch - start_epoch + 1}")
        print(f"Best validation Dice: {best_val_dice:.4f}")
        print(f"{'='*60}\n")

    if use_ddp:
        cleanup_ddp()


def main():
    args = parse_args()

    # Validate
    valid_combos = {
        "2d": ["unet", "resnet", "swin", "vit"],
        "3d": ["unet", "resnet", "swin", "vit"],
    }
    if args.model not in valid_combos[args.dim]:
        print(f"Error: {args.model} not available for {args.dim}")
        sys.exit(1)

    try:
        train_model(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
