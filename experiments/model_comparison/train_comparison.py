#!/usr/bin/env python
"""
Unified Training Script for Model Comparison Experiments

This script trains different models (UNet, ResNet, Swin Transformer, ViT) 
on the CellMap segmentation task and collects comprehensive metrics for comparison.

Features:
- Supports 2D and 3D models
- Extracts intermediate feature maps
- Saves visualizations (input, feature maps, predictions, ground truth)
- Tracks comprehensive metrics (BCE loss, Dice, IoU, accuracy)
- Uses fixed samples for fair visualization comparison
- Supports DDP for multi-GPU training

Usage:
    # Single model training
    python train_comparison.py --model unet --dim 2d --epochs 100
    
    # With DDP
    torchrun --nproc_per_node=2 train_comparison.py --model swin --dim 2d --epochs 100

Author: CellMap Segmentation Challenge
"""

import argparse
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path

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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cellmap_segmentation_challenge.models import (
    UNet_2D, UNet_3D, ResNet, SwinTransformer, ViTVNet, 
    SwinTransformer3D, ViTVNet2D
)
from cellmap_segmentation_challenge.utils import (
    CellMapLossWrapper, get_dataloader, make_datasplit_csv, format_string
)
from cellmap_segmentation_challenge.utils.ddp import (
    setup_ddp, cleanup_ddp, is_main_process, get_world_size, 
    reduce_value, sync_across_processes
)

# Local imports
from config_base import (
    CLASSES, VISUALIZATION_SEED, METRICS_PATH, CHECKPOINTS_PATH,
    TENSORBOARD_PATH, VISUALIZATIONS_PATH, MODEL_REGISTRY,
    get_model_config, get_input_shape, get_spatial_transforms,
    ensure_dirs_exist, GRADIENT_ACCUMULATION_STEPS, MAX_GRAD_NORM,
    LEARNING_RATE, WEIGHT_DECAY, BETAS, ITERATIONS_PER_EPOCH_2D,
    ITERATIONS_PER_EPOCH_3D, INPUT_SCALE_2D, INPUT_SCALE_3D,
    VALIDATION_PROB, EPOCHS_2D, EPOCHS_3D, MAX_CLASS_WEIGHT
)
from metrics_tracker import MetricsTracker, DiceLoss
from feature_extractor import FeatureExtractor, visualize_feature_maps


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train models for comparison experiment'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        choices=['unet', 'resnet', 'swin', 'vit'],
        help='Model architecture to train'
    )
    parser.add_argument(
        '--dim', 
        type=str, 
        required=True,
        choices=['2d', '3d'],
        help='Dimensionality of the model'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=None,
        help='Number of training epochs (default: from config)'
    )
    parser.add_argument(
        '--classes_file',
        type=str,
        default=None,
        help='Path to JSON file with CLASSES override, e.g. {"CLASSES": ["nuc","mito"]}'
    )
    parser.add_argument(
        '--iterations_per_epoch',
        type=int,
        default=None,
        help='Override iterations per epoch (default: from config)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=None,
        help='Batch size (default: from config)'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Resume from latest checkpoint'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Debug mode with fewer iterations'
    )
    parser.add_argument(
        '--save_features', 
        action='store_true',
        help='Save intermediate feature maps'
    )
    parser.add_argument(
        '--vis_every', 
        type=int, 
        default=10,
        help='Save visualizations every N epochs'
    )
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=10,
        help='Save periodic checkpoint every N epochs (default: 10)'
    )
    
    return parser.parse_args()


def create_model(model_name: str, dimension: str, device: torch.device) -> nn.Module:
    """Create model based on name and dimension."""
    config = get_model_config(model_name, dimension)
    model_class = config['class']
    model_kwargs = config['config']
    
    if model_class == 'UNet_2D':
        model = UNet_2D(**model_kwargs)
    elif model_class == 'UNet_3D':
        model = UNet_3D(**model_kwargs)
    elif model_class == 'ResNet':
        model = ResNet(**model_kwargs)
    elif model_class == 'SwinTransformer':
        model = SwinTransformer(**model_kwargs)
    elif model_class == 'ViTVNet':
        model = ViTVNet(**model_kwargs)
    elif model_class == 'SwinTransformer3D':
        model = SwinTransformer3D(**model_kwargs)
    elif model_class == 'ViTVNet2D':
        # ViTVNet2D takes config as a dict and separate in_channels, num_classes
        model = ViTVNet2D(
            config=model_kwargs,
            in_channels=1,
            num_classes=len(CLASSES)
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    return model.to(device)


def get_fixed_samples(
    dataloader, 
    num_samples: int = 3, 
    seed: int = VISUALIZATION_SEED,
    save_path: Path = None,
    force_regenerate: bool = False
) -> dict:
    """
    Get fixed samples for consistent visualization across ALL models.
    
    Samples are saved to disk so that different model runs use the exact
    same input images for fair visual comparison.
    
    Parameters
    ----------
    dataloader : DataLoader
        Validation dataloader to sample from
    num_samples : int
        Number of samples to collect
    seed : int
        Random seed for reproducibility
    save_path : Path
        Path to save/load fixed samples
    force_regenerate : bool
        If True, regenerate samples even if they exist on disk
    
    Returns
    -------
    dict
        Dictionary with 'inputs' and 'targets' tensors
    """
    if save_path is None:
        save_path = VISUALIZATIONS_PATH / "fixed_samples.pt"
    
    # Check if fixed samples already exist
    if save_path.exists() and not force_regenerate:
        print(f"Loading fixed visualization samples from {save_path}")
        samples = torch.load(save_path, weights_only=True)
        print(f"  Loaded {samples['inputs'].shape[0]} samples")
        return samples
    
    print(f"Generating new fixed visualization samples (seed={seed})...")
    
    # Set seed for reproducibility
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    
    # Collect samples
    samples = {'inputs': [], 'targets': []}
    
    dataloader.refresh()
    loader_iter = iter(dataloader.loader)
    
    collected = 0
    for batch in loader_iter:
        if collected >= num_samples:
            break
            
        # Get input and target keys
        input_keys = list(dataloader.dataset.input_arrays.keys())
        target_keys = list(dataloader.dataset.target_arrays.keys())
        
        if len(input_keys) > 1:
            inputs = {key: batch[key] for key in input_keys}
            inputs = list(inputs.values())[0]  # Take first for simplicity
        else:
            inputs = batch[input_keys[0]]
        
        if len(target_keys) > 1:
            targets = {key: batch[key] for key in target_keys}
            targets = list(targets.values())[0]
        else:
            targets = batch[target_keys[0]]
        
        # Take first sample from batch
        samples['inputs'].append(inputs[0:1])
        samples['targets'].append(targets[0:1])
        collected += 1
    
    # Stack samples
    samples['inputs'] = torch.cat(samples['inputs'], dim=0)
    samples['targets'] = torch.cat(samples['targets'], dim=0)
    
    # Save to disk for other models to use
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, save_path)
    print(f"  Saved {num_samples} fixed samples to {save_path}")
    
    return samples


def save_epoch_visualization(
    model: nn.Module,
    fixed_samples: dict,
    epoch: int,
    model_name: str,
    dimension: str,
    device: torch.device,
    save_path: Path,
    feature_extractor: FeatureExtractor = None
) -> None:
    """
    Save visualization of model predictions and feature maps.
    
    Saves:
    - Raw input
    - Ground truth
    - Model prediction
    - Feature maps (if extractor provided)
    """
    model.eval()
    
    with torch.no_grad():
        inputs = fixed_samples['inputs'].to(device)
        targets = fixed_samples['targets'].to(device)
        
        # Get predictions
        outputs = model(inputs)
        predictions = torch.sigmoid(outputs)
        
        # Get feature maps if extractor available
        features = None
        if feature_extractor is not None:
            features = feature_extractor(inputs)
    
    # Create figure for each sample
    for sample_idx in range(inputs.shape[0]):
        fig = create_comparison_figure(
            inputs[sample_idx],
            targets[sample_idx],
            predictions[sample_idx],
            features,
            sample_idx,
            dimension
        )
        
        # Save figure
        fig_path = save_path / f"{model_name}_{dimension}_epoch{epoch:04d}_sample{sample_idx}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Save feature map visualization if available
    if features is not None:
        try:
            feat_fig = visualize_feature_maps(features, batch_idx=0, max_channels=8)
            feat_path = save_path / f"{model_name}_{dimension}_epoch{epoch:04d}_features.png"
            feat_fig.savefig(feat_path, dpi=150, bbox_inches='tight')
            plt.close(feat_fig)
        except Exception as e:
            print(f"Warning: Could not save feature maps: {e}")


def create_comparison_figure(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    prediction_tensor: torch.Tensor,
    features: dict,
    sample_idx: int,
    dimension: str,
    show_all_classes: bool = True
) -> plt.Figure:
    """Create a comparison figure showing input, GT, and prediction for ALL classes."""
    # Handle 3D by taking middle slice
    if dimension == '3d' and input_tensor.dim() == 4:
        mid_slice = input_tensor.shape[1] // 2
        input_2d = input_tensor[:, mid_slice, :, :]
        target_2d = target_tensor[:, mid_slice, :, :]
        pred_2d = prediction_tensor[:, mid_slice, :, :]
    else:
        input_2d = input_tensor
        target_2d = target_tensor
        pred_2d = prediction_tensor
    
    # Show ALL classes (14 classes + 1 input row)
    n_classes = target_2d.shape[0]
    n_classes_to_show = n_classes if show_all_classes else min(4, n_classes)
    
    # Create larger figure to accommodate all classes
    fig_height = 2.5 * (n_classes_to_show + 1)  # Slightly smaller per row for 14 classes
    fig, axes = plt.subplots(n_classes_to_show + 1, 3, figsize=(12, fig_height))
    
    # Row 0: Input image (show in all 3 columns with labels)
    if input_2d.dim() == 3:
        input_img = input_2d[0].cpu().numpy()
    else:
        input_img = input_2d.cpu().numpy()
    # Ensure float32 to reduce host memory usage and avoid float64 upcasts
    if input_img.dtype != np.float32:
        input_img = input_img.astype(np.float32, copy=False)
    
    for col in range(3):
        axes[0, col].imshow(input_img, cmap='gray')
        axes[0, col].set_title(['Input', 'Ground Truth', 'Prediction'][col], fontsize=12, fontweight='bold')
        axes[0, col].axis('off')
    axes[0, 0].set_ylabel('Raw', fontsize=10, fontweight='bold')
    
    # Rows 1-N: ALL classes
    class_names = CLASSES[:n_classes_to_show]
    for i, class_name in enumerate(class_names):
        row = i + 1
        
        # Input (same for all rows) - helps with visual alignment
        axes[row, 0].imshow(input_img, cmap='gray', alpha=0.3)
        axes[row, 0].set_ylabel(class_name, fontsize=9, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Ground truth
        gt = target_2d[i].cpu().numpy()
        gt = np.nan_to_num(gt, nan=0)
        if gt.dtype != np.float32:
            gt = gt.astype(np.float32, copy=False)
        
        # Check if this class has any annotations in GT
        has_annotation = gt.sum() > 0
        
        axes[row, 1].imshow(gt, cmap='hot', vmin=0, vmax=1)
        if not has_annotation:
            axes[row, 1].text(0.5, 0.5, 'No GT', transform=axes[row, 1].transAxes,
                             ha='center', va='center', fontsize=8, color='white',
                             bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        axes[row, 1].axis('off')
        
        # Prediction
        pred = pred_2d[i].cpu().numpy()
        if pred.dtype != np.float32:
            pred = pred.astype(np.float32, copy=False)
        axes[row, 2].imshow(pred, cmap='hot', vmin=0, vmax=1)
        
        # Show prediction confidence stats
        pred_max = pred.max()
        pred_mean = pred.mean()
        axes[row, 2].text(0.02, 0.98, f'max:{pred_max:.2f}', transform=axes[row, 2].transAxes,
                         ha='left', va='top', fontsize=7, color='white',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.3))
        axes[row, 2].axis('off')
    
    plt.tight_layout()
    return fig


def train_model(args) -> dict:
    """
    Main training function.
    
    Returns a dictionary with training results summary.
    """
    # ============================================================
    # Setup
    # ============================================================
    
    # DDP setup (skip in worker processes)
    _is_worker = multiprocessing.parent_process() is not None
    
    if not _is_worker:
        local_rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device("cpu")
    
    use_ddp = world_size > 1
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create directories
    ensure_dirs_exist()
    
    # Model configuration
    model_config = get_model_config(args.model, args.dim)
    model_full_name = f"{args.model}_{args.dim}"
    
    # Get training parameters
    batch_size = args.batch_size or model_config['batch_size']
    epochs = args.epochs or (EPOCHS_2D if args.dim == '2d' else EPOCHS_3D)
    iterations_per_epoch = ITERATIONS_PER_EPOCH_2D if args.dim == '2d' else ITERATIONS_PER_EPOCH_3D
    
    if args.debug:
        epochs = 3
        iterations_per_epoch = 10
    
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"Training {model_full_name.upper()}")
        print(f"{'='*60}")
        print(f"Model type: {model_config['type']}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Iterations/epoch: {iterations_per_epoch}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"{'='*60}\n")
    
    # ============================================================
    # Create Model
    # ============================================================
    
    model = create_model(args.model, args.dim, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if is_main_process():
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # DDP wrapper
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=model_config['type'] == 'transformer'
        )
    
    # Get the underlying model for parameter access
    _model = model.module if hasattr(model, 'module') else model
    
    # ============================================================
    # Optimizer and Scheduler
    # ============================================================
    
    optimizer = torch.optim.AdamW(
        _model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS
    )
    
    total_steps = epochs * iterations_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000,
    )
    
    # ============================================================
    # Data Loaders
    # ============================================================
    
    input_shape = get_input_shape(args.dim)
    input_scale = INPUT_SCALE_2D if args.dim == '2d' else INPUT_SCALE_3D
    spatial_transforms = get_spatial_transforms(args.dim)
    
    input_array_info = {"shape": input_shape, "scale": input_scale}
    target_array_info = {"shape": input_shape, "scale": input_scale}
    
    # Create datasplit if needed
    datasplit_path = "datasplit.csv"
    if not os.path.exists(datasplit_path):
        make_datasplit_csv(
            classes=CLASSES,
            scale=input_scale,
            csv_path=datasplit_path,
            validation_prob=VALIDATION_PROB,
        )
    
    # Create data loaders
    train_raw_transforms = T.Compose([
        T.ToDtype(torch.float, scale=True),
        NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
    ])
    target_transforms = T.Compose([T.ToDtype(torch.float), Binarize()])
    
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
    )
    
    # Get fixed samples for visualization
    # Use dimension-specific path so 2D and 3D models use their own fixed samples
    # but ALL 2D models share the same samples, and ALL 3D models share the same samples
    if is_main_process():
        fixed_samples_path = VISUALIZATIONS_PATH / f"fixed_samples_{args.dim}.pt"
        fixed_samples = get_fixed_samples(
            val_loader, 
            num_samples=5,  # More samples for better comparison
            save_path=fixed_samples_path
        )
        print(f"Fixed samples shape: {fixed_samples['inputs'].shape}")
        
        # Log class distribution in fixed samples for verification
        print("\nClass distribution in fixed visualization samples:")
        for i, class_name in enumerate(CLASSES):
            target_class = fixed_samples['targets'][:, i]
            # Handle NaN values
            valid_mask = ~torch.isnan(target_class)
            if valid_mask.any():
                class_pixels = (target_class[valid_mask] > 0.5).sum().item()
                total_pixels = valid_mask.sum().item()
                coverage = class_pixels / total_pixels * 100 if total_pixels > 0 else 0
                print(f"  {class_name:12s}: {coverage:6.2f}% coverage ({class_pixels:,} pixels)")
            else:
                print(f"  {class_name:12s}: No valid annotations")
    else:
        fixed_samples = None
    
    # ============================================================
    # Loss and Metrics
    # ============================================================
    
    # Get class weights from dataset - these help balance rare classes
    pos_weight = list(train_loader.dataset.class_weights.values())
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device).flatten()
    
    # Optionally cap extreme weights to prevent gradient instability
    # Very high weights (>100) can cause training issues
    original_weights = pos_weight.clone()
    if MAX_CLASS_WEIGHT is not None:
        pos_weight = torch.clamp(pos_weight, max=MAX_CLASS_WEIGHT)
        n_capped = (original_weights > MAX_CLASS_WEIGHT).sum().item()
    else:
        n_capped = 0
    
    if is_main_process():
        print("\n" + "="*60)
        print("CLASS WEIGHTS (for loss balancing)")
        print("="*60)
        print("Higher weight = rarer class (model penalized more for missing it)")
        if MAX_CLASS_WEIGHT is not None:
            print(f"Max weight cap: {MAX_CLASS_WEIGHT} (configurable in config_base.py)")
            if n_capped > 0:
                print(f"WARNING: {n_capped} class(es) had weights capped!")
        else:
            print("Weight capping: DISABLED (using original weights)")
        print("-"*60)
        for i, class_name in enumerate(CLASSES):
            weight = pos_weight[i].item()
            orig_weight = original_weights[i].item()
            if MAX_CLASS_WEIGHT is not None and orig_weight > MAX_CLASS_WEIGHT:
                print(f"  {class_name:12s}: {weight:.4f} (capped from {orig_weight:.1f})")
            else:
                print(f"  {class_name:12s}: {weight:.4f}")
        print("="*60 + "\n")
    
    # Adjust for spatial dimensions
    spatial_dims = sum([s > 1 for s in input_shape])
    pos_weight = pos_weight[:, None, None]
    if spatial_dims == 3:
        pos_weight = pos_weight[..., None]
    
    criterion = CellMapLossWrapper(
        torch.nn.BCEWithLogitsLoss,
        pos_weight=pos_weight
    )
    
    # Metrics tracker
    metrics_tracker = MetricsTracker(
        classes=CLASSES,
        save_path=str(METRICS_PATH),
        model_name=model_full_name
    )
    
    # Feature extractor (for visualization)
    feature_extractor = None
    if args.save_features and is_main_process():
        feature_extractor = FeatureExtractor(model)
    
    # TensorBoard writer
    writer = None
    if is_main_process():
        log_path = str(TENSORBOARD_PATH / model_full_name)
        writer = SummaryWriter(log_path)
    
    # ============================================================
    # Training Loop
    # ============================================================
    
    input_keys = list(train_loader.dataset.input_arrays.keys())
    target_keys = list(train_loader.dataset.target_arrays.keys())
    
    n_iter = 0
    best_val_dice = 0.0
    
    if is_main_process():
        print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        metrics_tracker.start_epoch()
        
        train_loader.refresh()
        loader = iter(train_loader.loader)
        
        epoch_bar = tqdm(
            range(iterations_per_epoch),
            desc=f"Epoch {epoch}/{epochs}",
            disable=not is_main_process()
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
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets) / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass
            loss.backward()

            # Gradient clipping (capture grad norm)
            grad_norm = None
            if MAX_GRAD_NORM is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            # Optimizer step (apply gradient accumulation)
            if (iter_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                # Step scheduler per optimizer step so OneCycleLR updates per effective batch
                try:
                    scheduler.step()
                except Exception:
                    pass
                optimizer.zero_grad()
                # Log gradient norm and LR after the step
                if writer is not None:
                    if grad_norm is not None:
                        writer.add_scalar('train/grad_norm', float(grad_norm), n_iter)
                    try:
                        current_lr = scheduler.get_last_lr()[0]
                    except Exception:
                        current_lr = args.lr
                    writer.add_scalar('train/lr', current_lr, n_iter)

            # Log metrics
            loss_value = loss.item() * GRADIENT_ACCUMULATION_STEPS
            metrics_tracker.log_iteration(n_iter, loss_value, outputs, targets)

            epoch_bar.set_postfix({'loss': f'{loss_value:.4f}'})

            if writer is not None:
                writer.add_scalar('train/loss', loss_value, n_iter)
        
        # NOTE: Scheduler is stepped per optimizer step (handled inside loop)
        
        # ============================================================
        # Validation
        # ============================================================
        
        if use_ddp:
            sync_across_processes()
        
        model.eval()
        val_loader.refresh()
        
        val_outputs_list = []
        val_targets_list = []
        val_loss_sum = 0
        val_count = 0
        
        with torch.no_grad():
            val_bar = tqdm(
                val_loader.loader,
                desc="Validation",
                total=min(10, len(val_loader.loader)),
                disable=not is_main_process()
            )
            
            for batch_idx, batch in enumerate(val_bar):
                if batch_idx >= 10:  # Limit validation batches
                    break
                
                if len(input_keys) > 1:
                    inputs = {key: batch[key].to(device) for key in input_keys}
                else:
                    inputs = batch[input_keys[0]].to(device)
                
                if len(target_keys) > 1:
                    targets = {key: batch[key].to(device) for key in target_keys}
                else:
                    targets = batch[target_keys[0]].to(device)
                
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                
                val_loss_sum += val_loss.item()
                val_count += 1
                
                # Collect for metrics
                val_outputs_list.append(outputs.cpu())
                val_targets_list.append(targets.cpu())
        
        # Aggregate validation metrics
        val_outputs = torch.cat(val_outputs_list, dim=0)
        val_targets = torch.cat(val_targets_list, dim=0)
        
        if use_ddp:
            val_loss_avg = reduce_value(val_loss_sum / val_count, op='mean')
        else:
            val_loss_avg = val_loss_sum / val_count
        
        # End epoch and get summary
        epoch_summary = metrics_tracker.end_epoch(
            epoch,
            val_predictions=val_outputs,
            val_targets=val_targets,
            learning_rate=scheduler.get_last_lr()[0]
        )
        
        if is_main_process():
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('val/loss', val_loss_avg, n_iter)
                writer.add_scalar('val/dice', epoch_summary.get('val_dice', 0), n_iter)
            
            # Print epoch summary
            val_dice = epoch_summary.get('val_dice', 0)
            print(f"\nEpoch {epoch}: train_loss={epoch_summary['train_loss']:.4f}, "
                  f"val_loss={val_loss_avg:.4f}, val_dice={val_dice:.4f}")
            
            # Save best model
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                save_path = CHECKPOINTS_PATH / f"{model_full_name}_best.pth"
                torch.save(_model.state_dict(), save_path)
                print(f"  -> New best model saved (dice={val_dice:.4f})")
            
            # Save periodic checkpoint (configurable via --checkpoint_every)
            checkpoint_every = getattr(args, 'checkpoint_every', 10)
            if checkpoint_every is None:
                checkpoint_every = 10
            if epoch % int(checkpoint_every) == 0:
                save_path = CHECKPOINTS_PATH / f"{model_full_name}_epoch{epoch:04d}.pth"
                torch.save(_model.state_dict(), save_path)
            
            # Save visualizations
            if epoch % args.vis_every == 0 or epoch == 1:
                save_epoch_visualization(
                    model, fixed_samples, epoch, model_full_name,
                    args.dim, device, VISUALIZATIONS_PATH, feature_extractor
                )
        
        # Clear cache
        torch.cuda.empty_cache()
    
    # ============================================================
    # Finalize
    # ============================================================
    
    # Save final checkpoint and metrics
    if is_main_process():
        save_path = CHECKPOINTS_PATH / f"{model_full_name}_final.pth"
        torch.save(_model.state_dict(), save_path)
        
        metrics_tracker.save()
        
        if writer is not None:
            writer.close()
        
        if feature_extractor is not None:
            feature_extractor.remove_hooks()
        
        summary = metrics_tracker.get_summary()
        print(f"\n{'='*60}")
        print(f"Training Complete: {model_full_name}")
        print(f"{'='*60}")
        print(f"Total epochs: {summary['total_epochs']}")
        print(f"Total time: {summary['total_training_time']:.1f}s")
        print(f"Best validation Dice: {summary.get('best_val_dice', 0):.4f}")
        print(f"Final validation Dice: {summary.get('final_val_dice', 0):.4f}")
        print(f"{'='*60}\n")
        
        return summary
    
    return {}


def main():
    """Main entry point."""
    args = parse_args()

    # If user provided a classes override file, load it and replace global CLASSES
    if args.classes_file is not None:
        try:
            import json
            p = Path(args.classes_file)
            if p.exists():
                with open(p, 'r') as f:
                    data = json.load(f)
                if 'CLASSES' in data and isinstance(data['CLASSES'], list):
                    # override the global CLASSES imported from config_base
                    globals()['CLASSES'] = data['CLASSES']
                    print(f"Overrode CLASSES with {len(data['CLASSES'])} entries from {p}")
                else:
                    print(f"Warning: {p} does not contain a 'CLASSES' list. Ignoring.")
            else:
                print(f"Warning: classes_file {p} not found. Ignoring.")
        except Exception as e:
            print(f"Warning: could not load classes_file: {e}")

    # Allow override of iterations_per_epoch from CLI
    if args.iterations_per_epoch is not None:
        if args.dim == '2d':
            globals()['ITERATIONS_PER_EPOCH_2D'] = args.iterations_per_epoch
        else:
            globals()['ITERATIONS_PER_EPOCH_3D'] = args.iterations_per_epoch
    
    # Validate model/dimension combination
    # All architectures now support both 2D and 3D
    valid_combos = {
        '2d': ['unet', 'resnet', 'swin', 'vit'],
        '3d': ['unet', 'resnet', 'swin', 'vit'],
    }
    
    if args.model not in valid_combos[args.dim]:
        print(f"Error: Model '{args.model}' not available for {args.dim}.")
        print(f"Valid {args.dim} models: {valid_combos[args.dim]}")
        sys.exit(1)
    
    try:
        result = train_model(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
