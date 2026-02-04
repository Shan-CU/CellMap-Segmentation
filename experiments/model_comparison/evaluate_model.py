#!/usr/bin/env python
"""
Model Evaluation Script for CellMap Segmentation Challenge

This script evaluates a trained model checkpoint on the test crops from the 
CellMap challenge. It computes:
- Overall Dice score and IoU score
- Per-class Dice and IoU scores
- Generates visualizations (raw, ground truth, predictions) for 50 consistent samples

Usage:
    python evaluate_model.py --checkpoint checkpoints/unet_2d_best.pth --model unet --dim 2d
    python evaluate_model.py --checkpoint checkpoints/swin_2d_best.pth --model swin --dim 2d

Author: CellMap Segmentation Challenge
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from cellmap_data import CellMapImage, CellMapDataLoader, CellMapDataSplit
from cellmap_data.transforms.augment import NaNtoNum, Binarize
from tqdm import tqdm
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
    get_test_crops, fetch_test_crop_manifest, TEST_CROPS_DICT,
    get_dataloader
)
from cellmap_segmentation_challenge.config import (
    SEARCH_PATH, RAW_NAME, CROP_NAME
)

# Local imports
from config_base import (
    CLASSES, VISUALIZATION_SEED, MODEL_REGISTRY,
    get_model_config, get_input_shape, INPUT_SCALE_2D, INPUT_SCALE_3D,
    SPATIAL_TRANSFORMS_2D, SPATIAL_TRANSFORMS_3D, DATASPLIT_PATH
)


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

NUM_VIS_SAMPLES = 50  # Number of samples to visualize
MIN_CLASSES_FOR_VIS = 8  # Minimum number of classes present for a sample to be visualized

# Output directories
EVAL_OUTPUT_PATH = Path(__file__).parent / "evaluation_results"


# ============================================================
# METRICS COMPUTATION
# ============================================================

def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute Dice score for binary segmentation.
    
    Args:
        pred: Predicted binary mask (B, C, H, W) or (B, C, D, H, W)
        target: Ground truth binary mask (B, C, H, W) or (B, C, D, H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice score per class (C,)
    """
    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=(0, 2))
    union = pred_flat.sum(dim=(0, 2)) + target_flat.sum(dim=(0, 2))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute IoU (Intersection over Union) score for binary segmentation.
    
    Args:
        pred: Predicted binary mask (B, C, H, W) or (B, C, D, H, W)
        target: Ground truth binary mask (B, C, H, W) or (B, C, D, H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        IoU score per class (C,)
    """
    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=(0, 2))
    union = pred_flat.sum(dim=(0, 2)) + target_flat.sum(dim=(0, 2)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


def precision_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Compute precision per class."""
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)
    
    true_positive = (pred_flat * target_flat).sum(dim=(0, 2))
    predicted_positive = pred_flat.sum(dim=(0, 2))
    
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision


def recall_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Compute recall per class."""
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)
    
    true_positive = (pred_flat * target_flat).sum(dim=(0, 2))
    actual_positive = target_flat.sum(dim=(0, 2))
    
    recall = (true_positive + smooth) / (actual_positive + smooth)
    return recall


# ============================================================
# MODEL LOADING
# ============================================================

def create_model(model_name: str, dimension: str, device: torch.device, classes: list) -> nn.Module:
    """Create model based on name and dimension."""
    config = get_model_config(model_name, dimension)
    model_class = config['class']
    model_kwargs = config['config'].copy()
    
    # Update number of classes
    if 'n_classes' in model_kwargs:
        model_kwargs['n_classes'] = len(classes)
    if 'num_classes' in model_kwargs:
        model_kwargs['num_classes'] = len(classes)
    if 'output_nc' in model_kwargs:
        model_kwargs['output_nc'] = len(classes)
    
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
        model = ViTVNet2D(
            config=model_kwargs,
            in_channels=1,
            num_classes=len(classes)
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    return model.to(device)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    """
    Load model checkpoint and return metadata.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to
    
    Returns:
        Dictionary with checkpoint metadata (epoch, classes, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle DDP-wrapped models and torch.compile() models
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Remove prefixes if present:
    # - 'module.' from DDP
    # - '_orig_mod.' from torch.compile()
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        # Remove module. prefix (from DDP)
        if new_key.startswith('module.'):
            new_key = new_key[7:]
        # Remove _orig_mod. prefix (from torch.compile)
        if new_key.startswith('_orig_mod.'):
            new_key = new_key[10:]
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)
    
    metadata = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
        'classes': checkpoint.get('classes', CLASSES),
    }
    
    return metadata


# ============================================================
# DATA LOADING
# ============================================================

def get_validation_dataloader(
    classes: list,
    input_shape: tuple,
    input_scale: tuple,
    batch_size: int = 8,
    num_workers: int = 4,
    datasplit_path: str = None
) -> CellMapDataLoader:
    """
    Get the validation dataloader using the existing datasplit.
    
    Args:
        classes: List of class names
        input_shape: Input shape
        input_scale: Input scale (voxel size)
        batch_size: Batch size
        num_workers: Number of dataloader workers
        datasplit_path: Path to datasplit.csv
    
    Returns:
        Validation CellMapDataLoader
    """
    if datasplit_path is None:
        datasplit_path = str(Path(__file__).parent / DATASPLIT_PATH)
    
    input_array_info = {"shape": input_shape, "scale": input_scale}
    target_array_info = {"shape": input_shape, "scale": input_scale}
    
    # Use the existing get_dataloader function
    _, val_loader = get_dataloader(
        datasplit_path=datasplit_path,
        classes=classes,
        batch_size=batch_size,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=None,  # No augmentation for evaluation
        iterations_per_epoch=1000,
        random_validation=False,
        num_workers=num_workers,
    )
    
    return val_loader


def get_fixed_validation_samples(
    val_loader: CellMapDataLoader,
    num_samples: int = NUM_VIS_SAMPLES,
    min_classes: int = MIN_CLASSES_FOR_VIS,
    seed: int = VISUALIZATION_SEED,
    classes: list = None
) -> list:
    """
    Get fixed samples from validation loader for consistent visualization.
    
    Ensures diversity by:
    1. Collecting samples from ALL validation data (not just first few batches)
    2. Grouping by dataset/crop to ensure coverage across different sources
    3. Selecting samples with varied class compositions
    
    Args:
        val_loader: Validation dataloader
        num_samples: Number of samples to select
        min_classes: Minimum number of non-empty classes per sample
        seed: Random seed for reproducibility
        classes: List of class names
    
    Returns:
        List of sample dictionaries with raw, gt, and metadata
    """
    import re
    
    # Set seed for reproducibility
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Collect ALL candidate samples grouped by dataset/crop
    candidates_by_source = {}  # {dataset_crop: [candidates]}
    total_candidates = 0
    
    print(f"Scanning ALL validation data for diverse samples (target: {num_samples})...")
    
    for batch in tqdm(val_loader, desc="Scanning validation data"):
        raw = batch['input']
        target = batch['output']
        metadata_list = batch.get('__metadata__', [{}] * raw.size(0))
        
        # Handle NaN values in ground truth (unannotated regions are NaN)
        target = torch.nan_to_num(target, nan=0.0)
        
        for i in range(raw.size(0)):
            raw_i = raw[i].cpu()
            target_i = target[i].cpu()
            
            # Extract dataset and crop from metadata
            meta = metadata_list[i] if i < len(metadata_list) else {}
            target_path = meta.get('target_path_str', '')
            
            # Parse dataset name and crop from path like:
            # .../data/jrc_cos7-1a/jrc_cos7-1a.zarr/.../crop248/{label}
            dataset_match = re.search(r'/data/([^/]+)/[^/]+\.zarr/', target_path)
            crop_match = re.search(r'/(crop\d+)/', target_path)
            
            dataset_name = dataset_match.group(1) if dataset_match else 'unknown'
            crop_id = crop_match.group(1) if crop_match else f'sample_{total_candidates}'
            source_key = f"{dataset_name}_{crop_id}"
            
            # Count non-empty classes and which classes are present
            present_classes = set()
            if classes is not None:
                for j in range(target_i.size(0)):
                    if target_i[j].sum() > 0:
                        present_classes.add(classes[j])
            num_nonempty = len(present_classes)
            
            candidate = {
                'raw': raw_i,
                'gt': target_i,
                'num_nonempty_classes': num_nonempty,
                'present_classes': present_classes,
                'dataset': dataset_name,
                'crop_id': crop_id,
                'source_key': source_key,
                'sample_id': total_candidates
            }
            
            if source_key not in candidates_by_source:
                candidates_by_source[source_key] = []
            candidates_by_source[source_key].append(candidate)
            total_candidates += 1
    
    print(f"Found {total_candidates} total samples from {len(candidates_by_source)} unique dataset/crop combinations")
    
    # Strategy: Select diverse samples
    # 1. First, try to get at least one sample from each dataset
    # 2. Within each dataset, prefer samples with more classes
    # 3. Ensure variety in which classes are represented
    
    selected = []
    used_sources = set()
    all_selected_classes = set()
    
    # Get list of unique datasets
    datasets = list(set(c.split('_crop')[0] if '_crop' in c else c.rsplit('_', 1)[0] 
                       for c in candidates_by_source.keys()))
    rng.shuffle(datasets)
    
    print(f"Found {len(datasets)} unique datasets")
    
    # First pass: Get best sample from each dataset (prioritize class coverage)
    for dataset in datasets:
        if len(selected) >= num_samples:
            break
            
        # Find all sources for this dataset
        dataset_sources = [s for s in candidates_by_source.keys() if s.startswith(dataset)]
        
        # Collect all candidates from this dataset
        dataset_candidates = []
        for source in dataset_sources:
            dataset_candidates.extend(candidates_by_source[source])
        
        # Filter by minimum classes
        good_candidates = [c for c in dataset_candidates if c['num_nonempty_classes'] >= min_classes]
        if not good_candidates:
            good_candidates = dataset_candidates
        
        if not good_candidates:
            continue
        
        # Score candidates: prefer those with classes we haven't seen yet
        def diversity_score(c):
            new_classes = c['present_classes'] - all_selected_classes
            return (len(new_classes), c['num_nonempty_classes'])
        
        good_candidates.sort(key=diversity_score, reverse=True)
        
        # Select best candidate from this dataset
        best = good_candidates[0]
        selected.append(best)
        used_sources.add(best['source_key'])
        all_selected_classes.update(best['present_classes'])
    
    # Second pass: Fill remaining slots with diverse samples
    if len(selected) < num_samples:
        # Collect remaining candidates
        remaining = []
        for source, candidates in candidates_by_source.items():
            for c in candidates:
                if c['sample_id'] not in [s['sample_id'] for s in selected]:
                    if c['num_nonempty_classes'] >= min_classes:
                        remaining.append(c)
        
        # Sort by diversity (new classes) then by total classes
        def diversity_score(c):
            new_classes = c['present_classes'] - all_selected_classes
            # Also prefer samples from different sources
            source_penalty = 0 if c['source_key'] not in used_sources else -5
            return (source_penalty + len(new_classes), c['num_nonempty_classes'])
        
        remaining.sort(key=diversity_score, reverse=True)
        
        for c in remaining:
            if len(selected) >= num_samples:
                break
            selected.append(c)
            used_sources.add(c['source_key'])
            all_selected_classes.update(c['present_classes'])
    
    # Final shuffle for randomness in display order
    rng.shuffle(selected)
    
    # Print diversity statistics
    unique_datasets = set(s['dataset'] for s in selected)
    unique_crops = set(s['source_key'] for s in selected)
    
    print(f"\nSelected {len(selected)} diverse samples:")
    print(f"  - From {len(unique_datasets)} unique datasets: {sorted(unique_datasets)}")
    print(f"  - From {len(unique_crops)} unique crops")
    print(f"  - Avg non-empty classes per sample: {np.mean([s['num_nonempty_classes'] for s in selected]):.1f}")
    print(f"  - Total classes covered: {len(all_selected_classes)}/{len(classes) if classes else 'N/A'}")
    
    return selected


def get_sample_indices_for_visualization(
    test_crops: list,
    classes: list,
    num_samples: int = NUM_VIS_SAMPLES,
    min_classes: int = MIN_CLASSES_FOR_VIS,
    seed: int = VISUALIZATION_SEED
) -> list:
    """
    Get consistent sample indices for visualization across model runs.
    
    This function selects samples that have ground truth for most training classes
    to ensure meaningful visual comparison.
    
    Args:
        test_crops: List of test crop metadata
        classes: List of class names to evaluate
        num_samples: Number of samples to select
        min_classes: Minimum number of classes present in a sample
        seed: Random seed for reproducibility
    
    Returns:
        List of (crop_id, class_label, slice_idx) tuples
    """
    # Set seed for reproducibility
    rng = random.Random(seed)
    
    # Build index of which classes are available in which crops
    test_crop_manifest = fetch_test_crop_manifest()
    crop_class_map = {}
    
    for crop in test_crop_manifest:
        if crop.id not in crop_class_map:
            crop_class_map[crop.id] = {'classes': set(), 'shape': crop.shape}
        crop_class_map[crop.id]['classes'].add(crop.class_label)
    
    # Find crops that have at least min_classes of our training classes
    good_crops = []
    for crop_id, info in crop_class_map.items():
        overlap = info['classes'].intersection(set(classes))
        if len(overlap) >= min_classes:
            good_crops.append({
                'crop_id': crop_id,
                'classes_present': list(overlap),
                'num_classes': len(overlap),
                'shape': info['shape']
            })
    
    # Sort by number of classes present (descending) for better samples
    good_crops.sort(key=lambda x: x['num_classes'], reverse=True)
    
    # Generate sample indices
    samples = []
    samples_per_crop = max(1, num_samples // len(good_crops)) if good_crops else 0
    
    for crop_info in good_crops:
        crop_id = crop_info['crop_id']
        shape = crop_info['shape']
        z_dim = shape[0]
        
        # Select random slices from this crop
        if z_dim > 1:
            slice_indices = rng.sample(range(z_dim), min(samples_per_crop, z_dim))
        else:
            slice_indices = [0]
        
        for slice_idx in slice_indices:
            samples.append({
                'crop_id': crop_id,
                'slice_idx': slice_idx,
                'classes_present': crop_info['classes_present'],
                'shape': shape
            })
            
            if len(samples) >= num_samples:
                break
        
        if len(samples) >= num_samples:
            break
    
    # If we don't have enough samples, pad with more from the best crops
    while len(samples) < num_samples and good_crops:
        crop_info = rng.choice(good_crops)
        z_dim = crop_info['shape'][0]
        slice_idx = rng.randint(0, z_dim - 1) if z_dim > 1 else 0
        samples.append({
            'crop_id': crop_info['crop_id'],
            'slice_idx': slice_idx,
            'classes_present': crop_info['classes_present'],
            'shape': crop_info['shape']
        })
    
    return samples[:num_samples]


def load_test_data_for_crop(
    crop_id: int,
    classes: list,
    input_shape: tuple,
    input_scale: tuple,
    slice_idx: Optional[int] = None
) -> tuple:
    """
    Load raw and ground truth data for a specific crop.
    
    Args:
        crop_id: ID of the test crop
        classes: List of class names
        input_shape: Shape of input (D, H, W) or (1, H, W) for 2D
        input_scale: Scale of input data (voxel size)
        slice_idx: Specific slice to load (for 2D evaluation)
    
    Returns:
        Tuple of (raw_data, gt_data, metadata)
    """
    test_crops = get_test_crops()
    crop_info = None
    for crop in test_crops:
        if crop.id == crop_id:
            crop_info = crop
            break
    
    if crop_info is None:
        raise ValueError(f"Crop {crop_id} not found in test crops")
    
    # Get paths
    dataset = crop_info.dataset
    raw_path = SEARCH_PATH.format(dataset=dataset, name=RAW_NAME)
    
    # Load raw data
    raw_image = CellMapImage(
        raw_path,
        target_class="raw",
        target_scale=input_scale,
        target_voxel_shape=input_shape,
        pad=True,
        pad_value=0,
    )
    
    # Get the center of the crop for loading
    gt_source = crop_info.gt_source
    center = tuple(
        gt_source.translation[i] + (gt_source.voxel_size[i] * gt_source.shape[i]) / 2
        for i in range(3)
    )
    
    # Adjust center for specific slice if 2D
    if slice_idx is not None and input_shape[0] == 1:
        # Calculate z position for this slice
        z_start = gt_source.translation[0]
        z_size = gt_source.voxel_size[0] * gt_source.shape[0]
        z_center = z_start + (slice_idx / gt_source.shape[0]) * z_size
        center = (z_center, center[1], center[2])
    
    # Load ground truth for each class
    gt_data = {}
    for cls in classes:
        try:
            gt_path = SEARCH_PATH.format(
                dataset=dataset, 
                name=CROP_NAME.format(crop=f"crop{crop_id}", label=cls)
            )
            gt_image = CellMapImage(
                gt_path,
                target_class=cls,
                target_scale=input_scale,
                target_voxel_shape=input_shape,
                pad=True,
                pad_value=0,
            )
            gt_data[cls] = gt_image[center]
        except Exception as e:
            # Class not available for this crop
            gt_data[cls] = None
    
    raw_data = raw_image[center]
    
    metadata = {
        'crop_id': crop_id,
        'dataset': dataset,
        'center': center,
        'slice_idx': slice_idx,
    }
    
    return raw_data, gt_data, metadata


# ============================================================
# EVALUATION LOOP
# ============================================================

class TestDataset(torch.utils.data.Dataset):
    """Dataset for loading test crops."""
    
    def __init__(
        self,
        classes: list,
        input_shape: tuple,
        input_scale: tuple,
        samples: list,
        value_transforms=None
    ):
        self.classes = classes
        self.input_shape = input_shape
        self.input_scale = input_scale
        self.samples = samples
        self.value_transforms = value_transforms or T.Compose([
            T.ToDtype(torch.float, scale=True),
            NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
        ])
        self.target_transforms = T.Compose([T.ToDtype(torch.float), Binarize()])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            raw_data, gt_data, metadata = load_test_data_for_crop(
                crop_id=sample['crop_id'],
                classes=self.classes,
                input_shape=self.input_shape,
                input_scale=self.input_scale,
                slice_idx=sample.get('slice_idx')
            )
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return zeros for failed samples
            raw_tensor = torch.zeros(1, *self.input_shape[1:])
            gt_tensor = torch.zeros(len(self.classes), *self.input_shape[1:])
            return {
                'raw': raw_tensor,
                'gt': gt_tensor,
                'metadata': sample,
                'valid': False
            }
        
        # Process raw data
        if isinstance(raw_data, np.ndarray):
            raw_tensor = torch.from_numpy(raw_data)
        else:
            raw_tensor = raw_data
        
        raw_tensor = self.value_transforms(raw_tensor)
        
        # Remove singleton dimension for 2D
        if self.input_shape[0] == 1 and raw_tensor.dim() == 4:
            raw_tensor = raw_tensor.squeeze(1)  # Remove Z dimension
        elif self.input_shape[0] == 1 and raw_tensor.dim() == 3:
            raw_tensor = raw_tensor.squeeze(0)  # Remove Z dimension if (Z, H, W)
        
        # Ensure correct shape (C, H, W) for 2D
        if raw_tensor.dim() == 2:
            raw_tensor = raw_tensor.unsqueeze(0)  # Add channel dimension
        
        # Process ground truth
        gt_list = []
        for cls in self.classes:
            if gt_data.get(cls) is not None:
                if isinstance(gt_data[cls], np.ndarray):
                    gt_cls = torch.from_numpy(gt_data[cls])
                else:
                    gt_cls = gt_data[cls]
                
                gt_cls = self.target_transforms(gt_cls)
                
                # Remove singleton dimension for 2D
                if self.input_shape[0] == 1:
                    if gt_cls.dim() == 4:
                        gt_cls = gt_cls.squeeze(1)
                    elif gt_cls.dim() == 3:
                        gt_cls = gt_cls.squeeze(0)
                
                # Ensure 2D shape
                if gt_cls.dim() == 2:
                    gt_list.append(gt_cls)
                elif gt_cls.dim() == 3:
                    gt_list.append(gt_cls[0])  # Take first channel/slice
                else:
                    gt_list.append(gt_cls.squeeze())
            else:
                # No GT for this class - use zeros
                if self.input_shape[0] == 1:
                    gt_list.append(torch.zeros(*self.input_shape[1:]))
                else:
                    gt_list.append(torch.zeros(*self.input_shape))
        
        gt_tensor = torch.stack(gt_list, dim=0)
        
        return {
            'raw': raw_tensor,
            'gt': gt_tensor,
            'metadata': {**sample, **metadata} if 'center' in str(metadata) else sample,
            'valid': True
        }


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    classes: list,
    device: torch.device,
    is_2d: bool = True
) -> dict:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        dataloader: DataLoader with test data
        classes: List of class names
        device: Device to run evaluation on
        is_2d: Whether model is 2D
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Accumulators for metrics
    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []
    
    # Per-class accumulators
    class_dice = {cls: [] for cls in classes}
    class_iou = {cls: [] for cls in classes}
    
    num_valid = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            raw = batch['raw'].to(device)
            gt = batch['gt'].to(device)
            valid = batch['valid']
            
            # Skip invalid samples
            if not any(valid):
                continue
            
            # Forward pass
            outputs = model(raw)
            
            # Apply sigmoid for binary predictions
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > 0.5).float()
            
            # Compute metrics for valid samples only
            for i, v in enumerate(valid):
                if not v:
                    continue
                
                pred_i = preds_binary[i:i+1]
                gt_i = gt[i:i+1]
                
                # Compute per-sample metrics
                dice = dice_score(pred_i, gt_i)
                iou = iou_score(pred_i, gt_i)
                prec = precision_score(pred_i, gt_i)
                rec = recall_score(pred_i, gt_i)
                
                all_dice.append(dice.cpu())
                all_iou.append(iou.cpu())
                all_precision.append(prec.cpu())
                all_recall.append(rec.cpu())
                
                # Per-class metrics
                for j, cls in enumerate(classes):
                    class_dice[cls].append(dice[j].item())
                    class_iou[cls].append(iou[j].item())
                
                num_valid += 1
    
    # Aggregate metrics
    if all_dice:
        all_dice = torch.stack(all_dice)
        all_iou = torch.stack(all_iou)
        all_precision = torch.stack(all_precision)
        all_recall = torch.stack(all_recall)
        
        results = {
            'overall': {
                'dice_mean': all_dice.mean().item(),
                'dice_std': all_dice.std().item(),
                'iou_mean': all_iou.mean().item(),
                'iou_std': all_iou.std().item(),
                'precision_mean': all_precision.mean().item(),
                'recall_mean': all_recall.mean().item(),
            },
            'per_class': {
                cls: {
                    'dice_mean': np.mean(class_dice[cls]) if class_dice[cls] else 0,
                    'dice_std': np.std(class_dice[cls]) if class_dice[cls] else 0,
                    'iou_mean': np.mean(class_iou[cls]) if class_iou[cls] else 0,
                    'iou_std': np.std(class_iou[cls]) if class_iou[cls] else 0,
                    'num_samples': len(class_dice[cls]),
                }
                for cls in classes
            },
            'num_samples': num_valid,
        }
    else:
        results = {
            'overall': {
                'dice_mean': 0,
                'dice_std': 0,
                'iou_mean': 0,
                'iou_std': 0,
                'precision_mean': 0,
                'recall_mean': 0,
            },
            'per_class': {cls: {'dice_mean': 0, 'iou_mean': 0, 'num_samples': 0} for cls in classes},
            'num_samples': 0,
        }
    
    return results


def evaluate_model_on_dataloader(
    model: nn.Module,
    dataloader,
    classes: list,
    device: torch.device,
    is_2d: bool = True
) -> dict:
    """
    Evaluate model on data from CellMapDataLoader.
    
    Args:
        model: Trained model
        dataloader: CellMapDataLoader
        classes: List of class names
        device: Device to run evaluation on
        is_2d: Whether model is 2D
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Accumulators for metrics
    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []
    
    # Per-class accumulators
    class_dice = {cls: [] for cls in classes}
    class_iou = {cls: [] for cls in classes}
    
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # CellMapDataLoader returns 'input' and 'output'
            raw = batch['input'].to(device)
            gt = batch['output'].to(device)
            
            # Handle NaN values in ground truth (unannotated regions are NaN)
            # Replace NaN with 0 (treat as background/negative class)
            gt = torch.nan_to_num(gt, nan=0.0)
            
            # Handle 2D case: remove singleton Z dimension
            if is_2d and raw.dim() == 5:
                raw = raw.squeeze(2)  # (B, C, 1, H, W) -> (B, C, H, W)
            if is_2d and gt.dim() == 5:
                gt = gt.squeeze(2)  # (B, C, 1, H, W) -> (B, C, H, W)
            
            # Forward pass
            outputs = model(raw)
            
            # Apply sigmoid for binary predictions
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > 0.5).float()
            
            # Compute batch metrics
            batch_size = raw.size(0)
            for i in range(batch_size):
                pred_i = preds_binary[i:i+1]
                gt_i = gt[i:i+1]
                
                # Compute per-sample metrics
                dice = dice_score(pred_i, gt_i)
                iou = iou_score(pred_i, gt_i)
                prec = precision_score(pred_i, gt_i)
                rec = recall_score(pred_i, gt_i)
                
                all_dice.append(dice.cpu())
                all_iou.append(iou.cpu())
                all_precision.append(prec.cpu())
                all_recall.append(rec.cpu())
                
                # Per-class metrics
                for j, cls in enumerate(classes):
                    class_dice[cls].append(dice[j].item())
                    class_iou[cls].append(iou[j].item())
                
                num_samples += 1
    
    # Aggregate metrics
    if all_dice:
        all_dice = torch.stack(all_dice)
        all_iou = torch.stack(all_iou)
        all_precision = torch.stack(all_precision)
        all_recall = torch.stack(all_recall)
        
        results = {
            'overall': {
                'dice_mean': all_dice.mean().item(),
                'dice_std': all_dice.std().item(),
                'iou_mean': all_iou.mean().item(),
                'iou_std': all_iou.std().item(),
                'precision_mean': all_precision.mean().item(),
                'recall_mean': all_recall.mean().item(),
            },
            'per_class': {
                cls: {
                    'dice_mean': np.mean(class_dice[cls]) if class_dice[cls] else 0,
                    'dice_std': np.std(class_dice[cls]) if class_dice[cls] else 0,
                    'iou_mean': np.mean(class_iou[cls]) if class_iou[cls] else 0,
                    'iou_std': np.std(class_iou[cls]) if class_iou[cls] else 0,
                    'num_samples': len(class_dice[cls]),
                }
                for cls in classes
            },
            'num_samples': num_samples,
        }
    else:
        results = {
            'overall': {
                'dice_mean': 0, 'dice_std': 0,
                'iou_mean': 0, 'iou_std': 0,
                'precision_mean': 0, 'recall_mean': 0,
            },
            'per_class': {cls: {'dice_mean': 0, 'iou_mean': 0, 'num_samples': 0} for cls in classes},
            'num_samples': 0,
        }
    
    return results


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_samples(
    model: nn.Module,
    samples: list,
    classes: list,
    device: torch.device,
    output_dir: Path,
    model_name: str,
    is_2d: bool = True
):
    """
    Generate and save visualizations from pre-loaded samples.
    
    Args:
        model: Trained model
        samples: List of sample dictionaries with 'raw' and 'gt' tensors
        classes: List of class names
        device: Device
        output_dir: Output directory
        model_name: Name for file naming
        is_2d: Whether model is 2D
    """
    model.eval()
    
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata for reproducibility
    metadata_file = output_dir / "visualization_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'num_samples': len(samples),
            'classes': classes,
            'seed': VISUALIZATION_SEED,
        }, f, indent=2)
    
    # Color map for classes
    cmap = plt.cm.get_cmap('tab20', len(classes))
    
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(samples, desc="Generating visualizations")):
            raw = sample['raw']
            gt = sample['gt']
            
            # Add batch dimension if needed
            if raw.dim() == 3:
                raw = raw.unsqueeze(0)
            if gt.dim() == 3:
                gt = gt.unsqueeze(0)
            
            # Handle 2D case
            if is_2d and raw.dim() == 5:
                raw = raw.squeeze(2)
            if is_2d and gt.dim() == 4 and gt.size(1) == 1:
                gt = gt.squeeze(1)
            
            raw_device = raw.to(device)
            
            # Forward pass
            outputs = model(raw_device)
            preds = torch.sigmoid(outputs).cpu()[0]
            preds_binary = (preds > 0.5).float()
            
            # Get numpy arrays
            raw_np = raw[0, 0].cpu().numpy() if raw.dim() == 4 else raw[0].cpu().numpy()
            gt_np = gt[0].cpu().numpy() if gt.dim() == 4 else gt.cpu().numpy()
            
            # Create figure - 7 rows: 1 overview + 3 pairs of GT/Pred rows for 14 classes
            # Layout: Row 0 = overview, Rows 1-2 = classes 0-6, Rows 3-4 = classes 7-13
            n_classes = len(classes)
            n_cols = 7  # Show 7 classes per row pair
            fig, axes = plt.subplots(5, n_cols, figsize=(24, 16))
            
            # Row 0: Raw image, GT overlay, Pred overlay, and Legend (spanning multiple cells)
            axes[0, 0].imshow(raw_np, cmap='gray')
            axes[0, 0].set_title('Raw Input')
            axes[0, 0].axis('off')
            
            # GT overlay - blend raw image with colored GT masks
            gt_overlay = np.stack([raw_np] * 3, axis=-1)  # Convert grayscale to RGB
            gt_overlay = (gt_overlay - gt_overlay.min()) / (gt_overlay.max() - gt_overlay.min() + 1e-8)  # Normalize to 0-1
            gt_mask_any = np.zeros(raw_np.shape, dtype=bool)
            for j, cls in enumerate(classes):
                gt_cls = gt_np[j] if gt_np.ndim == 3 else gt_np
                if gt_cls.max() > 0:
                    mask = gt_cls > 0.5
                    gt_mask_any |= mask
                    color = cmap(j)[:3]
                    for c in range(3):
                        gt_overlay[:, :, c] = np.where(mask, color[c], gt_overlay[:, :, c])
            gt_overlay = np.clip(gt_overlay, 0, 1)
            
            axes[0, 1].imshow(gt_overlay)
            axes[0, 1].set_title('GT Overlay')
            axes[0, 1].axis('off')
            
            # Pred overlay - blend raw image with colored prediction masks
            pred_overlay = np.stack([raw_np] * 3, axis=-1)  # Convert grayscale to RGB
            pred_overlay = (pred_overlay - pred_overlay.min()) / (pred_overlay.max() - pred_overlay.min() + 1e-8)  # Normalize to 0-1
            for j, cls in enumerate(classes):
                pred_cls = preds_binary[j].numpy()
                if pred_cls.max() > 0:
                    mask = pred_cls > 0.5
                    color = cmap(j)[:3]
                    for c in range(3):
                        pred_overlay[:, :, c] = np.where(mask, color[c], pred_overlay[:, :, c])
            pred_overlay = np.clip(pred_overlay, 0, 1)
            
            axes[0, 2].imshow(pred_overlay)
            axes[0, 2].set_title('Pred Overlay')
            axes[0, 2].axis('off')
            
            # Legend - show all classes in remaining columns
            for col in range(3, n_cols):
                axes[0, col].axis('off')
            
            # Create legend in column 3-4 area
            legend_ax = axes[0, 3]
            legend_ax.axis('off')
            for j, cls in enumerate(classes[:7]):  # First 7 classes
                y_pos = 1 - (j + 1) * 0.13
                color = cmap(j)
                legend_ax.add_patch(plt.Rectangle((0, y_pos), 0.2, 0.1, color=color))
                legend_ax.text(0.25, y_pos + 0.05, cls, fontsize=8, va='center')
            legend_ax.set_xlim(0, 1)
            legend_ax.set_ylim(0, 1)
            legend_ax.set_title('Legend (1-7)')
            
            legend_ax2 = axes[0, 4]
            legend_ax2.axis('off')
            for j, cls in enumerate(classes[7:14]):  # Classes 7-13
                y_pos = 1 - (j + 1) * 0.13
                color = cmap(j + 7)
                legend_ax2.add_patch(plt.Rectangle((0, y_pos), 0.2, 0.1, color=color))
                legend_ax2.text(0.25, y_pos + 0.05, cls, fontsize=8, va='center')
            legend_ax2.set_xlim(0, 1)
            legend_ax2.set_ylim(0, 1)
            legend_ax2.set_title('Legend (8-14)')
            
            # Clear remaining top row cells
            for col in range(5, n_cols):
                axes[0, col].axis('off')
            
            # Rows 1-2: Classes 0-6 (GT in row 1, Pred in row 2)
            for j in range(min(7, n_classes)):
                cls = classes[j]
                gt_cls = gt_np[j] if gt_np.ndim == 3 else gt_np
                pred_cls = preds_binary[j].numpy()
                
                axes[1, j].imshow(gt_cls, cmap='gray')
                axes[1, j].set_title(f'GT: {cls}', fontsize=8)
                axes[1, j].axis('off')
                
                axes[2, j].imshow(pred_cls, cmap='gray')
                axes[2, j].set_title(f'Pred: {cls}', fontsize=8)
                axes[2, j].axis('off')
            
            # Rows 3-4: Classes 7-13 (GT in row 3, Pred in row 4)
            for j in range(7, min(14, n_classes)):
                col = j - 7
                cls = classes[j]
                gt_cls = gt_np[j] if gt_np.ndim == 3 else gt_np
                pred_cls = preds_binary[j].numpy()
                
                axes[3, col].imshow(gt_cls, cmap='gray')
                axes[3, col].set_title(f'GT: {cls}', fontsize=8)
                axes[3, col].axis('off')
                
                axes[4, col].imshow(pred_cls, cmap='gray')
                axes[4, col].set_title(f'Pred: {cls}', fontsize=8)
                axes[4, col].axis('off')
            
            # Clear unused cells
            for col in range(n_classes - 7 if n_classes > 7 else 0, n_cols):
                if n_classes <= 7 or col >= n_classes - 7:
                    axes[3, col].axis('off')
                    axes[4, col].axis('off')
            
            fig.suptitle(f'Sample {idx} (Classes with GT: {sample.get("num_nonempty_classes", "?")})', fontsize=14)
            plt.tight_layout()
            
            save_path = output_dir / f"sample_{idx:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    print(f"Saved {len(samples)} visualizations to {output_dir}")


def visualize_predictions(
    model: nn.Module,
    samples: list,
    classes: list,
    input_shape: tuple,
    input_scale: tuple,
    device: torch.device,
    output_dir: Path,
    model_name: str,
    is_2d: bool = True
):
    """
    Generate and save visualization of predictions.
    
    Creates a grid showing raw input, ground truth, and predictions for each sample.
    
    Args:
        model: Trained model
        samples: List of samples to visualize
        classes: List of class names
        input_shape: Input shape
        input_scale: Input scale
        device: Device
        output_dir: Output directory for visualizations
        model_name: Name of model for file naming
        is_2d: Whether model is 2D
    """
    model.eval()
    
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save visualization metadata for reproducibility
    metadata_file = output_dir / "visualization_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'samples': samples,
            'classes': classes,
            'input_shape': list(input_shape),
            'input_scale': list(input_scale),
            'seed': VISUALIZATION_SEED,
        }, f, indent=2)
    
    # Create dataset for the visualization samples
    dataset = TestDataset(
        classes=classes,
        input_shape=input_shape,
        input_scale=input_scale,
        samples=samples
    )
    
    # Color map for classes
    cmap = plt.cm.get_cmap('tab20', len(classes))
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Generating visualizations"):
            sample = dataset[idx]
            
            if not sample['valid']:
                continue
            
            raw = sample['raw'].unsqueeze(0).to(device)
            gt = sample['gt']
            metadata = sample['metadata']
            
            # Forward pass
            outputs = model(raw)
            preds = torch.sigmoid(outputs).cpu()[0]
            preds_binary = (preds > 0.5).float()
            
            # Create visualization
            fig, axes = plt.subplots(3, 5, figsize=(20, 12))
            
            # Row 0: Raw image and class-colored overlay
            raw_np = sample['raw'][0].cpu().numpy() if sample['raw'].dim() == 3 else sample['raw'].cpu().numpy()
            
            axes[0, 0].imshow(raw_np, cmap='gray')
            axes[0, 0].set_title('Raw Input')
            axes[0, 0].axis('off')
            
            # GT overlay
            gt_overlay = np.zeros((*raw_np.shape, 3))
            for j, cls in enumerate(classes):
                gt_cls = gt[j].cpu().numpy()
                if gt_cls.max() > 0:
                    color = cmap(j)[:3]
                    for c in range(3):
                        gt_overlay[:, :, c] += gt_cls * color[c]
            gt_overlay = np.clip(gt_overlay, 0, 1)
            
            axes[0, 1].imshow(raw_np, cmap='gray', alpha=0.5)
            axes[0, 1].imshow(gt_overlay, alpha=0.5)
            axes[0, 1].set_title('GT Overlay')
            axes[0, 1].axis('off')
            
            # Pred overlay
            pred_overlay = np.zeros((*raw_np.shape, 3))
            for j, cls in enumerate(classes):
                pred_cls = preds_binary[j].cpu().numpy()
                if pred_cls.max() > 0:
                    color = cmap(j)[:3]
                    for c in range(3):
                        pred_overlay[:, :, c] += pred_cls * color[c]
            pred_overlay = np.clip(pred_overlay, 0, 1)
            
            axes[0, 2].imshow(raw_np, cmap='gray', alpha=0.5)
            axes[0, 2].imshow(pred_overlay, alpha=0.5)
            axes[0, 2].set_title('Pred Overlay')
            axes[0, 2].axis('off')
            
            # Show first few classes GT and Pred
            num_classes_to_show = min(4, len(classes))
            for j in range(num_classes_to_show):
                cls = classes[j]
                gt_cls = gt[j].cpu().numpy()
                pred_cls = preds_binary[j].cpu().numpy()
                
                # Row 1: Ground Truth
                axes[1, j].imshow(gt_cls, cmap='gray')
                axes[1, j].set_title(f'GT: {cls}')
                axes[1, j].axis('off')
                
                # Row 2: Predictions
                axes[2, j].imshow(pred_cls, cmap='gray')
                axes[2, j].set_title(f'Pred: {cls}')
                axes[2, j].axis('off')
            
            # Add legend
            legend_ax = axes[0, 3]
            legend_ax.axis('off')
            for j, cls in enumerate(classes[:min(7, len(classes))]):
                color = cmap(j)
                legend_ax.add_patch(plt.Rectangle((0, 1 - (j + 1) * 0.12), 0.3, 0.1, color=color))
                legend_ax.text(0.35, 1 - (j + 0.5) * 0.12, cls, fontsize=8, va='center')
            legend_ax.set_xlim(0, 1)
            legend_ax.set_ylim(0, 1)
            legend_ax.set_title('Legend')
            
            # Clear unused axes
            axes[0, 4].axis('off')
            if num_classes_to_show < 4:
                for j in range(num_classes_to_show, 4):
                    axes[1, j].axis('off')
                    axes[2, j].axis('off')
            axes[1, 4].axis('off')
            axes[2, 4].axis('off')
            
            # Add metadata
            crop_id = metadata.get('crop_id', 'unknown')
            slice_idx = metadata.get('slice_idx', 0)
            fig.suptitle(f'Crop {crop_id}, Slice {slice_idx}', fontsize=14)
            
            plt.tight_layout()
            
            # Save
            save_path = output_dir / f"sample_{idx:03d}_crop{crop_id}_slice{slice_idx}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    print(f"Saved {len(dataset)} visualizations to {output_dir}")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained model on CellMap test crops'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (e.g., checkpoints/unet_2d_best.pth)'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['unet', 'resnet', 'swin', 'vit'],
        help='Model architecture'
    )
    parser.add_argument(
        '--dim',
        type=str,
        required=True,
        choices=['2d', '3d'],
        help='Model dimensionality'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: evaluation_results/)'
    )
    parser.add_argument(
        '--num_vis_samples',
        type=int,
        default=NUM_VIS_SAMPLES,
        help=f'Number of samples to visualize (default: {NUM_VIS_SAMPLES})'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for evaluation (default: 8)'
    )
    parser.add_argument(
        '--no_visualize',
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: cuda if available)'
    )
    
    return parser.parse_args()



def main():
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Determine if 2D
    is_2d = args.dim == '2d'
    
    # Get input shape and scale
    input_shape = get_input_shape(args.dim)
    input_scale = INPUT_SCALE_2D if is_2d else INPUT_SCALE_3D
    
    # Create model
    print(f"Creating {args.model} ({args.dim}) model...")
    model = create_model(args.model, args.dim, device, CLASSES)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent / checkpoint_path
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    metadata = load_checkpoint(model, str(checkpoint_path), device)
    print(f"Loaded checkpoint from epoch {metadata['epoch']}")
    
    # Get classes from checkpoint or use default
    classes = metadata.get('classes', CLASSES)
    print(f"Evaluating on {len(classes)} classes: {classes}")
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else EVAL_OUTPUT_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = f"{args.model}_{args.dim}"
    
    # =========================================================
    # APPROACH 1: Use existing validation dataloader (preferred)
    # =========================================================
    print("\n" + "=" * 60)
    print("Loading validation data using existing dataloader...")
    print("=" * 60)
    
    try:
        val_loader = get_validation_dataloader(
            classes=classes,
            input_shape=input_shape,
            input_scale=input_scale,
            batch_size=args.batch_size,
            num_workers=4,
            datasplit_path=str(Path(__file__).parent / DATASPLIT_PATH)
        )
        
        if val_loader is not None:
            # Get fixed samples for visualization
            print(f"\nSelecting {args.num_vis_samples} fixed samples for visualization...")
            vis_samples = get_fixed_validation_samples(
                val_loader=val_loader,
                num_samples=args.num_vis_samples,
                min_classes=MIN_CLASSES_FOR_VIS,
                seed=VISUALIZATION_SEED,
                classes=classes
            )
            
            # Evaluate on full validation set
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            
            results = evaluate_model_on_dataloader(
                model=model,
                dataloader=val_loader,
                classes=classes,
                device=device,
                is_2d=is_2d
            )
            
            use_validation_loader = True
        else:
            print("Warning: No validation loader available, falling back to test crop method")
            use_validation_loader = False
            
    except Exception as e:
        print(f"Warning: Could not load validation dataloader: {e}")
        print("Falling back to test crop method...")
        use_validation_loader = False
    
    # =========================================================
    # APPROACH 2: Fallback to test crop indices (if validation fails)
    # =========================================================
    if not use_validation_loader:
        print("\nUsing test crop sample selection...")
        vis_samples_indices = get_sample_indices_for_visualization(
            test_crops=get_test_crops(),
            classes=classes,
            num_samples=args.num_vis_samples,
            seed=VISUALIZATION_SEED
        )
        print(f"Selected samples from {len(set(s['crop_id'] for s in vis_samples_indices))} unique crops")
        
        # Create dataset and dataloader
        print("\nCreating test dataset...")
        dataset = TestDataset(
            classes=classes,
            input_shape=input_shape,
            input_scale=input_scale,
            samples=vis_samples_indices
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Convert to vis_samples format
        vis_samples = []
        for batch in dataloader:
            for i in range(batch['raw'].size(0)):
                vis_samples.append({
                    'raw': batch['raw'][i],
                    'gt': batch['gt'][i],
                    'num_nonempty_classes': (batch['gt'][i].sum(dim=(-1, -2)) > 0).sum().item(),
                    'sample_id': len(vis_samples)
                })
                if len(vis_samples) >= args.num_vis_samples:
                    break
            if len(vis_samples) >= args.num_vis_samples:
                break
        
        # Evaluate
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        results = evaluate_model(model, dataloader, classes, device, is_2d)
    
    # Print results
    print(f"\nOverall Metrics (n={results['num_samples']} samples):")
    print(f"  Dice Score: {results['overall']['dice_mean']:.4f} ± {results['overall']['dice_std']:.4f}")
    print(f"  IoU Score:  {results['overall']['iou_mean']:.4f} ± {results['overall']['iou_std']:.4f}")
    print(f"  Precision:  {results['overall']['precision_mean']:.4f}")
    print(f"  Recall:     {results['overall']['recall_mean']:.4f}")
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Dice':<20} {'IoU':<20} {'Samples':<8}")
    print("-" * 65)
    for cls in classes:
        cls_results = results['per_class'][cls]
        dice_str = f"{cls_results['dice_mean']:.4f}±{cls_results.get('dice_std', 0):.4f}"
        iou_str = f"{cls_results['iou_mean']:.4f}±{cls_results.get('iou_std', 0):.4f}"
        print(f"{cls:<15} {dice_str:<20} {iou_str:<20} {cls_results.get('num_samples', 0):<8}")
    
    # Save results
    results_file = output_dir / f"{model_name}_results.json"
    results['model'] = model_name
    results['checkpoint'] = str(checkpoint_path)
    results['classes'] = classes
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Generate visualizations
    if not args.no_visualize and vis_samples:
        print(f"\nGenerating visualizations for {len(vis_samples)} samples...")
        visualize_samples(
            model=model,
            samples=vis_samples,
            classes=classes,
            device=device,
            output_dir=output_dir / "visualizations",
            model_name=model_name,
            is_2d=is_2d
        )
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
