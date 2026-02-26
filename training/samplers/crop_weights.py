"""
Class-aware crop weighting sampler for cellmap-data.

Cherry-picked from OrganelleSeg's 3-layer sampling system. Assigns higher
sampling weights to crops that contain rare organelle classes, using inverse
square root of global voxel counts.

Compatible with CellMapDataLoader's `sampler: Callable` interface —
returns a callable that produces a fresh WeightedRandomSampler on each
call (i.e., each epoch via `refresh()`).

Usage:
    from training.samplers.crop_weights import make_class_aware_sampler
    sampler_fn = make_class_aware_sampler(
        dataset=train_loader.dataset,
        iterations_per_epoch=500,
        batch_size=8,
        blend_ratio=0.7,
    )
    # Pass to CellMapDataLoader as: sampler=sampler_fn
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger(__name__)


def compute_class_aware_weights(
    dataset,
    blend_ratio: float = 0.7,
) -> list[float]:
    """Compute per-sample weights using inverse-sqrt class-aware weighting.

    For each dataset (crop), we compute:
        class_score = Σ_{c ∈ annotated_classes} 1 / √(global_count(c))

    Then blend with uniform:
        weight = blend_ratio × class_score + (1 - blend_ratio) × uniform

    This upweights crops that contain rare organelles (e.g., NE, PO, MT-in)
    while maintaining some uniform coverage to avoid overfitting to rare classes.

    Args:
        dataset: CellMapMultiDataset with class_counts property.
        blend_ratio: Weight given to class-aware term (vs uniform). Default: 0.7.

    Returns:
        List of per-sample weights (one per flat index in the multi-dataset).
    """
    class_counts = dataset.class_counts
    totals = class_counts["totals"]

    # Get foreground counts per class (skip _bg keys)
    fg_counts = {}
    for cls_name, count in totals.items():
        if cls_name.endswith("_bg"):
            continue
        # Avoid division by zero for classes with no annotations
        fg_counts[cls_name] = max(count, 1.0)

    # Compute inverse sqrt weights per class
    inv_sqrt = {cls: 1.0 / math.sqrt(count) for cls, count in fg_counts.items()}

    # Normalize so max weight = 1 (prevents explosion)
    max_inv_sqrt = max(inv_sqrt.values()) if inv_sqrt else 1.0
    inv_sqrt_norm = {cls: w / max_inv_sqrt for cls, w in inv_sqrt.items()}

    # Compute per-dataset (crop) weight
    dataset_weights = []
    for sub_dataset in dataset.datasets:
        # Check which classes this dataset annotates
        # Each CellMapDataset has a 'classes' attribute
        ds_classes = getattr(sub_dataset, "classes", [])

        # Sum inverse-sqrt scores for classes present in this crop
        class_score = 0.0
        n_classes = 0
        for cls in ds_classes:
            if cls in inv_sqrt_norm:
                class_score += inv_sqrt_norm[cls]
                n_classes += 1

        # Average over annotated classes (avoid bias toward many-class crops)
        if n_classes > 0:
            class_score /= n_classes

        # Blend with uniform
        uniform = 1.0
        weight = blend_ratio * class_score + (1.0 - blend_ratio) * uniform

        # Each dataset has len() samples (grid cells)
        n_samples = len(sub_dataset)
        dataset_weights.append((weight, n_samples))

    # Expand to per-sample weights
    sample_weights = []
    for weight, n_samples in dataset_weights:
        sample_weights.extend([weight] * n_samples)

    # Log summary
    if dataset_weights:
        weights_only = [w for w, _ in dataset_weights]
        logger.info(
            f"Class-aware sampling: {len(dataset_weights)} datasets, "
            f"weight range [{min(weights_only):.3f}, {max(weights_only):.3f}], "
            f"mean={sum(weights_only)/len(weights_only):.3f}"
        )

    return sample_weights


def make_class_aware_sampler(
    dataset,
    iterations_per_epoch: int,
    batch_size: int,
    blend_ratio: float = 0.7,
    rng: Optional[torch.Generator] = None,
) -> callable:
    """Create a callable sampler factory for CellMapDataLoader.

    Returns a callable that, when called (by CellMapDataLoader.refresh()),
    produces a fresh WeightedRandomSampler with class-aware weights.

    This is compatible with CellMapDataLoader's `sampler: Callable` interface.

    Args:
        dataset: CellMapMultiDataset.
        iterations_per_epoch: Number of iterations per epoch.
        batch_size: Batch size.
        blend_ratio: Class-aware vs uniform blend. Default: 0.7.
        rng: Optional random number generator.

    Returns:
        Callable that returns a WeightedRandomSampler.
    """
    num_samples = iterations_per_epoch * batch_size
    weights = compute_class_aware_weights(dataset, blend_ratio=blend_ratio)
    weights_tensor = torch.tensor(weights, dtype=torch.float64)

    logger.info(
        f"Class-aware sampler: {len(weights)} total samples, "
        f"drawing {num_samples} per epoch"
    )

    def _make_sampler() -> WeightedRandomSampler:
        return WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=num_samples,
            replacement=True,
            generator=rng,
        )

    return _make_sampler
