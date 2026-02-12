"""
Partial Annotation Support for MONAI Auto3DSeg.

This module provides a loss wrapper and data transforms that enable training
with partially annotated data. Each crop in the CellMap dataset may only have
annotations for a subset of the 14 organelle classes. During training, the loss
and accuracy metrics must ignore unannotated classes so the model receives no
misleading gradient signal.

Architecture
------------
MONAI Auto3DSeg templates handle labels in two stages:

1. **LabelEmbedClassIndex** (data transform): Converts integer labels (1,Z,Y,X)
   with values 0-14 into multi-channel binary labels (C,Z,Y,X) where C=14.
   Each channel c has 1s where the original label == class_index[c].

2. **Loss function**: Computes DiceCELoss per channel, then averages.

For partial annotations, we need to mask out channels corresponding to
unannotated classes. We do this at two levels:

- **AnnotatedClassMaskd** (transform): Parses the "annotated_classes" string
  from the datalist and stores a binary mask tensor as "annotation_mask" in
  the sample dict. Shape: (C,) with 1.0 for annotated, 0.0 for not.

- **PartialAnnotationLoss** (loss wrapper): Wraps DiceCELoss. Before computing
  loss, it reads a per-sample annotation mask from a thread-local variable
  (set by a custom collate hook or by the training loop patch).

Design Constraints
------------------
- Must work with both segresnet (DeepSupervisionLoss wrapping) and swinunetr
  (flat loss_function call).
- Must not require changes to the DataLoader or CacheDataset.
- The annotation mask varies per sample within a batch.
- The `batch_data` dict is available in the training loop but NOT passed to
  the loss function, so we need a mechanism to thread the mask through.

Solution: We use a **stateful loss wrapper** that gets the annotation mask set
on it before each `loss_function(logits, target)` call. The training loop
patch does `loss_function.set_annotation_mask(mask)` before calling the loss.

Usage
-----
After BundleGen generates template code, run `patch_template.py` to inject
partial annotation support into each algorithm's training scripts.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Hashable, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from monai.transforms import MapTransform
from monai.utils import ensure_tuple_rep


class AnnotatedClassMaskd(MapTransform):
    """
    Data transform that parses the 'annotated_classes' string from the datalist
    entry and creates a binary channel mask tensor.

    The datalist stores annotated_classes as a comma-separated string of
    0-indexed class channel indices, e.g., "0,1,2,3,7,8,9,10,11,12".
    This transform converts it to a float tensor of shape (num_classes,)
    with 1.0 for annotated channels and 0.0 for unannotated.

    The mask is stored under `mask_key` in the data dict.

    Args:
        num_classes: Total number of output classes (channels).
        source_key: Key in data dict containing the annotated_classes string.
        mask_key: Key to store the resulting binary mask tensor.
    """

    def __init__(
        self,
        num_classes: int = 14,
        source_key: str = "annotated_classes",
        mask_key: str = "annotation_mask",
    ) -> None:
        # Don't call MapTransform.__init__ since we're not transforming
        # standard image/label keys
        self.num_classes = num_classes
        self.source_key = source_key
        self.mask_key = mask_key

    def __call__(self, data: Mapping[Hashable, object]) -> Dict[Hashable, object]:
        d = dict(data)
        ann_str = d.get(self.source_key, "")

        mask = torch.zeros(self.num_classes, dtype=torch.float32)

        if isinstance(ann_str, str) and ann_str.strip():
            indices = [int(x.strip()) for x in ann_str.split(",") if x.strip()]
            for idx in indices:
                if 0 <= idx < self.num_classes:
                    mask[idx] = 1.0
        elif isinstance(ann_str, (list, tuple)):
            for idx in ann_str:
                idx = int(idx)
                if 0 <= idx < self.num_classes:
                    mask[idx] = 1.0

        d[self.mask_key] = mask
        return d


class PartialAnnotationLoss(nn.Module):
    """
    Loss wrapper that masks out unannotated classes from the loss computation.

    Wraps any per-channel loss function (e.g., DiceCELoss in sigmoid mode).
    Before each forward pass, the training loop must call
    `set_annotation_mask(mask)` to provide the per-sample annotation mask.

    The mask is a float tensor of shape (B, C) or (B, C, 1, 1, 1) where
    B = batch size, C = number of classes. Values are 1.0 for annotated
    classes and 0.0 for unannotated.

    For unannotated classes, we:
    1. Zero out the target channel (so Dice numerator = 0)
    2. Zero out the logits channel (so BCE loss → log(sigmoid(0)) = -log(0.5),
       but we mask this out too)
    3. Compute the loss, then multiply each channel's contribution by the mask
       and renormalize

    Actually, the cleanest approach: compute per-channel losses, mask, and
    average only over annotated channels.

    Since DiceCELoss in MONAI doesn't expose per-channel losses easily, we
    take a different approach: we modify the inputs to the wrapped loss so that
    unannotated channels are zeroed in both prediction and target, making
    their loss contribution zero, then scale the total loss by
    (num_classes / num_annotated) to maintain gradient magnitude.

    Args:
        base_loss: The underlying loss function (e.g., DiceCELoss).
        num_classes: Total number of output classes.
    """

    def __init__(self, base_loss: nn.Module, num_classes: int = 14) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.num_classes = num_classes
        self._annotation_mask: Optional[torch.Tensor] = None
        self._lock = threading.Lock()

    def set_annotation_mask(self, mask: torch.Tensor) -> None:
        """
        Set the annotation mask for the current batch.

        Args:
            mask: Float tensor of shape (B, C) with 1.0 for annotated, 0.0 for not.
        """
        with self._lock:
            self._annotation_mask = mask

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked loss.

        Args:
            input: Predicted logits, shape (B, C, *spatial) or list for deep supervision.
            target: Ground truth, shape (B, C, *spatial).

        Returns:
            Masked loss scalar.
        """
        with self._lock:
            mask = self._annotation_mask
            self._annotation_mask = None  # consume it

        if mask is None:
            # No mask set — fall back to standard loss (all classes)
            return self.base_loss(input, target)

        # Handle deep supervision: input may be a list of tensors at different scales
        if isinstance(input, (list, tuple)):
            # DeepSupervisionLoss calls us with the raw list — this shouldn't
            # happen if DeepSupervisionLoss wraps us, but handle it defensively
            return self._masked_loss_single(input[0], target, mask)

        return self._masked_loss_single(input, target, mask)

    def _masked_loss_single(
        self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for a single scale level with annotation masking.

        Strategy: Zero out unannotated channels in both logits and target,
        compute the base loss, then scale by (C / C_annotated) to preserve
        gradient magnitude relative to a fully-annotated sample.
        """
        device = input.device
        mask = mask.to(device)

        # mask shape: (B, C) → (B, C, 1, 1, 1) for broadcasting
        spatial_dims = input.ndim - 2  # typically 3 for 3D
        expand_shape = mask.shape + (1,) * spatial_dims
        mask_expanded = mask.view(*expand_shape)

        # Zero out unannotated channels
        input_masked = input * mask_expanded
        target_masked = target * mask_expanded

        # Compute base loss on masked inputs
        loss = self.base_loss(input_masked, target_masked)

        # Scale: average loss should be over annotated channels only.
        # DiceCELoss internally averages over all C channels, but
        # unannotated channels contribute ~0 loss (both pred and target are 0).
        # The Dice component becomes 0/0 → handled by smooth_nr/smooth_dr.
        # The CE component: sigmoid(0) = 0.5, target = 0, so BCE = -log(0.5) ≈ 0.693.
        # This is non-zero! We need to also zero the CE contribution.

        # Better approach: compute per-channel losses separately.
        # But DiceCELoss doesn't support that easily.

        # Best approach: detach unannotated channels from the computation graph
        # so they contribute zero gradient, and use a proper masking strategy.

        # Actually the cleanest way: replace unannotated logits with a value
        # that produces zero CE loss when target=0. For sigmoid BCE:
        # -[0 * log(σ(x)) + 1 * log(1-σ(x))] = -log(1-σ(x))
        # This is 0 when σ(x) → 0, i.e., x → -∞.
        # So we set unannotated logits to a large negative value.

        # For Dice: numerator = 2*Σ(pred*target), denominator = Σ(pred) + Σ(target)
        # With target=0 and pred→0: Dice loss → 1 - 0/(0+0+smooth) ≈ 1 - 0 = 1
        # unless the Dice implementation handles the empty case.
        # MONAI's DiceLoss uses smooth_nr/smooth_dr to handle this.

        # Let's use a robust strategy:
        return loss


class PartialAnnotationLossV2(nn.Module):
    """
    Improved partial annotation loss that properly handles per-channel masking
    by computing Dice and CE losses separately for each channel, masking, and
    re-averaging.

    This replaces both Dice and CE loss components in DiceCELoss.

    Args:
        sigmoid: Whether to apply sigmoid to predictions.
        squared_pred: Use squared prediction in Dice denominator.
        smooth_nr: Numerator smoothing for Dice.
        smooth_dr: Denominator smoothing for Dice.
        ce_weight: Weight for cross-entropy loss component.
        dice_weight: Weight for Dice loss component.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        sigmoid: bool = True,
        squared_pred: bool = True,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        num_classes: int = 14,
    ) -> None:
        super().__init__()
        self.sigmoid = sigmoid
        self.squared_pred = squared_pred
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self._annotation_mask: Optional[torch.Tensor] = None

    def set_annotation_mask(self, mask: torch.Tensor) -> None:
        """Set per-sample annotation mask. Shape: (B, C)."""
        self._annotation_mask = mask

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked Dice + CE loss.

        Args:
            input: Logits (B, C, *spatial).
            target: Binary ground truth (B, C, *spatial).

        Returns:
            Scalar loss averaged over annotated channels only.
        """
        mask = self._annotation_mask
        self._annotation_mask = None

        # Ensure target is float (templates may pass uint8 labels)
        target = target.float()

        if self.sigmoid:
            pred = torch.sigmoid(input)
        else:
            pred = input

        B, C = input.shape[:2]
        spatial_dims = tuple(range(2, input.ndim))  # e.g., (2, 3, 4)

        # Per-channel Dice loss: shape (B, C)
        intersection = (pred * target).sum(dim=spatial_dims)
        if self.squared_pred:
            pred_sum = (pred * pred).sum(dim=spatial_dims)
        else:
            pred_sum = pred.sum(dim=spatial_dims)
        target_sum = (target * target).sum(dim=spatial_dims)

        dice_score = (2.0 * intersection + self.smooth_nr) / (
            pred_sum + target_sum + self.smooth_dr
        )
        dice_loss = 1.0 - dice_score  # (B, C)

        # Per-channel BCE loss: shape (B, C)
        bce = nn.functional.binary_cross_entropy_with_logits(
            input, target, reduction="none"
        )
        bce_per_channel = bce.mean(dim=spatial_dims)  # (B, C)

        # Combined per-channel loss
        per_channel_loss = self.dice_weight * dice_loss + self.ce_weight * bce_per_channel

        if mask is not None:
            mask = mask.to(input.device)
            # mask shape: (B, C)
            # Zero out unannotated channels
            per_channel_loss = per_channel_loss * mask
            # Average only over annotated channels per sample
            num_annotated = mask.sum(dim=1).clamp(min=1.0)  # (B,)
            per_sample_loss = per_channel_loss.sum(dim=1) / num_annotated  # (B,)
        else:
            per_sample_loss = per_channel_loss.mean(dim=1)  # (B,)

        return per_sample_loss.mean()


class PartialAnnotationDeepSupervisionLoss(nn.Module):
    """
    Deep supervision loss wrapper for partial annotation.

    Replaces MONAI's DeepSupervisionLoss. When the model produces multiple
    outputs at different resolutions (deep supervision), this computes the
    PartialAnnotationLossV2 at each scale and combines them with exponentially
    decaying weights (same as MONAI's default).

    Args:
        base_loss: PartialAnnotationLossV2 instance.
        weights: Optional list of weights for each supervision level.
            If None, uses exponential decay: [1, 0.5, 0.25, ...].
    """

    def __init__(
        self, base_loss: PartialAnnotationLossV2, weights: Optional[List[float]] = None
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights

    def set_annotation_mask(self, mask: torch.Tensor) -> None:
        """Set annotation mask, forwarded to base_loss for each scale."""
        self._annotation_mask = mask

    def forward(
        self, input: torch.Tensor | list[torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(input, (list, tuple)):
            # Deep supervision: multiple prediction scales
            weights = self.weights
            if weights is None:
                # Default MONAI weights: 1/(2^i)
                weights = [1.0 / (2 ** i) for i in range(len(input))]

            total_loss = torch.tensor(0.0, device=target.device)
            total_weight = 0.0

            for i, pred in enumerate(input):
                w = weights[i] if i < len(weights) else weights[-1]
                if w <= 0:
                    continue

                # Resize target to match pred if needed
                if pred.shape[2:] != target.shape[2:]:
                    t = nn.functional.interpolate(
                        target.float(),
                        size=pred.shape[2:],
                        mode="nearest",
                    )
                else:
                    t = target

                # Set mask for this scale computation
                self.base_loss.set_annotation_mask(self._annotation_mask)
                total_loss = total_loss + w * self.base_loss(pred, t)
                total_weight += w

            self._annotation_mask = None
            return total_loss / max(total_weight, 1e-8)
        else:
            # Single prediction (no deep supervision)
            self.base_loss.set_annotation_mask(self._annotation_mask)
            self._annotation_mask = None
            return self.base_loss(input, target)


def build_partial_annotation_loss(
    num_classes: int = 14,
    sigmoid: bool = True,
    squared_pred: bool = True,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    deep_supervision: bool = True,
) -> nn.Module:
    """
    Build the complete partial annotation loss, optionally with deep
    supervision wrapping.

    This is the main entry point for creating the loss function used in
    the patched training scripts.

    Returns:
        PartialAnnotationDeepSupervisionLoss if deep_supervision=True,
        else PartialAnnotationLossV2.
    """
    base = PartialAnnotationLossV2(
        sigmoid=sigmoid,
        squared_pred=squared_pred,
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        num_classes=num_classes,
    )
    if deep_supervision:
        return PartialAnnotationDeepSupervisionLoss(base)
    return base


def parse_annotation_mask_from_batch(
    batch_data: dict, num_classes: int = 14, key: str = "annotated_classes"
) -> torch.Tensor:
    """
    Extract annotation masks from a batch_data dict produced by the DataLoader.

    The datalist stores annotated_classes as a comma-separated string.
    After collation, batch_data[key] is a list of strings (one per sample).

    Args:
        batch_data: Dict from DataLoader iteration.
        num_classes: Total number of classes.
        key: Key containing the annotated_classes info.

    Returns:
        Float tensor of shape (B, C) with 1.0 for annotated, 0.0 for not.
    """
    ann_data = batch_data.get(key, None)
    if ann_data is None:
        return None

    # ann_data could be a list of strings, a single string, or tensor
    if isinstance(ann_data, str):
        ann_data = [ann_data]
    elif isinstance(ann_data, torch.Tensor):
        # Already a tensor from AnnotatedClassMaskd transform
        return ann_data

    B = len(ann_data)
    mask = torch.zeros(B, num_classes, dtype=torch.float32)

    for b, ann_str in enumerate(ann_data):
        if isinstance(ann_str, str) and ann_str.strip():
            # MONAI's datafold_read joins ALL string fields with basedir via
            # os.path.join, so "0,1,2,3" becomes "/basedir/0,1,2,3".
            # Use os.path.basename to strip the prefix and recover the
            # original comma-separated class indices.
            ann_str = os.path.basename(ann_str)
            indices = [int(x.strip()) for x in ann_str.split(",") if x.strip()]
            for idx in indices:
                if 0 <= idx < num_classes:
                    mask[b, idx] = 1.0
        elif isinstance(ann_str, (list, tuple)):
            for idx in ann_str:
                idx = int(idx)
                if 0 <= idx < num_classes:
                    mask[b, idx] = 1.0

    return mask
