"""
Partial Annotation Loss for CellMap segmentation.

Loss functions:
- PartialTverskyLoss: Per-channel Tversky (α=0.6, β=0.4) with annotation masking
- BalancedSoftmaxTverskyLoss: Logit-adjusted Tversky with online frequency estimation,
  annotation-mask-aware accumulation, partial annotation masking, AND per-class
  bounding-box spatial masking (box_class_mask_tight)
- PartialAnnotationDeepSupervisionLoss: Multi-scale wrapper for deep supervision

Based on results from THREE prior experiments:
- loss_optimization: Per-class Tversky (α=0.6, β=0.4) was the best base loss
- class_weighting: Balanced Softmax τ=1.0 was the best weighting strategy (0.5711 mean Dice)
- masking_strategies: box_class_mask_tight was the best masking strategy (0.376 eval Dice,
  +55% over no_mask baseline of 0.243). It computes per-class 3D bounding boxes around
  annotated foreground, applies full weight inside bbox (+ 5% padding), and bg_weight=0.05
  outside. This provides a proper negative signal in annotated regions while de-weighting
  predictions far from any annotation — preventing the model from learning to predict
  everything as positive (the degenerate mode seen in R2).

Adapted from:
- auto3dseg/partial_annotation.py (partial annotation handling)
- experiments/class_weighting/losses_class_weighting.py (Tversky + Balanced Softmax)
- experiments/masking_strategies/masking_losses.py (BoxClassMaskTverskyLoss)
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Foreground masking threshold — pixels below this in the normalized [0,1]
# EM image are black padding (from zarr boundary or SpatialPad). Loss is
# zeroed on these voxels to prevent false-positive penalties on empty regions.
# This was the single biggest gain in 2D experiments: +110% baseline Dice.
FG_THRESHOLD = 0.01


class PartialTverskyLoss(nn.Module):
    """Per-channel Tversky loss with partial annotation masking.

    Computes per-channel Tversky index, masks out unannotated channels,
    and averages only over annotated ones. Handles zero-annotation crops
    safely via .clamp(min=1.0).

    Tversky(c) = (TP_c + ε) / (TP_c + α·FP_c + β·FN_c + ε)
    Loss(c) = 1 − Tversky(c)

    Args:
        alpha: FP weight. Higher → penalise false positives more (precision bias).
            Default 0.6 per loss_optimization experiment results.
        beta: FN weight. Higher → penalise false negatives more (recall bias).
            Default 0.4.
        smooth: Smoothing to prevent division by zero.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        smooth: float = 1e-6,
        num_classes: int = 35,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes
        self._annotation_mask: Optional[torch.Tensor] = None

    def set_annotation_mask(self, mask: torch.Tensor) -> None:
        """Set per-sample annotation mask. Shape: (B, C)."""
        self._annotation_mask = mask

    def _per_channel_tversky_loss(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-channel Tversky loss (vectorized for 3D).

        Args:
            input: Logits (B, C, *spatial).
            target: Binary ground truth (B, C, *spatial).

        Returns:
            Per-channel loss tensor (B, C).
        """
        pred = torch.sigmoid(input)
        target = target.float()

        spatial_dims = tuple(range(2, input.ndim))  # e.g., (2, 3, 4)

        # TP, FP, FN per sample per channel: (B, C)
        tp = (pred * target).sum(dim=spatial_dims)
        fp = (pred * (1.0 - target)).sum(dim=spatial_dims)
        fn = ((1.0 - pred) * target).sum(dim=spatial_dims)

        denom = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = (tp + self.smooth) / denom.clamp(min=self.smooth)

        return 1.0 - tversky  # (B, C)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute masked Tversky loss.

        Args:
            input: Logits (B, C, *spatial).
            target: Binary ground truth (B, C, *spatial).

        Returns:
            Scalar loss averaged over annotated channels only.
        """
        mask = self._annotation_mask
        self._annotation_mask = None

        per_channel_loss = self._per_channel_tversky_loss(input, target)  # (B, C)

        if mask is not None:
            mask = mask.to(input.device)
            per_channel_loss = per_channel_loss * mask
            num_annotated = mask.sum(dim=1).clamp(min=1.0)  # (B,)
            per_sample_loss = per_channel_loss.sum(dim=1) / num_annotated
        else:
            per_sample_loss = per_channel_loss.mean(dim=1)

        return per_sample_loss.mean()


class BalancedSoftmaxTverskyLoss(nn.Module):
    """Logit-adjusted Tversky loss with partial annotation + spatial bbox masking.

    Shifts logits by a class-frequency prior before applying sigmoid in the
    Tversky computation. Rare classes get a positive offset → sigmoid biased
    upward → model needs less evidence to predict the rare class.

        adjusted_logit_c = logit_c − τ · (log(n_c) − mean(log(n)))

    The winning weighting strategy from class_weighting experiment (τ=1.0,
    mean Dice 0.5711, beating inverse-frequency, CB loss, and Seesaw).

    **Spatial bbox masking (box_class_mask_tight)**:
    For each annotated class, computes the 3D bounding box of foreground
    voxels, pads it by `bbox_pad_fraction` of the bbox size, and creates a
    spatial weight mask: 1.0 inside the padded bbox, `bbox_bg_weight` outside.
    Unannotated channels get a spatial mask of 0 (no loss contribution).
    This prevents the degenerate mode where the model predicts everything
    positive because there's no spatial false-positive penalty outside
    annotated foreground regions.

    Results from masking_strategies experiment:
        box_class_mask_tight (pad=0.05, bg=0.05): 0.376 eval Dice (+55% vs no_mask)

    Args:
        tau: Temperature for logit adjustment. Default 1.0 (theory-optimal).
        alpha: FP weight for Tversky. Default 0.6.
        beta: FN weight for Tversky. Default 0.4.
        smooth: Smoothing for Tversky. Default 1e-6.
        num_classes: Number of output classes.
        update_interval: Recompute adjustments every N batches. Default 50.
        bbox_pad_fraction: Fraction of bbox size to pad. Default 0.05.
        bbox_bg_weight: Weight for voxels outside all class bboxes. Default 0.05.
    """

    def __init__(
        self,
        tau: float = 1.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        smooth: float = 1e-6,
        num_classes: int = 35,
        update_interval: int = 50,
        bbox_pad_fraction: float = 0.05,
        bbox_bg_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes
        self.update_interval = update_interval
        self.bbox_pad_fraction = bbox_pad_fraction
        self.bbox_bg_weight = bbox_bg_weight
        self._annotation_mask: Optional[torch.Tensor] = None
        self._foreground_mask: Optional[torch.Tensor] = None

        # Online frequency estimation buffers
        self.register_buffer(
            "logit_adj", torch.zeros(num_classes, dtype=torch.float32)
        )
        self.register_buffer(
            "_accum_counts", torch.ones(num_classes, dtype=torch.float64)
        )  # init to 1 to avoid log(0)
        self._batch_counter = 0

    def set_annotation_mask(self, mask: torch.Tensor) -> None:
        """Set per-sample annotation mask. Shape: (B, C)."""
        self._annotation_mask = mask

    def set_foreground_mask(self, fg_mask: torch.Tensor) -> None:
        """Set foreground mask from input EM image. Shape: (B, 1, *spatial).

        Voxels where fg_mask=False are black padding — loss contribution
        is zeroed. This was the single biggest gain in 2D experiments
        (+110% baseline Dice improvement).
        """
        self._foreground_mask = fg_mask

    @staticmethod
    def _compute_adjustments(
        counts: torch.Tensor, tau: float
    ) -> torch.Tensor:
        """Compute centred log-frequency adjustments.

        Args:
            counts: (C,) positive voxel counts per class.
            tau: Temperature scaling.

        Returns:
            (C,) logit adjustments. Positive = frequent class, negative = rare.
        """
        log_counts = torch.log(counts.clamp(min=1.0).float())
        centred = log_counts - log_counts.mean()
        return tau * centred

    def _accumulate(
        self, target: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> None:
        """Accumulate positive voxel counts from annotated channels only.

        Args:
            target: (B, C, *spatial) binary targets.
            mask: (B, C) annotation mask, or None.
        """
        spatial_dims = tuple(range(2, target.ndim))

        # Positive voxels per sample per channel: (B, C)
        positives = target.sum(dim=spatial_dims)  # (B, C)

        if mask is not None:
            mask_dev = mask.to(target.device)
            # Only accumulate from annotated channels
            positives = positives * mask_dev

        self._accum_counts += positives.sum(dim=0).double()
        self._batch_counter += 1

        if self._batch_counter % self.update_interval == 0:
            self.logit_adj.copy_(
                self._compute_adjustments(self._accum_counts, self.tau)
            )

    def _compute_spatial_mask(
        self,
        target: torch.Tensor,
        annotation_mask: Optional[torch.Tensor],
        foreground_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-class 3D bounding-box spatial weight mask.

        For each annotated class in each sample:
        1. Find bounding box of foreground voxels in (D, H, W)
        2. Pad bbox by bbox_pad_fraction of its size in each dimension
        3. Set voxels inside padded bbox to 1.0, outside to bbox_bg_weight
        4. Multiply by annotation mask (unannotated channels → all zeros)
        5. Zero out voxels where the input EM is black padding (foreground mask)

        For annotated classes with NO foreground voxels in this crop:
        - bbox_bg_weight everywhere → model IS penalised for any positive
          predictions, providing proper negative signal.

        Args:
            target: (B, C, *spatial) binary ground truth.
            annotation_mask: (B, C) annotation mask, or None.
            foreground_mask: (B, 1, *spatial) boolean mask, True = real EM data,
                False = black padding. If None, all voxels are treated as foreground.

        Returns:
            (B, C, *spatial) spatial weight mask.
        """
        B, C = target.shape[:2]
        spatial_shape = target.shape[2:]  # (D, H, W) or (H, W)
        ndim_spatial = len(spatial_shape)
        device = target.device

        # Start with bg_weight everywhere
        spatial_mask = torch.full(
            (B, C, *spatial_shape), self.bbox_bg_weight,
            device=device, dtype=target.dtype,
        )

        for b in range(B):
            for c in range(C):
                # Skip unannotated channels
                if annotation_mask is not None:
                    if annotation_mask[b, c] < 0.5:
                        spatial_mask[b, c] = 0.0
                        continue

                # Find foreground voxels for this sample and class
                pos = target[b, c] > 0.5
                if not pos.any():
                    # Annotated but no foreground → bg_weight everywhere
                    # (already set by default). Provides negative-only signal.
                    continue

                # Compute bounding box in each spatial dimension
                coords = torch.where(pos)  # tuple of (D_indices, H_indices, W_indices)

                slices = []
                for dim_idx in range(ndim_spatial):
                    dim_coords = coords[dim_idx]
                    lo = dim_coords.min().item()
                    hi = dim_coords.max().item()

                    # Pad by fraction of bbox extent
                    extent = hi - lo + 1
                    pad = max(1, int(extent * self.bbox_pad_fraction))
                    lo = max(0, lo - pad)
                    hi = min(spatial_shape[dim_idx] - 1, hi + pad)
                    slices.append(slice(lo, hi + 1))

                # Set inside padded bbox to 1.0
                spatial_mask[b, c][tuple(slices)] = 1.0

        # Zero out black-padding voxels (foreground masking fix)
        # This is the single biggest gain from 2D experiments (+110% Dice).
        # fg_mask is (B, 1, *spatial) → broadcast across C channels.
        if foreground_mask is not None:
            fg = foreground_mask.to(device=device, dtype=spatial_mask.dtype)
            spatial_mask = spatial_mask * fg  # (B, C, *spatial) × (B, 1, *spatial)

        return spatial_mask

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute logit-adjusted Tversky loss with spatial bbox masking.

        The spatial mask weights TP/FP/FN per-voxel before summing over
        spatial dimensions. This means:
        - Inside padded bbox: full contribution (weight=1.0)
        - Outside bbox: reduced contribution (weight=bbox_bg_weight=0.05)
        - Unannotated channels: zero contribution (weight=0.0)

        Args:
            input: Logits (B, C, *spatial).
            target: Binary ground truth (B, C, *spatial).

        Returns:
            Scalar loss averaged over annotated channels only.
        """
        mask = self._annotation_mask
        self._annotation_mask = None
        fg_mask = self._foreground_mask
        self._foreground_mask = None

        # Online frequency accumulation (training only)
        # Use float32 for accumulation precision, but keep target in
        # its original dtype (bf16 under autocast) for loss computation
        # to avoid OOM from 4 GiB fp32 intermediates at 160³×35.
        if self.training:
            self._accumulate(target.float(), mask)

        # Compute spatial bbox mask: (B, C, *spatial)
        # Includes foreground masking (zeroes out black-padding voxels)
        spatial_w = self._compute_spatial_mask(target, mask, fg_mask)

        # Apply logit adjustment per channel
        adj = self.logit_adj.to(input.device)
        adj_shape = [1, self.num_classes] + [1] * (input.ndim - 2)
        adjusted_input = input - adj.view(*adj_shape)

        # Tversky with spatial weighting
        pred = torch.sigmoid(adjusted_input)
        spatial_dims = tuple(range(2, input.ndim))

        # Spatially weighted TP/FP/FN
        tp = (spatial_w * pred * target).sum(dim=spatial_dims)                  # (B, C)
        fp = (spatial_w * pred * (1.0 - target)).sum(dim=spatial_dims)          # (B, C)
        fn = (spatial_w * (1.0 - pred) * target).sum(dim=spatial_dims)          # (B, C)

        denom = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = (tp + self.smooth) / denom.clamp(min=self.smooth)
        per_channel_loss = 1.0 - tversky  # (B, C)

        # Average over annotated channels only (spatial mask already zeros
        # out unannotated channels, but we still need to divide by the
        # correct count of annotated channels per sample)
        if mask is not None:
            mask = mask.to(input.device)
            per_channel_loss = per_channel_loss * mask
            num_annotated = mask.sum(dim=1).clamp(min=1.0)  # (B,)
            per_sample_loss = per_channel_loss.sum(dim=1) / num_annotated
        else:
            per_sample_loss = per_channel_loss.mean(dim=1)

        return per_sample_loss.mean()


class PartialAnnotationDeepSupervisionLoss(nn.Module):
    """Deep supervision loss wrapper for partial annotation.

    Computes the partial annotation loss at each output scale and combines
    with configurable weights (default: exponential decay [1, 0.5, 0.25, ...]).

    Args:
        base_loss: Loss with .set_annotation_mask() method.
        weights: Optional list of weights per supervision level.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights
        self._annotation_mask: Optional[torch.Tensor] = None
        self._foreground_mask: Optional[torch.Tensor] = None

    def set_annotation_mask(self, mask: torch.Tensor) -> None:
        self._annotation_mask = mask

    def set_foreground_mask(self, fg_mask: torch.Tensor) -> None:
        self._foreground_mask = fg_mask

    def forward(
        self,
        input: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(input, (list, tuple)):
            weights = self.weights
            if weights is None:
                weights = [1.0 / (2**i) for i in range(len(input))]

            total_loss = torch.tensor(0.0, device=target.device)
            total_weight = 0.0

            for i, pred in enumerate(input):
                w = weights[i] if i < len(weights) else weights[-1]
                if w <= 0:
                    continue

                # Resize target to match prediction if needed
                if pred.shape[2:] != target.shape[2:]:
                    t = F.interpolate(
                        target.float(), size=pred.shape[2:], mode="nearest"
                    )
                else:
                    t = target

                # Resize foreground mask to match prediction if needed
                fg = self._foreground_mask
                if fg is not None and pred.shape[2:] != fg.shape[2:]:
                    fg = F.interpolate(
                        fg.float(), size=pred.shape[2:], mode="nearest"
                    ) > 0.5

                self.base_loss.set_annotation_mask(self._annotation_mask)
                if hasattr(self.base_loss, 'set_foreground_mask'):
                    self.base_loss.set_foreground_mask(fg)
                total_loss = total_loss + w * self.base_loss(pred, t)
                total_weight += w

            self._annotation_mask = None
            self._foreground_mask = None
            return total_loss / max(total_weight, 1e-8)
        else:
            self.base_loss.set_annotation_mask(self._annotation_mask)
            if hasattr(self.base_loss, 'set_foreground_mask'):
                self.base_loss.set_foreground_mask(self._foreground_mask)
            self._annotation_mask = None
            self._foreground_mask = None
            return self.base_loss(input, target)


def build_partial_annotation_loss(
    num_classes: int = 35,
    loss_type: str = "balanced_softmax_tversky",
    # Tversky parameters
    tversky_alpha: float = 0.6,
    tversky_beta: float = 0.4,
    smooth: float = 1e-6,
    # Balanced Softmax parameters
    tau: float = 1.0,
    update_interval: int = 50,
    # Spatial bbox masking parameters (box_class_mask_tight)
    bbox_pad_fraction: float = 0.05,
    bbox_bg_weight: float = 0.05,
    # Deep supervision
    deep_supervision: bool = False,
    ds_weights: Optional[List[float]] = None,
) -> nn.Module:
    """Factory for building the partial annotation loss.

    Args:
        num_classes: Number of output channels.
        loss_type: One of 'tversky', 'balanced_softmax_tversky'.
        tversky_alpha: FP penalty weight (0.6 = precision bias).
        tversky_beta: FN penalty weight (0.4).
        smooth: Tversky smoothing.
        tau: Balanced Softmax temperature (1.0 = theory-optimal).
        update_interval: Batches between logit adjustment updates.
        bbox_pad_fraction: Fraction of bbox to pad (0.05 = tight).
        bbox_bg_weight: Weight outside bbox (0.05 = strongly de-weighted).
        deep_supervision: Wrap with multi-scale DS loss.
        ds_weights: Per-level weights for deep supervision.

    Returns:
        Loss module with .set_annotation_mask(mask) method.
    """
    if loss_type == "balanced_softmax_tversky":
        base = BalancedSoftmaxTverskyLoss(
            tau=tau,
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=smooth,
            num_classes=num_classes,
            update_interval=update_interval,
            bbox_pad_fraction=bbox_pad_fraction,
            bbox_bg_weight=bbox_bg_weight,
        )
    elif loss_type == "tversky":
        base = PartialTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=smooth,
            num_classes=num_classes,
        )
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            f"Choose from: 'tversky', 'balanced_softmax_tversky'"
        )

    if deep_supervision:
        return PartialAnnotationDeepSupervisionLoss(base, weights=ds_weights)
    return base
