"""
Partial Annotation Loss for CellMap segmentation.

Loss functions:
- PartialTverskyLoss: Per-channel Tversky (α=0.6, β=0.4) with annotation masking
- BalancedSoftmaxTverskyLoss: Logit-adjusted Tversky with online frequency estimation,
  annotation-mask-aware accumulation, partial annotation masking, AND per-class
  bounding-box spatial masking (box_class_mask_tight)
- PartialAnnotationDeepSupervisionLoss: Multi-scale wrapper for deep supervision

Based on results from THREE prior experiments (run on fewer classes + broken NIfTI data):
- loss_optimization: Per-class Tversky (α=0.6, β=0.4) was the best base loss
- class_weighting: Balanced Softmax τ=1.0 was the best weighting strategy
- masking_strategies: box_class_mask_tight was the best masking strategy

NOTE: These results need re-validation on all 35 classes with correct zarr data.
That is the purpose of the ablation experiments in training/configs/.
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
FG_THRESHOLD = 0.01


class PartialTverskyLoss(nn.Module):
    """Per-channel Tversky loss with partial annotation masking.

    Computes per-channel Tversky index, masks out unannotated channels,
    and averages only over annotated ones.

    Tversky(c) = (TP_c + ε) / (TP_c + α·FP_c + β·FN_c + ε)
    Loss(c) = 1 − Tversky(c)

    Args:
        alpha: FP weight. Higher → penalise false positives more.
        beta: FN weight. Higher → penalise false negatives more.
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
        """Compute per-channel Tversky loss.

        Args:
            input: Logits (B, C, *spatial).
            target: Binary ground truth (B, C, *spatial).

        Returns:
            Per-channel loss tensor (B, C).
        """
        pred = torch.sigmoid(input)
        target = target.float()

        spatial_dims = tuple(range(2, input.ndim))

        tp = (pred * target).sum(dim=spatial_dims)
        fp = (pred * (1.0 - target)).sum(dim=spatial_dims)
        fn = ((1.0 - pred) * target).sum(dim=spatial_dims)

        denom = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = (tp + self.smooth) / denom.clamp(min=self.smooth)

        return 1.0 - tversky  # (B, C)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self._annotation_mask
        self._annotation_mask = None

        per_channel_loss = self._per_channel_tversky_loss(input, target)

        if mask is not None:
            mask = mask.to(input.device)
            per_channel_loss = per_channel_loss * mask
            num_annotated = mask.sum(dim=1).clamp(min=1.0)
            per_sample_loss = per_channel_loss.sum(dim=1) / num_annotated
        else:
            per_sample_loss = per_channel_loss.mean(dim=1)

        return per_sample_loss.mean()


class BalancedSoftmaxTverskyLoss(nn.Module):
    """Logit-adjusted Tversky loss with partial annotation + spatial bbox masking.

    Shifts logits by a class-frequency prior before applying sigmoid:
        adjusted_logit_c = logit_c − τ · (log(n_c) − mean(log(n)))

    **Spatial bbox masking (box_class_mask_tight)**:
    For each annotated class, computes the bounding box of foreground voxels,
    pads by bbox_pad_fraction, sets weight=1.0 inside, bbox_bg_weight outside.
    Unannotated channels get weight=0.

    **Mask-supervised reconstruction (masksup)**:
    Randomly masks masksup_ratio of annotated voxels and adds a weighted
    reconstruction Tversky loss on the masked subset.

    Args:
        tau: Temperature for logit adjustment. Default 1.0.
        alpha: FP weight for Tversky. Default 0.6.
        beta: FN weight for Tversky. Default 0.4.
        smooth: Smoothing for Tversky. Default 1e-6.
        num_classes: Number of output classes.
        update_interval: Recompute adjustments every N batches. Default 50.
        bbox_pad_fraction: Fraction of bbox size to pad. Default 0.05.
        bbox_bg_weight: Weight for voxels outside all class bboxes. Default 0.05.
        masksup_ratio: Fraction of annotated voxels to mask. Default 0.0.
        masksup_recon_weight: Weight for reconstruction loss. Default 0.5.
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
        masksup_ratio: float = 0.0,
        masksup_recon_weight: float = 0.5,
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
        self.masksup_ratio = masksup_ratio
        self.masksup_recon_weight = masksup_recon_weight
        self._annotation_mask: Optional[torch.Tensor] = None
        self._foreground_mask: Optional[torch.Tensor] = None

        self.register_buffer(
            "logit_adj", torch.zeros(num_classes, dtype=torch.float32)
        )
        self.register_buffer(
            "_accum_counts", torch.ones(num_classes, dtype=torch.float64)
        )
        self._batch_counter = 0

    def set_annotation_mask(self, mask: torch.Tensor) -> None:
        self._annotation_mask = mask

    def set_foreground_mask(self, fg_mask: torch.Tensor) -> None:
        self._foreground_mask = fg_mask

    @staticmethod
    def _compute_adjustments(counts: torch.Tensor, tau: float) -> torch.Tensor:
        log_counts = torch.log(counts.clamp(min=1.0).float())
        centred = log_counts - log_counts.mean()
        return tau * centred

    def _accumulate(
        self, target: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> None:
        spatial_dims = tuple(range(2, target.ndim))
        positives = target.sum(dim=spatial_dims)

        if mask is not None:
            mask_dev = mask.to(target.device)
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
        B, C = target.shape[:2]
        spatial_shape = target.shape[2:]
        ndim_spatial = len(spatial_shape)
        device = target.device

        spatial_mask = torch.full(
            (B, C, *spatial_shape), self.bbox_bg_weight,
            device=device, dtype=target.dtype,
        )

        for b in range(B):
            for c in range(C):
                if annotation_mask is not None:
                    if annotation_mask[b, c] < 0.5:
                        spatial_mask[b, c] = 0.0
                        continue

                pos = target[b, c] > 0.5
                if not pos.any():
                    continue

                coords = torch.where(pos)
                slices = []
                for dim_idx in range(ndim_spatial):
                    dim_coords = coords[dim_idx]
                    lo = dim_coords.min().item()
                    hi = dim_coords.max().item()
                    extent = hi - lo + 1
                    pad = max(1, int(extent * self.bbox_pad_fraction))
                    lo = max(0, lo - pad)
                    hi = min(spatial_shape[dim_idx] - 1, hi + pad)
                    slices.append(slice(lo, hi + 1))

                spatial_mask[b, c][tuple(slices)] = 1.0

        if foreground_mask is not None:
            fg = foreground_mask.to(device=device, dtype=spatial_mask.dtype)
            spatial_mask = spatial_mask * fg

        return spatial_mask

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self._annotation_mask
        self._annotation_mask = None
        fg_mask = self._foreground_mask
        self._foreground_mask = None

        if self.training:
            self._accumulate(target.float(), mask)

        spatial_w = self._compute_spatial_mask(target, mask, fg_mask)

        adj = self.logit_adj.to(input.device)
        adj_shape = [1, self.num_classes] + [1] * (input.ndim - 2)
        adjusted_input = input - adj.view(*adj_shape)

        pred = torch.sigmoid(adjusted_input)
        spatial_dims = tuple(range(2, input.ndim))

        def _tversky_from_weights(w):
            tp = (w * pred * target).sum(dim=spatial_dims)
            fp = (w * pred * (1.0 - target)).sum(dim=spatial_dims)
            fn = (w * (1.0 - pred) * target).sum(dim=spatial_dims)
            denom = tp + self.alpha * fp + self.beta * fn + self.smooth
            tversky = (tp + self.smooth) / denom.clamp(min=self.smooth)
            return 1.0 - tversky

        if self.training and self.masksup_ratio > 0:
            rand = torch.rand_like(spatial_w)
            recon_mask = (spatial_w > 0) & (rand < self.masksup_ratio)
            visible_w = spatial_w * (~recon_mask).float()
            recon_w = spatial_w * recon_mask.float()

            per_channel_loss = _tversky_from_weights(visible_w)

            if recon_mask.any():
                recon_loss = _tversky_from_weights(recon_w)
                per_channel_loss = per_channel_loss + self.masksup_recon_weight * recon_loss
        else:
            per_channel_loss = _tversky_from_weights(spatial_w)

        if mask is not None:
            mask = mask.to(input.device)
            per_channel_loss = per_channel_loss * mask
            num_annotated = mask.sum(dim=1).clamp(min=1.0)
            per_sample_loss = per_channel_loss.sum(dim=1) / num_annotated
        else:
            per_sample_loss = per_channel_loss.mean(dim=1)

        return per_sample_loss.mean()


class PartialAnnotationDeepSupervisionLoss(nn.Module):
    """Deep supervision loss wrapper for partial annotation."""

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

                if pred.shape[2:] != target.shape[2:]:
                    t = F.interpolate(
                        target.float(), size=pred.shape[2:], mode="nearest"
                    )
                else:
                    t = target

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
    tversky_alpha: float = 0.6,
    tversky_beta: float = 0.4,
    smooth: float = 1e-6,
    tau: float = 1.0,
    update_interval: int = 50,
    bbox_pad_fraction: float = 0.05,
    bbox_bg_weight: float = 0.05,
    masksup_ratio: float = 0.0,
    masksup_recon_weight: float = 0.5,
    deep_supervision: bool = False,
    ds_weights: Optional[List[float]] = None,
) -> nn.Module:
    """Factory for building the partial annotation loss."""
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
            masksup_ratio=masksup_ratio,
            masksup_recon_weight=masksup_recon_weight,
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
