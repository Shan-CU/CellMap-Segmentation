"""
Partial Annotation Loss for CellMap segmentation.

Loss functions:
- PartialTverskyLoss: Per-channel Tversky (α=0.6, β=0.4) with annotation masking
- BalancedSoftmaxTverskyLoss: Logit-adjusted Tversky with online frequency estimation,
  annotation-mask-aware accumulation, and partial annotation masking
- PartialAnnotationDeepSupervisionLoss: Multi-scale wrapper for deep supervision

Based on results from two prior experiments:
- loss_optimization: Per-class Tversky (α=0.6, β=0.4) was the best base loss
- class_weighting: Balanced Softmax τ=1.0 was the best weighting strategy (0.5711 mean Dice)

Adapted from:
- auto3dseg/partial_annotation.py (partial annotation handling)
- experiments/class_weighting/losses_class_weighting.py (Tversky + Balanced Softmax)
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        num_classes: int = 14,
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
    """Logit-adjusted Tversky loss with partial annotation masking.

    Shifts logits by a class-frequency prior before applying sigmoid in the
    Tversky computation. Rare classes get a positive offset → sigmoid biased
    upward → model needs less evidence to predict the rare class.

        adjusted_logit_c = logit_c − τ · (log(n_c) − mean(log(n)))

    The winning weighting strategy from class_weighting experiment (τ=1.0,
    mean Dice 0.5711, beating inverse-frequency, CB loss, and Seesaw).

    Frequency estimation is done online: accumulates positive voxel counts
    during training, recomputes adjustments every `update_interval` batches.
    **Annotation-mask-aware**: only counts voxels from annotated channels,
    preventing unannotated all-zero channels from biasing estimates.

    Args:
        tau: Temperature for logit adjustment. Default 1.0 (theory-optimal).
        alpha: FP weight for Tversky. Default 0.6.
        beta: FN weight for Tversky. Default 0.4.
        smooth: Smoothing for Tversky. Default 1e-6.
        num_classes: Number of output classes.
        update_interval: Recompute adjustments every N batches. Default 50.
    """

    def __init__(
        self,
        tau: float = 1.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        smooth: float = 1e-6,
        num_classes: int = 14,
        update_interval: int = 50,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes
        self.update_interval = update_interval
        self._annotation_mask: Optional[torch.Tensor] = None

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

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute logit-adjusted Tversky loss with annotation masking.

        Args:
            input: Logits (B, C, *spatial).
            target: Binary ground truth (B, C, *spatial).

        Returns:
            Scalar loss averaged over annotated channels only.
        """
        mask = self._annotation_mask
        self._annotation_mask = None

        target = target.float()

        # Online frequency accumulation (training only)
        if self.training:
            self._accumulate(target, mask)

        # Apply logit adjustment per channel
        # logit_adj shape (C,) → broadcast to (1, C, 1, 1, 1)
        adj = self.logit_adj.to(input.device)
        adj_shape = [1, self.num_classes] + [1] * (input.ndim - 2)
        adjusted_input = input - adj.view(*adj_shape)

        # Tversky with adjusted logits
        pred = torch.sigmoid(adjusted_input)
        spatial_dims = tuple(range(2, input.ndim))

        tp = (pred * target).sum(dim=spatial_dims)          # (B, C)
        fp = (pred * (1.0 - target)).sum(dim=spatial_dims)  # (B, C)
        fn = ((1.0 - pred) * target).sum(dim=spatial_dims)  # (B, C)

        denom = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = (tp + self.smooth) / denom.clamp(min=self.smooth)
        per_channel_loss = 1.0 - tversky  # (B, C)

        # Apply annotation mask
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

    def set_annotation_mask(self, mask: torch.Tensor) -> None:
        self._annotation_mask = mask

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

                self.base_loss.set_annotation_mask(self._annotation_mask)
                total_loss = total_loss + w * self.base_loss(pred, t)
                total_weight += w

            self._annotation_mask = None
            return total_loss / max(total_weight, 1e-8)
        else:
            self.base_loss.set_annotation_mask(self._annotation_mask)
            self._annotation_mask = None
            return self.base_loss(input, target)


def build_partial_annotation_loss(
    num_classes: int = 14,
    loss_type: str = "balanced_softmax_tversky",
    # Tversky parameters
    tversky_alpha: float = 0.6,
    tversky_beta: float = 0.4,
    smooth: float = 1e-6,
    # Balanced Softmax parameters
    tau: float = 1.0,
    update_interval: int = 50,
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
