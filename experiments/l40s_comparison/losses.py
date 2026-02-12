"""
Unified Loss Functions for L40S Model Comparison Experiment

Combines findings from three experiments:
1. **Class Weighting** (Rocinante): Balanced Softmax with τ=1.0 was the winner
   (0.5711 mean Dice across 15 configurations)
2. **Loss Optimization** (Shenron): Per-class Tversky (α=0.6, β=0.4) outperforms
   Dice+BCE and Focal for membrane/lumen imbalance
3. **Auto3DSeg** (Longleaf): Partial annotation masking is required for 14-class
   training — each crop only has annotations for a subset of classes

The core loss is BalancedSoftmaxPartialTverskyLoss:
- Per-class Tversky index (α=0.6 FP weight, β=0.4 FN weight) for asymmetric
  FP/FN trade-off (penalizes false positives more heavily)
- Balanced Softmax logit adjustment: logit_c -= τ * (log(n_c) - mean(log(n)))
  to counteract class frequency imbalance before computing loss
- Partial annotation masking: channels without annotation are excluded from loss
- Online class frequency estimation (updated every N batches) to adapt τ scaling

Author: CellMap Segmentation Challenge / L40S Experiment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class BalancedSoftmaxPartialTverskyLoss(nn.Module):
    """
    Balanced Softmax + Partial Annotation + Per-Class Tversky Loss.

    This is the experiment's unified loss function. It applies three techniques:

    1. **Partial Annotation Masking**: Each training crop only has ground truth
       for a subset of the 14 classes. Channels without annotation are detected
       (all-NaN or all-zero after binarization) and excluded from the loss.
       Call `set_annotation_mask(mask)` before each forward pass, OR pass
       targets with NaN for unannotated channels.

    2. **Balanced Softmax Logit Adjustment**: Before computing the per-class
       Tversky loss, logits are adjusted by the log-frequency of each class:
           adjusted_logit_c = logit_c - τ * (log(n_c) - mean(log(n)))
       where n_c is the number of annotated pixels for class c, and τ controls
       the strength of the adjustment. With τ=1.0 (the winner), rare classes
       get their logits boosted, common classes get suppressed.

    3. **Per-Class Tversky Index**: Computed per channel with α=0.6 (FP weight)
       and β=0.4 (FN weight). This penalizes false positives slightly more,
       which is beneficial for thin membrane structures.

    Args:
        classes: List of 14 class names
        alpha: Tversky FP weight (default 0.6 — penalizes FP more)
        beta: Tversky FN weight (default 0.4)
        tau: Balanced Softmax temperature (default 1.0 — winner config)
        smooth: Smoothing constant for numerical stability
        freq_update_interval: Update class frequencies every N batches
        min_annotated_pixels: Minimum annotated pixels to include channel in loss
    """

    def __init__(
        self,
        classes: List[str],
        alpha: float = 0.6,
        beta: float = 0.4,
        tau: float = 1.0,
        smooth: float = 1e-6,
        freq_update_interval: int = 50,
        min_annotated_pixels: int = 10,
    ):
        super().__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.smooth = smooth
        self.freq_update_interval = freq_update_interval
        self.min_annotated_pixels = min_annotated_pixels

        # Online class frequency tracking (registered as buffers for DDP sync)
        # These are running sums of (annotated_positive_pixels, total_annotated_pixels)
        self.register_buffer(
            "class_pos_counts",
            torch.ones(self.n_classes, dtype=torch.float64),
        )
        self.register_buffer(
            "class_total_counts",
            torch.ones(self.n_classes, dtype=torch.float64) * 100,
        )
        self.register_buffer(
            "log_freq_adjustment",
            torch.zeros(self.n_classes, dtype=torch.float32),
        )
        self.register_buffer("batch_counter", torch.tensor(0, dtype=torch.long))

        # Optional external annotation mask (set before forward if available)
        self._annotation_mask: Optional[torch.Tensor] = None

        print(f"BalancedSoftmaxPartialTverskyLoss initialized:")
        print(f"  Classes: {self.n_classes}")
        print(f"  Tversky α={alpha}, β={beta}")
        print(f"  Balanced Softmax τ={tau}")
        print(f"  Frequency update interval: {freq_update_interval} batches")

    # ------------------------------------------------------------------
    # Partial Annotation Interface
    # ------------------------------------------------------------------

    def set_annotation_mask(self, mask: Optional[torch.Tensor]) -> None:
        """
        Set the annotation mask for the current batch.

        Args:
            mask: Boolean tensor of shape (B, C) or (C,) where True means
                  the channel IS annotated in this crop. If None, annotation
                  presence will be inferred from NaN values in targets.
        """
        self._annotation_mask = mask

    # ------------------------------------------------------------------
    # Balanced Softmax Logit Adjustment
    # ------------------------------------------------------------------

    def _update_class_frequencies(
        self,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """
        Update running class frequency estimates from the current batch.

        Args:
            targets: Clean targets (NaN replaced with 0), shape (B, C, ...)
            valid_mask: Boolean mask of annotated pixels, shape (B, C, ...)
        """
        with torch.no_grad():
            for c in range(self.n_classes):
                channel_mask = valid_mask[:, c]
                if channel_mask.any():
                    n_total = channel_mask.sum().double()
                    n_pos = (targets[:, c] * channel_mask.float()).sum().double()
                    # Exponential moving average (decay=0.99)
                    self.class_total_counts[c] = (
                        0.99 * self.class_total_counts[c] + 0.01 * n_total
                    )
                    self.class_pos_counts[c] = (
                        0.99 * self.class_pos_counts[c] + 0.01 * n_pos.clamp(min=1)
                    )

    def _compute_logit_adjustment(self) -> torch.Tensor:
        """
        Compute the balanced softmax logit adjustment vector.

        Returns bias of shape (C,) to subtract from logits:
            bias_c = τ * (log(n_c) - mean(log(n)))

        Rare classes get negative bias (logits boosted), common classes
        get positive bias (logits suppressed).
        """
        # Class frequencies: fraction of positive pixels among annotated pixels
        freqs = (self.class_pos_counts / self.class_total_counts.clamp(min=1)).float()
        freqs = freqs.clamp(min=1e-8, max=1.0 - 1e-8)

        log_freqs = torch.log(freqs)
        mean_log_freq = log_freqs.mean()

        adjustment = self.tau * (log_freqs - mean_log_freq)
        return adjustment

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the unified loss.

        Args:
            pred: Model logits, shape (B, C, ...) where C = n_classes
            target: Ground truth with NaN for unannotated channels,
                    shape (B, C, ...)

        Returns:
            Scalar loss value
        """
        batch_size, n_classes = pred.shape[:2]
        assert n_classes == self.n_classes, (
            f"Expected {self.n_classes} channels, got {n_classes}"
        )

        # ----------------------------------------------------------
        # 1. Build valid mask (True where annotated)
        # ----------------------------------------------------------
        nan_mask = ~target.isnan()  # True where NOT NaN
        target_clean = target.nan_to_num(0.0)

        if self._annotation_mask is not None:
            # External mask: (B, C) or (C,) — expand to spatial dims
            ext_mask = self._annotation_mask
            if ext_mask.dim() == 1:
                ext_mask = ext_mask.unsqueeze(0).expand(batch_size, -1)
            # Expand to match spatial dimensions
            for _ in range(pred.dim() - 2):
                ext_mask = ext_mask.unsqueeze(-1)
            valid_mask = nan_mask & ext_mask.bool().to(pred.device)
            self._annotation_mask = None  # Reset after use
        else:
            valid_mask = nan_mask

        # ----------------------------------------------------------
        # 2. Update class frequencies periodically
        # ----------------------------------------------------------
        self.batch_counter += 1
        if self.batch_counter % self.freq_update_interval == 0:
            self._update_class_frequencies(target_clean, valid_mask)
            self.log_freq_adjustment = self._compute_logit_adjustment()

        # ----------------------------------------------------------
        # 3. Apply balanced softmax logit adjustment
        # ----------------------------------------------------------
        # Shape: (C,) -> (1, C, 1, 1) or (1, C, 1, 1, 1)
        adjustment = self.log_freq_adjustment.to(pred.device)
        adj_shape = [1, n_classes] + [1] * (pred.dim() - 2)
        adjustment = adjustment.view(*adj_shape)

        pred_adjusted = pred - adjustment  # Subtract: rare classes boosted

        # ----------------------------------------------------------
        # 4. Per-class Tversky loss with partial masking
        # ----------------------------------------------------------
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        n_valid_classes = 0

        for c in range(n_classes):
            # Extract channel
            pred_c = pred_adjusted[:, c : c + 1]
            target_c = target_clean[:, c : c + 1]
            mask_c = valid_mask[:, c : c + 1]

            # Skip channels with insufficient annotation
            n_annotated = mask_c.sum()
            if n_annotated < self.min_annotated_pixels:
                continue

            # Apply sigmoid to adjusted logits
            pred_prob = torch.sigmoid(pred_c)

            # Masked flatten
            pred_flat = (pred_prob * mask_c.float()).flatten()
            target_flat = (target_c * mask_c.float()).flatten()

            # Tversky components
            tp = (pred_flat * target_flat).sum()
            fp = (pred_flat * (1 - target_flat)).sum()
            fn = ((1 - pred_flat) * target_flat).sum()

            # Tversky index
            denom = tp + self.alpha * fp + self.beta * fn + self.smooth
            tversky = (tp + self.smooth) / denom.clamp(min=self.smooth)

            class_loss = 1.0 - tversky
            total_loss = total_loss + class_loss
            n_valid_classes += 1

        # Average across valid classes
        if n_valid_classes > 0:
            total_loss = total_loss / n_valid_classes
        else:
            # Fallback: return zero loss if no valid classes
            total_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

        return total_loss

    def get_class_frequencies(self) -> Dict[str, float]:
        """Return current estimated class frequencies (for logging)."""
        freqs = (self.class_pos_counts / self.class_total_counts.clamp(min=1)).float()
        return {
            name: freq.item()
            for name, freq in zip(self.classes, freqs)
        }

    def get_logit_adjustments(self) -> Dict[str, float]:
        """Return current logit adjustment values (for logging)."""
        return {
            name: adj.item()
            for name, adj in zip(self.classes, self.log_freq_adjustment)
        }


class DiceLoss(nn.Module):
    """
    NaN-safe Dice Loss for segmentation.

    Handles NaN values in targets by masking them out.
    Used as a standalone metric/loss and inside MetricsTracker.
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(predictions)

        mask = ~torch.isnan(targets)
        targets = targets.nan_to_num(0)

        probs = probs.flatten(2)
        targets = targets.flatten(2)
        mask = mask.flatten(2)

        intersection = ((probs * targets) * mask).sum(dim=2)
        union = ((probs + targets) * mask).sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "none":
            return dice_loss
        else:
            return dice_loss.sum()


class FocalLoss(nn.Module):
    """
    NaN-safe Focal Loss with optional per-class pos_weight.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(predictions)

        mask = ~torch.isnan(targets)
        targets = targets.nan_to_num(0)

        ce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction="none"
        )

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * ce_loss

        if self.pos_weight is not None:
            weight = targets * self.pos_weight + (1 - targets)
            loss = loss * weight

        loss = loss * mask

        if self.reduction == "mean":
            return loss.sum() / mask.sum().clamp(min=1)
        elif self.reduction == "none":
            return loss
        else:
            return loss.sum()


def get_loss_function(
    loss_type: str,
    classes: List[str],
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss by name.

    Args:
        loss_type: 'balanced_softmax_tversky' (default), 'dice', 'focal'
        classes: List of class names
        **kwargs: Forwarded to the loss constructor
    """
    loss_type = loss_type.lower().replace("-", "_").replace(" ", "_")

    if loss_type in ("balanced_softmax_tversky", "bst", "default"):
        return BalancedSoftmaxPartialTverskyLoss(classes=classes, **kwargs)
    elif loss_type == "dice":
        return DiceLoss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Options: balanced_softmax_tversky, dice, focal"
        )
