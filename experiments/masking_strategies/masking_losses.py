# -*- coding: utf-8 -*-
"""
Masking Strategy Losses for CellMap Segmentation

All strategies share the same underlying Tversky metric (alpha=0.6, beta=0.4)
and the same BalancedSoftmax logit adjustment (tau=1.0) -- the best config
from the class-weighting experiments (Dice=0.3171).

What varies is HOW unannotated (NaN) pixels are handled during training.
The baseline class-weighting experiment simply excluded them (mask_nan=True),
which caused a train/eval mismatch: model never penalised for FP on
unannotated regions during training, but evaluation counted them as FP.

Strategies implemented:
  0. no_mask          -- NaN -> 0, all pixels contribute (simple baseline)
  1. masksup          -- Siamese reconstruction of randomly masked pixels
  2. regional_weight  -- Grid-based adaptive per-region loss weighting
  3. uncertainty_eu   -- Epistemic uncertainty guided masking
  4. uncertainty_au   -- Aleatoric uncertainty guided masking
  5. box_class_mask   -- Per-class foreground bounding-box masking
  6. salient_mask     -- Structure-aware differential masking ratios
  7. entropy_mask     -- Dynamic entropy threshold masking
  8. class_presence   -- If class absent in image, mask entire image for that class

References:
- MaskSup: Fan et al., "Masked Supervised Learning for Semantic Seg." 2022
- Regional: Yao et al., "Regionally Adaptive Loss" / inspired by GridCut
- Uncertainty: Kendall & Gal, "What Uncertainties Do We Need?" NeurIPS 2017
- BCM: Song et al., "Box-Driven Class-Wise Masking" CVPR 2019
- Salient/Structure: He et al., MAE / adapted for segmentation
- Dynamic Entropy: Zou et al., "PseudoSeg" / Zheng et al., "Rectifying"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


# ================================================================
# 0.  Base Tversky with BalancedSoftmax (shared by ALL strategies)
# ================================================================

class BaseTverskyLoss(nn.Module):
    """Per-class Tversky loss with optional BalancedSoftmax logit adjustment.

    This is the FIXED underlying metric.  Masking strategies wrap this
    and provide a processed (pred, target, mask) to the per-class loop.

    Args:
        alpha:  FP weight (default 0.6)
        beta:   FN weight (default 0.4)
        smooth: Smoothing constant
        tau:    BalancedSoftmax temperature (0 = no adjustment)
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.4,
                 smooth: float = 1e-6, tau: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.tau = tau
        # Logit adjustment -- lazily sized on first forward
        self._logit_adj = None
        self._accum_counts = None
        self._online = True
        self._update_interval = 50
        self._batch_counter = 0
        self._n_classes = None

    def _init_buffers(self, n_classes: int, device):
        """Lazily initialise per-class buffers on first forward."""
        if self._n_classes is not None:
            return
        self._n_classes = n_classes
        self._logit_adj = torch.zeros(n_classes, device=device)
        self._accum_counts = torch.zeros(n_classes, dtype=torch.float64,
                                         device=device)

    def _accumulate(self, target: torch.Tensor):
        """Online class-frequency estimation for logit adjustment."""
        self._init_buffers(target.shape[1], target.device)
        for c in range(target.shape[1]):
            valid = ~target[:, c].isnan()
            positives = (target[:, c].nan_to_num(0) * valid).sum()
            self._accum_counts[c] += positives.double()
        self._batch_counter += 1
        if self._batch_counter % self._update_interval == 0:
            log_counts = []
            for i in range(target.shape[1]):
                n = max(self._accum_counts[i].item(), 1.0)
                log_counts.append(math.log(n))
            t = torch.tensor(log_counts, dtype=torch.float32,
                             device=target.device)
            t = t - t.mean()
            self._logit_adj = self.tau * t

    def per_class_tversky(self, pred: torch.Tensor, target: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute per-class Tversky with an explicit spatial mask.

        Args:
            pred:   (B, C, H, W) logits
            target: (B, C, H, W) clean targets (no NaN -- already handled)
            mask:   (B, C, H, W) or (B, 1, H, W) float mask, 1 = include
                    If None, all pixels included.

        Returns:
            (C,) tensor of 1 - Tversky per class
        """
        n_classes = pred.shape[1]
        self._init_buffers(n_classes, pred.device)
        losses = []

        adj = self._logit_adj if (self.tau > 0 and self._logit_adj is not None) else None

        for c in range(n_classes):
            pred_c = pred[:, c]
            target_c = target[:, c]

            if adj is not None and adj.numel() == n_classes:
                pred_c = pred_c - adj[c]

            pred_sig = torch.sigmoid(pred_c)

            if mask is not None:
                if mask.shape[1] == 1:
                    m = mask[:, 0]  # broadcast single-channel mask
                else:
                    m = mask[:, c]
                if m.sum() == 0:
                    losses.append(torch.tensor(0.0, device=pred.device))
                    continue
                p = (pred_sig * m).flatten()
                t = (target_c * m).flatten()
            else:
                p = pred_sig.flatten()
                t = target_c.flatten()

            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()

            denom = tp + self.alpha * fp + self.beta * fn + self.smooth
            tversky = (tp + self.smooth) / denom.clamp(min=self.smooth)
            losses.append(1 - tversky)

        return torch.stack(losses)


# ================================================================
# Strategy 0: No Masking
# ================================================================

class NoMaskTverskyLoss(BaseTverskyLoss):
    """NaN pixels treated as background (0).  All pixels contribute.

    This is the simplest fix for the train/eval mismatch: the model
    gets penalised for false positives on unannotated regions.
    """

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if self._online and self.training:
            self._accumulate(target)
        target_clean = target.nan_to_num(0)
        losses = self.per_class_tversky(pred, target_clean, mask=None)
        return losses.mean()


# ================================================================
# Strategy 1: MaskSup (Masked Supervised Learning)
# ================================================================

class MaskSupTverskyLoss(BaseTverskyLoss):
    """Siamese-style masked supervision.

    During training, randomly masks a fraction of ANNOTATED pixels and
    adds a reconstruction loss that forces the model to predict those
    pixels from surrounding context.  This encourages short-range
    contextual learning and reduces FP on ambiguous regions.

    Loss = standard_tversky(visible) + lambda * reconstruction_tversky(masked)

    Args:
        mask_ratio:     Fraction of annotated pixels to mask (default 0.3)
        recon_weight:   Weight for the reconstruction branch (default 0.5)
    """

    def __init__(self, mask_ratio: float = 0.3, recon_weight: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
        self.recon_weight = recon_weight

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if self._online and self.training:
            self._accumulate(target)

        # Build annotation mask
        annotated = ~target.isnan()                     # (B, C, H, W)
        target_clean = target.nan_to_num(0)

        if self.training:
            # Randomly mask a fraction of annotated pixels
            rand = torch.rand_like(target_clean)
            recon_mask = annotated & (rand < self.mask_ratio)   # pixels to reconstruct
            visible_mask = annotated & ~recon_mask               # pixels for standard loss

            # Standard loss on visible annotated pixels
            loss_visible = self.per_class_tversky(
                pred, target_clean, visible_mask.float()).mean()

            # Reconstruction loss: model must predict masked annotated pixels
            if recon_mask.any():
                loss_recon = self.per_class_tversky(
                    pred, target_clean, recon_mask.float()).mean()
            else:
                loss_recon = torch.tensor(0.0, device=pred.device)

            return loss_visible + self.recon_weight * loss_recon
        else:
            # Eval: use all annotated pixels
            return self.per_class_tversky(
                pred, target_clean, annotated.float()).mean()


# ================================================================
# Strategy 2: Regionally Adaptive Weighting
# ================================================================

class RegionalAdaptiveTverskyLoss(BaseTverskyLoss):
    """Divide image into grid and weight sub-regions by difficulty.

    Computes per-region Tversky loss on a grid, then weights each region
    inversely by its running average performance.  Hard regions (rare-class
    boundaries, high FP areas) automatically get higher weight.

    Args:
        grid_size:  Grid divisions along H and W (default 8 for 256x256)
        momentum:   EMA momentum for difficulty tracking (default 0.9)
    """

    def __init__(self, grid_size: int = 8, momentum: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.momentum = momentum
        # Will lazily init the difficulty tracker
        self._difficulty = None

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if self._online and self.training:
            self._accumulate(target)

        B, C, H, W = pred.shape
        G = self.grid_size
        gH, gW = H // G, W // G

        annotated = ~target.isnan()
        target_clean = target.nan_to_num(0)

        # Lazily init difficulty grid
        if self._difficulty is None:
            self._difficulty = torch.ones(G, G, device=pred.device)

        # Snapshot weights BEFORE the loop so inplace updates don't
        # invalidate the autograd graph ("modified by inplace op" error).
        weights_snapshot = self._difficulty.detach().clone()

        total_loss = torch.tensor(0.0, device=pred.device)
        region_count = 0
        region_losses = {}  # (gi, gj) -> detached loss for EMA

        for gi in range(G):
            for gj in range(G):
                h_start, h_end = gi * gH, (gi + 1) * gH
                w_start, w_end = gj * gW, (gj + 1) * gW

                pred_r = pred[:, :, h_start:h_end, w_start:w_end]
                tgt_r = target_clean[:, :, h_start:h_end, w_start:w_end]
                ann_r = annotated[:, :, h_start:h_end, w_start:w_end]

                if ann_r.sum() == 0:
                    continue

                # Per-class Tversky in this region
                region_loss = self.per_class_tversky(
                    pred_r, tgt_r, ann_r.float()).mean()

                # Weight by frozen snapshot (no inplace conflict)
                weight = weights_snapshot[gi, gj]
                total_loss = total_loss + weight * region_loss
                region_count += 1

                # Save for deferred EMA update
                if self.training:
                    region_losses[(gi, gj)] = region_loss.detach()

        # Deferred EMA update AFTER forward graph is complete
        if self.training and region_losses:
            with torch.no_grad():
                for (gi, gj), rl in region_losses.items():
                    self._difficulty[gi, gj] = (
                        self.momentum * self._difficulty[gi, gj] +
                        (1 - self.momentum) * rl
                    )

        if region_count > 0:
            # Normalise difficulty weights so they sum to G*G
            with torch.no_grad():
                self._difficulty.div_(
                    self._difficulty.mean().clamp(min=1e-6)
                )
            total_loss = total_loss / region_count

        return total_loss


# ================================================================
# Strategy 3: Epistemic Uncertainty Masking
# ================================================================

class EpistemicUncertaintyTverskyLoss(BaseTverskyLoss):
    """Focus on high-uncertainty regions using MC-Dropout.

    During training, runs N_mc stochastic forward passes (with dropout)
    to estimate per-pixel epistemic uncertainty (variance of predictions).
    High-EU pixels are UP-WEIGHTED to force the model to refine its
    representation of challenging regions.

    NOTE: Requires the model to have dropout layers.  If no dropout,
    this degrades to standard loss with slight overhead.

    Args:
        n_mc:           Number of MC forward passes for uncertainty (def 4)
        uncertainty_weight: How much to up-weight high-EU pixels (def 2.0)
        warmup_epochs:  Skip MC uncertainty for first N epochs (def 5)
    """

    def __init__(self, n_mc: int = 4, uncertainty_weight: float = 2.0,
                 warmup_epochs: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.n_mc = n_mc
        self.uncertainty_weight = uncertainty_weight
        self.warmup_epochs = warmup_epochs
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _estimate_uncertainty(self, model: nn.Module,
                              inputs: torch.Tensor) -> torch.Tensor:
        """Run MC-Dropout passes and return per-pixel variance."""
        model.train()  # Enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(self.n_mc):
                out = model(inputs)
                preds.append(torch.sigmoid(out))

        stacked = torch.stack(preds)          # (N_mc, B, C, H, W)
        variance = stacked.var(dim=0)         # (B, C, H, W)
        return variance

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                uncertainty_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred:   (B, C, H, W) logits
            target: (B, C, H, W) with NaN
            uncertainty_map: (B, C, H, W) pre-computed uncertainty.
                             If None, uses uniform weighting.
        """
        if self._online and self.training:
            self._accumulate(target)

        annotated = ~target.isnan()
        target_clean = target.nan_to_num(0)

        if uncertainty_map is not None and self._epoch >= self.warmup_epochs:
            # Up-weight high-uncertainty annotated pixels
            # uncertainty_map: higher = more uncertain
            norm_unc = uncertainty_map / (uncertainty_map.max() + 1e-8)
            weight_map = 1.0 + (self.uncertainty_weight - 1.0) * norm_unc
            mask = annotated.float() * weight_map
        else:
            mask = annotated.float()

        return self.per_class_tversky(pred, target_clean, mask).mean()


# ================================================================
# Strategy 4: Aleatoric Uncertainty Masking
# ================================================================

class AleatoricUncertaintyTverskyLoss(BaseTverskyLoss):
    """Exclude noisy/ambiguous pixels using aleatoric uncertainty.

    Learns a per-pixel log-variance alongside predictions.  Pixels with
    high aleatoric uncertainty (noise, labelling errors) are DOWN-WEIGHTED,
    preventing the model from fitting to noisy annotations that cause
    low-precision blob predictions.

    The model should output 2*C channels: C logits + C log-variance.

    Args:
        au_threshold:  Percentile of AU above which pixels are excluded
                       (default 0.9 = exclude top 10% noisiest)
    """

    def __init__(self, au_threshold: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.au_threshold = au_threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                log_variance: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred:         (B, C, H, W) logits (standard C channels)
            target:       (B, C, H, W) with NaN
            log_variance: (B, C, H, W) learned log-variance.  If None,
                          falls back to standard annotated masking.
        """
        if self._online and self.training:
            self._accumulate(target)

        annotated = ~target.isnan()
        target_clean = target.nan_to_num(0)

        if log_variance is not None and self.training:
            # Convert to variance, compute per-sample threshold
            variance = torch.exp(log_variance)
            # Per-batch percentile threshold
            flat_var = variance[annotated].flatten()
            if flat_var.numel() > 0:
                threshold = torch.quantile(flat_var, self.au_threshold)
                # Exclude high-variance pixels
                reliable = (variance <= threshold).float()
                mask = annotated.float() * reliable
            else:
                mask = annotated.float()

            # Tversky loss
            tversky_loss = self.per_class_tversky(pred, target_clean, mask).mean()

            # Regularise log-variance to prevent collapse (all excluded)
            reg = log_variance[annotated].mean() * 0.01
            return tversky_loss + reg
        else:
            mask = annotated.float()
            return self.per_class_tversky(pred, target_clean, mask).mean()


# ================================================================
# Strategy 5: Box-Driven Class-Wise Masking (BCM)
# ================================================================

class BoxClassMaskTverskyLoss(BaseTverskyLoss):
    """Per-class spatial masking using bounding boxes of annotated regions.

    For each class, computes the bounding box of all annotated positive
    pixels and creates a soft spatial mask that restricts loss computation
    to within those bounds (with padding).  This prevents the model from
    being penalised for predictions far from any annotated foreground.

    Args:
        pad_fraction: Fraction of bbox size to pad (default 0.15)
        bg_weight:    Weight for pixels outside all class bboxes (def 0.1)
    """

    def __init__(self, pad_fraction: float = 0.15, bg_weight: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.pad_fraction = pad_fraction
        self.bg_weight = bg_weight

    def _compute_class_bbox_mask(self, target_c: torch.Tensor,
                                 annotated_c: torch.Tensor,
                                 H: int, W: int) -> torch.Tensor:
        """Compute padded bounding box mask for one class across batch.

        Returns: (B, H, W) float mask
        """
        B = target_c.shape[0]
        mask = torch.full((B, H, W), self.bg_weight,
                          device=target_c.device)

        for b in range(B):
            # Find annotated positive pixels
            pos = (target_c[b] > 0.5) & annotated_c[b]
            if not pos.any():
                # No positive pixels: let bg_weight apply everywhere
                continue

            # Bounding box
            ys, xs = torch.where(pos)
            y_min, y_max = ys.min().item(), ys.max().item()
            x_min, x_max = xs.min().item(), xs.max().item()

            # Pad
            pad_h = int((y_max - y_min + 1) * self.pad_fraction)
            pad_w = int((x_max - x_min + 1) * self.pad_fraction)
            y_min = max(0, y_min - pad_h)
            y_max = min(H - 1, y_max + pad_h)
            x_min = max(0, x_min - pad_w)
            x_max = min(W - 1, x_max + pad_w)

            mask[b, y_min:y_max + 1, x_min:x_max + 1] = 1.0

        return mask

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if self._online and self.training:
            self._accumulate(target)

        B, C, H, W = pred.shape
        annotated = ~target.isnan()
        target_clean = target.nan_to_num(0)

        # Build per-class bbox mask
        per_class_mask = []
        for c in range(C):
            bbox_m = self._compute_class_bbox_mask(
                target_clean[:, c], annotated[:, c], H, W)
            # Combine with annotation mask: annotated pixels inside bbox = 1.0,
            # annotated outside bbox = bg_weight, unannotated = 0
            combined = annotated[:, c].float() * bbox_m
            per_class_mask.append(combined)

        mask = torch.stack(per_class_mask, dim=1)  # (B, C, H, W)
        return self.per_class_tversky(pred, target_clean, mask).mean()


# ================================================================
# Strategy 6: Structure-Aware / Salient Masking
# ================================================================

class SalientMaskTverskyLoss(BaseTverskyLoss):
    """Differential masking ratios for foreground vs background.

    Uses a lower random masking ratio for pixels near rare-class
    structures and a higher ratio for background, preserving critical
    structural features while challenging the model on easy regions.

    Args:
        fg_mask_ratio:  Masking ratio for foreground/rare pixels (def 0.15)
        bg_mask_ratio:  Masking ratio for background pixels (def 0.5)
        rare_classes:   Indices of classes considered "rare" (def all)
    """

    def __init__(self, fg_mask_ratio: float = 0.15,
                 bg_mask_ratio: float = 0.5,
                 rare_classes: Optional[List[int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.fg_mask_ratio = fg_mask_ratio
        self.bg_mask_ratio = bg_mask_ratio
        self.rare_classes = rare_classes  # None = all classes are "rare"

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if self._online and self.training:
            self._accumulate(target)

        B, C, H, W = pred.shape
        annotated = ~target.isnan()
        target_clean = target.nan_to_num(0)

        if self.training:
            # Identify foreground pixels (any positive class)
            if self.rare_classes is not None:
                rare_idx = self.rare_classes
            else:
                rare_idx = list(range(C))

            is_fg = torch.zeros(B, H, W, device=pred.device, dtype=torch.bool)
            for c in rare_idx:
                is_fg |= (target_clean[:, c] > 0.5) & annotated[:, c]

            # Different masking ratios
            rand = torch.rand(B, H, W, device=pred.device)
            fg_keep = rand >= self.fg_mask_ratio     # keep MORE fg pixels
            bg_keep = rand >= self.bg_mask_ratio     # keep FEWER bg pixels

            keep = torch.where(is_fg, fg_keep, bg_keep)   # (B, H, W)
            keep = keep.unsqueeze(1).expand_as(target_clean)  # (B, C, H, W)

            mask = annotated.float() * keep.float()
        else:
            mask = annotated.float()

        return self.per_class_tversky(pred, target_clean, mask).mean()


# ================================================================
# Strategy 7: Dynamic Entropy Masking
# ================================================================

class DynamicEntropyTverskyLoss(BaseTverskyLoss):
    """Entropy-based dynamic thresholding.

    Computes per-pixel prediction entropy.  Pixels with:
      - Very LOW entropy (confident) -> normal weight (model already knows)
      - MODERATE entropy (uncertain) -> UP-weighted (most informative)
      - Very HIGH entropy (confused) -> excluded (unreliable signal)

    The entropy thresholds evolve with an EMA during training.

    Args:
        high_entropy_percentile:  Above this percentile, exclude (def 0.9)
        mid_entropy_boost:        Up-weight factor for moderate entropy (def 1.5)
        momentum:                 EMA momentum for threshold (def 0.95)
    """

    def __init__(self, high_entropy_percentile: float = 0.9,
                 mid_entropy_boost: float = 1.5,
                 momentum: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.high_pct = high_entropy_percentile
        self.mid_boost = mid_entropy_boost
        self.momentum = momentum
        self.register_buffer('_ema_high_threshold', torch.tensor(0.5))
        self.register_buffer('_ema_low_threshold', torch.tensor(0.1))

    def _pixel_entropy(self, pred: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy based per-pixel entropy.  (B, C, H, W)"""
        p = torch.sigmoid(pred)
        p = p.clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        return entropy  # max = log(2) ~ 0.693

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if self._online and self.training:
            self._accumulate(target)

        annotated = ~target.isnan()
        target_clean = target.nan_to_num(0)

        if self.training:
            with torch.no_grad():
                entropy = self._pixel_entropy(pred.detach())
                # Mean entropy per pixel across classes
                mean_entropy = entropy.mean(dim=1, keepdim=True)  # (B,1,H,W)
                mean_entropy = mean_entropy.expand_as(pred)

                # Compute adaptive thresholds from annotated pixels
                ann_entropy = mean_entropy[annotated[:, :1].expand_as(annotated)]
                if ann_entropy.numel() > 0:
                    high_t = torch.quantile(ann_entropy, self.high_pct)
                    low_t = torch.quantile(ann_entropy, 0.3)

                    # EMA update (use .data to avoid graph interference)
                    self._ema_high_threshold.data.copy_(
                        self.momentum * self._ema_high_threshold.data +
                        (1 - self.momentum) * high_t
                    )
                    self._ema_low_threshold.data.copy_(
                        self.momentum * self._ema_low_threshold.data +
                        (1 - self.momentum) * low_t
                    )

                # Build weight map
                weight_map = torch.ones_like(pred)

                # Up-weight moderate entropy
                mid_mask = ((mean_entropy >= self._ema_low_threshold) &
                            (mean_entropy < self._ema_high_threshold))
                weight_map[mid_mask] = self.mid_boost

                # Exclude very high entropy
                high_mask = mean_entropy >= self._ema_high_threshold
                weight_map[high_mask] = 0.0

            mask = annotated.float() * weight_map
        else:
            mask = annotated.float()

        return self.per_class_tversky(pred, target_clean, mask).mean()


# ================================================================
# Strategy 8: Class-Presence Masking
# ================================================================

class ClassPresenceTverskyLoss(BaseTverskyLoss):
    """If a class has NO positive pixels in the image, mask the entire
    image for that class channel.  If the class IS present, include ALL
    pixels (annotated AND unannotated) so the model gets penalised for
    FP on unannotated regions.

    Rationale: If the class genuinely isn't in the image, we don't want
    to penalise the model for low-confidence activations that would be
    FP.  But if the class IS present, unannotated pixels are likely
    true-negative, so FP on them should be penalised.

    Args:
        presence_threshold:  Min fraction of positive annotated pixels to
                             consider the class "present" (default 0.001)
    """

    def __init__(self, presence_threshold: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.presence_threshold = presence_threshold

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if self._online and self.training:
            self._accumulate(target)

        B, C, H, W = pred.shape
        annotated = ~target.isnan()
        target_clean = target.nan_to_num(0)
        n_pixels = H * W

        # Per-class, per-sample: is the class present?
        mask = torch.zeros_like(target_clean)

        for c in range(C):
            for b in range(B):
                ann_c = annotated[b, c]
                tgt_c = target_clean[b, c]

                # Count positive annotated pixels
                n_positive = (tgt_c * ann_c.float()).sum().item()
                fraction = n_positive / max(ann_c.sum().item(), 1)

                if fraction >= self.presence_threshold:
                    # Class IS present -> use ALL pixels (no masking)
                    mask[b, c] = 1.0
                else:
                    # Class NOT present -> mask entire image for this class
                    mask[b, c] = 0.0

        return self.per_class_tversky(pred, target_clean, mask).mean()


# ================================================================
# Factory
# ================================================================

STRATEGY_REGISTRY = {
    'no_mask':          NoMaskTverskyLoss,
    'masksup':          MaskSupTverskyLoss,
    'regional_weight':  RegionalAdaptiveTverskyLoss,
    'uncertainty_eu':   EpistemicUncertaintyTverskyLoss,
    'uncertainty_au':   AleatoricUncertaintyTverskyLoss,
    'box_class_mask':   BoxClassMaskTverskyLoss,
    'salient_mask':     SalientMaskTverskyLoss,
    'entropy_mask':     DynamicEntropyTverskyLoss,
    'class_presence':   ClassPresenceTverskyLoss,
}


def get_masking_loss(strategy: str, **kwargs) -> nn.Module:
    """Create a masking strategy loss by name.

    All strategies use the same underlying BalancedSoftmax Tversky
    (alpha=0.6, beta=0.4, tau=1.0).  The strategy name selects HOW
    unannotated pixels are handled.

    Args:
        strategy:  One of STRATEGY_REGISTRY keys.
        **kwargs:  Strategy-specific params + Tversky overrides
                   (alpha, beta, smooth, tau).

    Returns:
        nn.Module  -- criterion(pred, target) -> scalar loss
    """
    s = strategy.lower().replace('-', '_')
    if s not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown masking strategy '{strategy}'. Choose from: "
            f"{', '.join(STRATEGY_REGISTRY.keys())}"
        )

    # Extract base Tversky params
    alpha = kwargs.pop('alpha', 0.6)
    beta = kwargs.pop('beta', 0.4)
    smooth = kwargs.pop('smooth', 1e-6)
    tau = kwargs.pop('tau', 1.0)

    cls = STRATEGY_REGISTRY[s]
    return cls(alpha=alpha, beta=beta, smooth=smooth, tau=tau, **kwargs)
