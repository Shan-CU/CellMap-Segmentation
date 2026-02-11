"""
Class Weighting Loss Functions for CellMap Segmentation

Implements three principled class-weighting strategies on top of
**per-class Tversky loss (α=0.6, β=0.4)** — the best-performing loss
function from the Shenron loss comparison experiments.

1. **Class-Balanced (CB) Loss** — Uses "Effective Number of Samples" to
   dampen the weight explosion that inverse-frequency weighting causes on
   ultra-rare classes in correlated 3-D EM data.

2. **Balanced Softmax / Logit Adjustment** — Shifts decision boundaries by
   injecting the class-frequency prior directly into the logits, so that
   rare organelles are not overwhelmed by majority classes.

3. **Seesaw Loss** — Dynamically mitigates the over-punishment of rare
   classes caused by negative gradients from frequent classes, while
   compensating for false-positive leakage into rare classes.

All losses:
- Use per-class Tversky (α=0.6, β=0.4) as the underlying metric loss.
- Handle NaN targets via masking (standard in this codebase).
- Are drop-in replacements: criterion(pred_logits, target) → scalar loss.
- Can be instantiated through `get_weighting_loss()` factory.

References:
- CB Loss: Cui et al., "Class-Balanced Loss Based on Effective Number of
  Samples", CVPR 2019
- Balanced Softmax: Ren et al., "Balanced Meta-Softmax for Long-Tailed
  Visual Recognition", NeurIPS 2020
- Logit Adjustment: Menon et al., "Long-tail learning via logit adjustment",
  ICLR 2021
- Seesaw Loss: Wang et al., "Seesaw Loss for Long-Tailed Instance
  Segmentation", CVPR 2021
- Tversky Loss: Salehi et al., "Tversky loss function for image
  segmentation using 3D fully convolutional deep networks", 2017
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


# ================================================================
# Helper: NaN-safe per-class Tversky base (shared by all strategies)
# ================================================================

class _PerClassTverskyBase(nn.Module):
    """Per-class Tversky loss base with NaN masking.

    All three weighting strategies build on top of this base.
    The base computes per-class Tversky losses and returns them as a
    tensor of shape (C,) so the weighting layer can apply its own
    per-class scaling.

    Args:
        alpha: FP weight (higher → penalise false positives more →
               higher precision).  Default 0.6 per Shenron results.
        beta:  FN weight (higher → penalise false negatives more →
               higher recall).  Default 0.4.
        smooth: Smoothing factor to prevent division by zero.
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.4,
                 smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def per_class_tversky(self, pred: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        """Compute per-class Tversky loss.

        Args:
            pred:   (B, C, H, W) logits
            target: (B, C, H, W) binary targets (may contain NaN)

        Returns:
            tversky_per_class: (C,) tensor of 1 − Tversky index per class
        """
        n_classes = pred.shape[1]
        losses = []

        for c in range(n_classes):
            pred_c = pred[:, c]        # (B, H, W)
            target_c = target[:, c]    # (B, H, W)

            valid = ~target_c.isnan()
            target_clean = target_c.nan_to_num(0)

            if valid.sum() == 0:
                losses.append(torch.tensor(0.0, device=pred.device))
                continue

            pred_sig = torch.sigmoid(pred_c)
            p = (pred_sig * valid).flatten()
            t = (target_clean * valid).flatten()

            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()

            denom = tp + self.alpha * fp + self.beta * fn + self.smooth
            tversky = (tp + self.smooth) / denom.clamp(min=self.smooth)
            losses.append(1 - tversky)

        return torch.stack(losses)


# ================================================================
# Simple per-class weighted Tversky (static weights from config)
# ================================================================

class PerClassWeightedTversky(_PerClassTverskyBase):
    """Per-class Tversky with static class weights.

    This is the simplest form: multiply each class's Tversky loss by a
    fixed weight and average.  Used for the uniform, manual,
    inverse-frequency, sqrt-inverse, log-inverse, and effective-number
    weight strategies defined in the config.

    Args:
        classes:       Ordered list of class names.
        class_weights: Dict mapping class name → weight multiplier.
        alpha, beta, smooth: Tversky parameters.
    """

    def __init__(
        self,
        classes: List[str],
        class_weights: Optional[Dict[str, float]] = None,
        alpha: float = 0.6,
        beta: float = 0.4,
        smooth: float = 1e-6,
    ):
        super().__init__(alpha=alpha, beta=beta, smooth=smooth)
        self.classes = classes

        weights = class_weights or {c: 1.0 for c in classes}
        weight_list = [weights.get(c, 1.0) for c in classes]
        self.register_buffer(
            'weights', torch.tensor(weight_list, dtype=torch.float32)
        )

        print(f"PerClassWeightedTversky (α={alpha}, β={beta}):")
        for c, w in zip(classes, weight_list):
            print(f"  {c}: weight={w:.3f}")

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        per_class = self.per_class_tversky(pred, target)  # (C,)
        weighted = per_class * self.weights.to(per_class.device)
        return weighted.mean()


# ================================================================
# 1. Class-Balanced (CB) Tversky Loss
# ================================================================

class ClassBalancedTverskyLoss(_PerClassTverskyBase):
    r"""Class-Balanced Tversky Loss using the Effective Number of Samples.

    Replaces raw voxel counts with *effective* counts:

    .. math::
        E_{n_c} = \frac{1 - \beta_{\text{cb}}^{n_c}}{1 - \beta_{\text{cb}}}

    and derives per-class weights :math:`w_c = 1 / E_{n_c}`, normalised
    so :math:`\sum w_c = C`.  These weights multiply the per-class
    Tversky loss.

    Supports both pre-computed voxel counts and online estimation.

    Args:
        classes:            Ordered list of class names.
        class_voxel_counts: Dict mapping class name → voxel count.
                            If *None*, counts are estimated online.
        beta_cb:            CB hyperparameter in [0, 1).  Typical values:
                            0.99 (mild), 0.999 (medium), 0.9999 (strong).
        alpha, beta, smooth: Tversky parameters.
    """

    BETA_PRESETS = {
        'mild':   0.99,
        'medium': 0.999,
        'strong': 0.9999,
    }

    def __init__(
        self,
        classes: List[str],
        class_voxel_counts: Optional[Dict[str, float]] = None,
        beta_cb: float = 0.9999,
        alpha: float = 0.6,
        beta: float = 0.4,
        smooth: float = 1e-6,
    ):
        super().__init__(alpha=alpha, beta=beta, smooth=smooth)
        self.classes = classes
        self.beta_cb = beta_cb

        if class_voxel_counts is not None:
            weights = self._compute_weights(classes, class_voxel_counts,
                                            beta_cb)
            self.register_buffer('weights', weights)
            self._online = False
        else:
            self.register_buffer('weights', torch.ones(len(classes)))
            self.register_buffer(
                '_accum_counts',
                torch.zeros(len(classes), dtype=torch.float64))
            self._online = True
            self._update_interval = 50
            self._batch_counter = 0

        print(f"ClassBalancedTverskyLoss (β_cb={beta_cb}, α={alpha}, β={beta}):")
        for c, w in zip(classes, self.weights.tolist()):
            print(f"  {c}: weight={w:.4f}")

    @staticmethod
    def _compute_weights(classes, counts, beta_cb):
        raw = []
        for c in classes:
            n = counts.get(c, 1.0)
            effective = (1.0 - beta_cb ** n) / (1.0 - beta_cb)
            raw.append(1.0 / max(effective, 1e-12))
        w = torch.tensor(raw, dtype=torch.float32)
        w = w / w.sum() * len(classes)   # normalise → sum = C
        return w

    def _accumulate(self, target: torch.Tensor):
        for c in range(target.shape[1]):
            valid = ~target[:, c].isnan()
            positives = (target[:, c].nan_to_num(0) * valid).sum()
            self._accum_counts[c] += positives.double()
        self._batch_counter += 1
        if self._batch_counter % self._update_interval == 0:
            counts_dict = {
                cls: max(self._accum_counts[i].item(), 1.0)
                for i, cls in enumerate(self.classes)
            }
            self.weights.copy_(
                self._compute_weights(self.classes, counts_dict, self.beta_cb)
            )

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if self._online and self.training:
            self._accumulate(target)
        per_class = self.per_class_tversky(pred, target)
        weighted = per_class * self.weights.to(per_class.device)
        return weighted.mean()


# ================================================================
# 2. Balanced Softmax / Logit Adjustment + Tversky
# ================================================================

class BalancedSoftmaxTverskyLoss(_PerClassTverskyBase):
    r"""Logit-adjusted Tversky loss.

    Shifts the logits by a class-frequency prior *before* applying the
    sigmoid in the Tversky computation:

    .. math::
        \hat{f}_c = f_c - \tau \cdot (\log n_c - \overline{\log n})

    Rare classes get a positive offset → sigmoid pushes towards 1 →
    model needs less evidence to predict the rare class.

    The adjustment is applied inside the Tversky loss itself (unlike the
    Dice+BCE version where only the BCE was adjusted).  This makes the
    Tversky metric directly class-aware.

    Args:
        classes:            Ordered list of class names.
        class_voxel_counts: Dict → voxel counts.  If *None*, online.
        tau:                Temperature (default 1.0).
        alpha, beta, smooth: Tversky parameters.
    """

    def __init__(
        self,
        classes: List[str],
        class_voxel_counts: Optional[Dict[str, float]] = None,
        tau: float = 1.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        smooth: float = 1e-6,
    ):
        super().__init__(alpha=alpha, beta=beta, smooth=smooth)
        self.classes = classes
        self.tau = tau

        if class_voxel_counts is not None:
            adj = self._compute_adjustments(classes, class_voxel_counts, tau)
            self.register_buffer('logit_adj', adj)
            self._online = False
        else:
            self.register_buffer('logit_adj', torch.zeros(len(classes)))
            self.register_buffer(
                '_accum_counts',
                torch.zeros(len(classes), dtype=torch.float64))
            self._online = True
            self._update_interval = 50
            self._batch_counter = 0

        print(f"BalancedSoftmaxTverskyLoss (τ={tau}, α={alpha}, β={beta}):")
        for c, a in zip(classes, self.logit_adj.tolist()):
            print(f"  {c}: logit_adj={a:+.4f}")

    @staticmethod
    def _compute_adjustments(classes, counts, tau):
        log_counts = []
        for c in classes:
            n = max(counts.get(c, 1.0), 1.0)
            log_counts.append(math.log(n))
        t = torch.tensor(log_counts, dtype=torch.float32)
        t = t - t.mean()       # centre → mean adjustment = 0
        return tau * t

    def _accumulate(self, target: torch.Tensor):
        for c in range(target.shape[1]):
            valid = ~target[:, c].isnan()
            positives = (target[:, c].nan_to_num(0) * valid).sum()
            self._accum_counts[c] += positives.double()
        self._batch_counter += 1
        if self._batch_counter % self._update_interval == 0:
            counts_dict = {
                cls: max(self._accum_counts[i].item(), 1.0)
                for i, cls in enumerate(self.classes)
            }
            self.logit_adj.copy_(
                self._compute_adjustments(self.classes, counts_dict, self.tau)
            )

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if self._online and self.training:
            self._accumulate(target)

        n_classes = pred.shape[1]
        adj = self.logit_adj.to(pred.device)
        losses = []

        for c in range(n_classes):
            pred_c = pred[:, c]
            target_c = target[:, c]
            valid = ~target_c.isnan()
            target_clean = target_c.nan_to_num(0)

            if valid.sum() == 0:
                losses.append(torch.tensor(0.0, device=pred.device))
                continue

            # Adjusted logits: subtract centred log-prior
            # Rare class → adj[c] < 0 → subtract neg = add → logit ↑ → sigmoid ↑
            adjusted = pred_c - adj[c]
            pred_sig = torch.sigmoid(adjusted)

            p = (pred_sig * valid).flatten()
            t = (target_clean * valid).flatten()

            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()

            denom = tp + self.alpha * fp + self.beta * fn + self.smooth
            tversky = (tp + self.smooth) / denom.clamp(min=self.smooth)
            losses.append(1 - tversky)

        return torch.stack(losses).mean()


# ================================================================
# 3. Seesaw + Tversky Loss
# ================================================================

class SeesawTverskyLoss(_PerClassTverskyBase):
    r"""Seesaw-weighted per-class Tversky loss.

    Applies two dynamic factors to the per-class Tversky losses:

    **Mitigation factor** — Reduces loss weight for frequent classes when
    training on rare-class samples, preventing the "over-punishment"
    problem where negative gradients from majority classes erase the
    learning signal for minority classes.

    **Compensation factor** — Increases the penalty when the model
    produces excessive false positives for a particular class (based on
    cumulative FP / total-error ratio).

    Args:
        classes:            Ordered list of class names.
        class_voxel_counts: Dict → voxel counts.  If *None*, online.
        p_mitigation:       Mitigation exponent (default 0.8).
        q_compensation:     Compensation exponent (default 2.0).
        alpha, beta, smooth: Tversky parameters.
    """

    def __init__(
        self,
        classes: List[str],
        class_voxel_counts: Optional[Dict[str, float]] = None,
        p_mitigation: float = 0.8,
        q_compensation: float = 2.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        smooth: float = 1e-6,
    ):
        super().__init__(alpha=alpha, beta=beta, smooth=smooth)
        self.classes = classes
        self.p = p_mitigation
        self.q = q_compensation
        n = len(classes)

        if class_voxel_counts is not None:
            counts_t = torch.tensor(
                [max(class_voxel_counts.get(c, 1.0), 1.0) for c in classes],
                dtype=torch.float32)
        else:
            counts_t = torch.ones(n, dtype=torch.float32)

        self.register_buffer('counts', counts_t)
        self.register_buffer('cum_fp', torch.zeros(n, dtype=torch.float64))
        self.register_buffer('cum_fn', torch.zeros(n, dtype=torch.float64))

        self._online = class_voxel_counts is None
        self._batch_counter = 0

        print(f"SeesawTverskyLoss (p={p_mitigation}, q={q_compensation}, "
              f"α={alpha}, β={beta}):")
        for c, cnt in zip(classes, counts_t.tolist()):
            print(f"  {c}: voxel_count={cnt:.0f}")

    def _update_counts(self, pred: torch.Tensor, target: torch.Tensor):
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        for c in range(target.shape[1]):
            valid = ~target[:, c].isnan()
            target_clean = target[:, c].nan_to_num(0)
            positives = (target_clean * valid).sum()
            self.counts[c] = self.counts[c] + positives.float()

            pred_c = pred_binary[:, c] * valid.float()
            tgt_c = target_clean * valid.float()
            self.cum_fp[c] += (pred_c * (1 - tgt_c)).sum().double()
            self.cum_fn[c] += ((1 - pred_c) * tgt_c).sum().double()
        self._batch_counter += 1

    def _mitigation_weights(self):
        counts = self.counts.clamp(min=1.0)
        median_count = counts.median()
        ratios = median_count / counts
        weights = ratios.pow(self.p)
        weights = weights / weights.sum() * len(self.classes)
        return weights

    def _compensation_weights(self, device):
        n_classes = len(self.classes)
        comp = torch.ones(n_classes, device=device)
        total_errors = self.cum_fp + self.cum_fn
        if total_errors.sum() < 1:
            return comp
        for c in range(n_classes):
            fp_ratio = self.cum_fp[c] / total_errors[c].clamp(min=1)
            comp[c] = 1.0 + (fp_ratio.float() ** self.q)
        return comp

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_counts(pred.detach(), target)

        mitigation = self._mitigation_weights().to(pred.device)
        compensation = self._compensation_weights(pred.device)

        per_class = self.per_class_tversky(pred, target)  # (C,)
        weighted = per_class * mitigation * compensation
        return weighted.mean()


# ================================================================
# Factory
# ================================================================

def get_weighting_loss(
    loss_type: str,
    classes: List[str],
    class_voxel_counts: Optional[Dict[str, float]] = None,
    **kwargs,
) -> nn.Module:
    """Create a class-weighting loss by name.

    All losses use per-class Tversky (α=0.6, β=0.4) as the underlying
    metric.  The ``loss_type`` selects the *weighting strategy* applied
    on top.

    Args:
        loss_type:  One of ``'per_class_tversky'`` (static weights),
                    ``'class_balanced'``, ``'balanced_softmax'``,
                    ``'seesaw'``.
        classes:    Ordered list of class names.
        class_voxel_counts: Optional dict of estimated voxel counts.
        **kwargs:   Extra arguments forwarded to the loss constructor.

    Returns:
        nn.Module
    """
    t = loss_type.lower().replace('-', '_')

    # Common Tversky params
    alpha = kwargs.pop('alpha', 0.6)
    beta = kwargs.pop('beta', 0.4)
    smooth = kwargs.pop('smooth', 1e-6)

    if t == 'per_class_tversky':
        return PerClassWeightedTversky(
            classes=classes,
            class_weights=kwargs.get('class_weights'),
            alpha=alpha, beta=beta, smooth=smooth,
        )

    elif t == 'class_balanced':
        return ClassBalancedTverskyLoss(
            classes=classes,
            class_voxel_counts=class_voxel_counts,
            beta_cb=kwargs.get('beta_cb', 0.9999),
            alpha=alpha, beta=beta, smooth=smooth,
        )

    elif t in ('balanced_softmax', 'logit_adjustment'):
        return BalancedSoftmaxTverskyLoss(
            classes=classes,
            class_voxel_counts=class_voxel_counts,
            tau=kwargs.get('tau', 1.0),
            alpha=alpha, beta=beta, smooth=smooth,
        )

    elif t == 'seesaw':
        return SeesawTverskyLoss(
            classes=classes,
            class_voxel_counts=class_voxel_counts,
            p_mitigation=kwargs.get('p_mitigation', 0.8),
            q_compensation=kwargs.get('q_compensation', 2.0),
            alpha=alpha, beta=beta, smooth=smooth,
        )

    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. Choose from: "
            f"per_class_tversky, class_balanced, balanced_softmax, seesaw"
        )
