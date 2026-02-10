"""
Improved Loss Functions for CellMap Segmentation

This module provides loss functions optimized for:
- Class imbalance (rare structures like endo, er)
- Hard examples (nucleus in 2D, thin membranes)
- Direct optimization of Dice metric

Usage:
    from losses import ComboLoss, DiceBCELoss, FocalLoss, TverskyLoss
    
    criterion = ComboLoss(
        bce_weight=0.3,
        dice_weight=0.5,
        focal_weight=0.2,
        class_weights={'nuc': 3.0, 'endo_mem': 2.5}
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class DiceLoss(nn.Module):
    """
    Dice Loss for binary/multi-label segmentation.
    
    Directly optimizes the Dice coefficient, which is the evaluation metric.
    Better than BCE for imbalanced segmentation tasks.
    
    Args:
        smooth: Smoothing factor to prevent division by zero
        per_channel: If True, compute Dice per channel then average
    """
    
    def __init__(self, smooth: float = 1e-6, per_channel: bool = True):
        super().__init__()
        self.smooth = smooth
        self.per_channel = per_channel
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) or (B, C, D, H, W) logits
            target: Same shape as pred, binary labels (may contain NaN)
        """
        # Apply sigmoid to get probabilities
        pred_sigmoid = torch.sigmoid(pred)
        
        # Handle NaN in targets (mask them out)
        valid_mask = ~target.isnan()
        target = target.nan_to_num(0)
        
        if self.per_channel:
            # Compute Dice per channel
            # Flatten spatial dimensions: (B, C, *)
            pred_flat = pred_sigmoid.flatten(2)
            target_flat = target.flatten(2)
            mask_flat = valid_mask.flatten(2)
            
            # Masked intersection and union
            intersection = (pred_flat * target_flat * mask_flat).sum(dim=2)
            pred_sum = (pred_flat * mask_flat).sum(dim=2)
            target_sum = (target_flat * mask_flat).sum(dim=2)
            
            dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            
            # Average over channels and batch
            return 1 - dice.mean()
        else:
            # Global Dice
            intersection = (pred_sigmoid * target * valid_mask).sum()
            pred_sum = (pred_sigmoid * valid_mask).sum()
            target_sum = (target * valid_mask).sum()
            
            dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE Loss.
    
    BCE provides pixel-level supervision while Dice optimizes the global metric.
    This combination often works better than either alone.
    
    Args:
        bce_weight: Weight for BCE component (default 0.5)
        dice_weight: Weight for Dice component (default 0.5)
        smooth: Dice smoothing factor
        pos_weight: Optional per-class positive weights for BCE
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle NaN masking for BCE
        valid_mask = ~target.isnan()
        target_clean = target.nan_to_num(0)
        
        # BCE with masking
        bce_loss = self.bce(pred, target_clean)
        bce_loss = (bce_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        
        # Dice loss
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for hard example mining.
    
    Down-weights easy examples (high confidence correct predictions) and
    focuses training on hard examples. Especially useful for:
    - Rare classes
    - Ambiguous boundaries
    - Classes that look similar to others
    
    Args:
        alpha: Balancing factor for positive class (default 0.25)
        gamma: Focusing parameter - higher = more focus on hard examples (default 2.0)
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle NaN masking
        valid_mask = ~target.isnan()
        target_clean = target.nan_to_num(0)
        
        # BCE loss per pixel
        bce = F.binary_cross_entropy_with_logits(pred, target_clean, reduction='none')
        
        # pt = p if y=1, else 1-p
        pt = torch.exp(-bce)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha balancing
        alpha_weight = target_clean * self.alpha + (1 - target_clean) * (1 - self.alpha)
        
        # Combined focal loss
        focal_loss = alpha_weight * focal_weight * bce
        
        # Apply valid mask and average
        return (focal_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)


class TverskyLoss(nn.Module):
    """
    Tversky Loss for controlling precision/recall trade-off.
    
    Generalizes Dice loss with asymmetric weighting of false positives vs
    false negatives. Useful when you want to prioritize one over the other.
    
    Args:
        alpha: Weight for false positives (higher = penalize FP more = higher precision)
        beta: Weight for false negatives (higher = penalize FN more = higher recall)
        smooth: Smoothing factor
    
    Note: alpha=beta=0.5 is equivalent to Dice loss
    
    Reference: "Tversky loss function for image segmentation" (Salehi et al., 2017)
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        
        # Handle NaN masking
        valid_mask = ~target.isnan()
        target_clean = target.nan_to_num(0)
        
        # Flatten for computation
        pred_flat = (pred_sigmoid * valid_mask).flatten(2)
        target_flat = (target_clean * valid_mask).flatten(2)
        
        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum(dim=2)
        fp = (pred_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - pred_flat) * target_flat).sum(dim=2)
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky.mean()


class ComboLoss(nn.Module):
    """
    Combination of multiple loss functions with per-class weighting.
    
    This is the recommended loss for CellMap segmentation. It combines:
    - BCE: Pixel-level supervision
    - Dice: Global metric optimization
    - Focal: Hard example mining
    
    Plus optional per-class weighting to boost struggling classes.
    
    Args:
        bce_weight: Weight for BCE loss (default 0.3)
        dice_weight: Weight for Dice loss (default 0.5)
        focal_weight: Weight for Focal loss (default 0.2)
        class_weights: Dict mapping class names to loss multipliers
        classes: List of class names (must match order in predictions)
        focal_gamma: Focal loss gamma parameter
        smooth: Dice/Tversky smoothing factor
    """
    
    def __init__(
        self,
        bce_weight: float = 0.3,
        dice_weight: float = 0.5,
        focal_weight: float = 0.2,
        class_weights: Optional[Dict[str, float]] = None,
        classes: Optional[list] = None,
        focal_gamma: float = 2.0,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(gamma=focal_gamma)
        
        # Build class weight tensor if provided
        self.class_weight_tensor = None
        if class_weights is not None and classes is not None:
            weights = [class_weights.get(c, 1.0) for c in classes]
            self.register_buffer(
                'class_weight_tensor',
                torch.tensor(weights, dtype=torch.float32)
            )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle NaN masking for BCE
        valid_mask = ~target.isnan()
        target_clean = target.nan_to_num(0)
        
        # BCE loss
        bce_loss = self.bce(pred, target_clean)
        bce_loss = (bce_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        
        # Dice loss
        dice_loss = self.dice(pred, target)
        
        # Focal loss
        focal_loss = self.focal(pred, target)
        
        # Combine losses
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.focal_weight * focal_loss
        )
        
        return total_loss


class PerClassComboLoss(nn.Module):
    """
    ComboLoss with explicit per-channel (per-class) weighting.
    
    Computes loss for each class separately and applies class-specific weights.
    This allows boosting loss for hard classes like nucleus.
    
    Args:
        classes: List of class names
        class_weights: Dict mapping class name -> weight multiplier
        base_loss: Base loss function to use ('dice_bce', 'focal', 'tversky')
    """
    
    # Default weights based on UNet 2D evaluation results
    # Weights are INVERSELY proportional to Dice score:
    #   Dice < 0.15 -> weight 3.0-3.5 (critical)
    #   Dice 0.15-0.25 -> weight 2.0-2.5 (hard)
    #   Dice 0.25-0.40 -> weight 1.5 (moderate)
    #   Dice > 0.40 -> weight 1.0 (good)
    DEFAULT_WEIGHTS = {
        # Critical (Dice < 0.15) - need heavy boosting
        'endo_mem': 3.0,    # 0.081 Dice - worst performer
        'endo_lum': 3.0,    # 0.099 Dice
        'nuc': 3.5,         # 0.111 Dice - composite class, needs 3D context
        'pm': 2.5,          # 0.113 Dice - thin boundary
        'er_mem': 2.5,      # 0.116 Dice
        'er_lum': 2.0,      # 0.143 Dice
        
        # Hard (Dice 0.15-0.25)
        'ves_mem': 2.0,     # 0.193 Dice
        'mito_mem': 1.8,    # 0.218 Dice
        
        # Moderate (Dice 0.25-0.40)
        'mito_lum': 1.5,    # 0.257 Dice
        'ves_lum': 1.5,     # 0.270 Dice
        'ecs': 1.5,         # 0.291 Dice
        'golgi_lum': 1.5,   # 0.317 Dice
        
        # Good (Dice > 0.40) - no boost needed
        'mito_ribo': 1.0,   # 0.643 Dice
        'golgi_mem': 1.0,   # 0.680 Dice - best performer
    }
    
    def __init__(
        self,
        classes: list,
        class_weights: Optional[Dict[str, float]] = None,
        bce_weight: float = 0.4,
        dice_weight: float = 0.6,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.classes = classes
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.smooth = smooth
        
        # Use provided weights or defaults
        weights = class_weights or self.DEFAULT_WEIGHTS
        weight_list = [weights.get(c, 1.0) for c in classes]
        self.register_buffer(
            'weights',
            torch.tensor(weight_list, dtype=torch.float32)
        )
        
        components = []
        if bce_weight > 0: components.append(f'BCE({bce_weight})')
        if dice_weight > 0: components.append(f'Dice({dice_weight})')
        if focal_weight > 0: components.append(f'Focal({focal_weight}, γ={focal_gamma})')
        print(f"PerClassComboLoss initialized: {' + '.join(components)}")
        print(f"  Class weights:")
        for c, w in zip(classes, weight_list):
            print(f"    {c}: {w:.1f}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, C, H, W) binary targets (may have NaN)
        """
        batch_size, n_classes = pred.shape[:2]
        total_loss = 0.0
        
        for c in range(n_classes):
            pred_c = pred[:, c:c+1]
            target_c = target[:, c:c+1]
            
            # Handle NaN
            valid_mask = ~target_c.isnan()
            target_clean = target_c.nan_to_num(0)
            
            # Skip if no valid pixels
            if valid_mask.sum() == 0:
                continue
            
            # BCE for this class
            bce = F.binary_cross_entropy_with_logits(
                pred_c, target_clean, reduction='none'
            )
            bce_loss = (bce * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            
            # Dice for this class
            pred_sigmoid = torch.sigmoid(pred_c)
            pred_flat = (pred_sigmoid * valid_mask).flatten()
            target_flat = (target_clean * valid_mask).flatten()
            
            intersection = (pred_flat * target_flat).sum()
            dice = (2 * intersection + self.smooth) / (
                pred_flat.sum() + target_flat.sum() + self.smooth
            )
            dice_loss = 1 - dice
            
            # Focal for this class (reuses BCE computation)
            focal_loss = 0.0
            if self.focal_weight > 0:
                pt = torch.exp(-bce)  # probability of correct class
                focal_mod = (1 - pt) ** self.focal_gamma
                alpha_t = target_clean * self.focal_alpha + (1 - target_clean) * (1 - self.focal_alpha)
                focal_loss = (alpha_t * focal_mod * bce * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            
            # Combine and weight
            class_loss = (self.bce_weight * bce_loss 
                         + self.dice_weight * dice_loss 
                         + self.focal_weight * focal_loss)
            weighted_loss = self.weights[c] * class_loss
            total_loss += weighted_loss
        
        return total_loss / n_classes


class PerClassTverskyLoss(nn.Module):
    """
    Per-class weighted Tversky loss.
    
    Combines per-class weighting (to boost hard classes) with Tversky's
    asymmetric FP/FN trade-off. With beta > alpha, this penalizes false
    negatives more heavily — ideal for thin structures (membranes) where
    missing a pixel is worse than hallucinating one.
    
    Args:
        classes: List of class names
        class_weights: Dict mapping class name -> weight multiplier
        alpha: FP weight (higher = penalize false positives more)
        beta: FN weight (higher = penalize false negatives more)
        smooth: Smoothing factor
    """
    
    def __init__(
        self,
        classes: list,
        class_weights: Optional[Dict[str, float]] = None,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.classes = classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
        weights = class_weights or PerClassComboLoss.DEFAULT_WEIGHTS
        weight_list = [weights.get(c, 1.0) for c in classes]
        self.register_buffer(
            'weights',
            torch.tensor(weight_list, dtype=torch.float32)
        )
        
        print(f"PerClassTverskyLoss initialized (α={alpha}, β={beta}):")
        for c, w in zip(classes, weight_list):
            print(f"  {c}: {w:.1f}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, n_classes = pred.shape[:2]
        total_loss = 0.0
        
        for c in range(n_classes):
            pred_c = pred[:, c:c+1]
            target_c = target[:, c:c+1]
            
            # Handle NaN
            valid_mask = ~target_c.isnan()
            target_clean = target_c.nan_to_num(0)
            
            if valid_mask.sum() == 0:
                continue
            
            pred_sigmoid = torch.sigmoid(pred_c)
            pred_flat = (pred_sigmoid * valid_mask).flatten()
            target_flat = (target_clean * valid_mask).flatten()
            
            # Tversky components
            tp = (pred_flat * target_flat).sum()
            fp = (pred_flat * (1 - target_flat)).sum()
            fn = ((1 - pred_flat) * target_flat).sum()
            
            denom = tp + self.alpha * fp + self.beta * fn + self.smooth
            tversky = (tp + self.smooth) / denom.clamp(min=self.smooth)
            class_loss = 1 - tversky
            
            weighted_loss = self.weights[c] * class_loss
            total_loss += weighted_loss
        
        return total_loss / n_classes


class NaNSafeBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss that handles NaN targets by masking them out."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid_mask = ~target.isnan()
        target_clean = target.nan_to_num(0)
        bce = F.binary_cross_entropy_with_logits(pred, target_clean, reduction='none')
        return (bce * valid_mask).sum() / valid_mask.sum().clamp(min=1)


# Convenience function to create loss from config
def get_loss_function(
    loss_type: str = 'combo',
    classes: Optional[list] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss by name.
    
    Args:
        loss_type: One of 'bce', 'dice', 'dice_bce', 'focal', 'tversky', 
                   'combo', 'per_class_combo'
        classes: List of class names (required for per_class_combo)
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss module
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'bce':
        return NaNSafeBCEWithLogitsLoss()
    elif loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'dice_bce':
        return DiceBCELoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'combo':
        return ComboLoss(classes=classes, **kwargs)
    elif loss_type == 'per_class_combo':
        if classes is None:
            raise ValueError("classes must be provided for per_class_combo loss")
        return PerClassComboLoss(classes=classes, **kwargs)
    elif loss_type == 'per_class_tversky':
        if classes is None:
            raise ValueError("classes must be provided for per_class_tversky loss")
        return PerClassTverskyLoss(classes=classes, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
