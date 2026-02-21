# Comprehensive Metrics Tracker for Model Comparison
# Tracks training/validation metrics and enables cross-model comparison

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import json
import os
from pathlib import Path
import time


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    DiceLoss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice Loss.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions (logits), shape (B, C, ...)
        targets : torch.Tensor
            Ground truth, shape (B, C, ...)
            
        Returns
        -------
        torch.Tensor
            Dice loss value
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(predictions)
        
        # Handle NaN values in targets
        mask = ~torch.isnan(targets)
        targets = targets.nan_to_num(0)
        
        # Flatten spatial dimensions
        probs = probs.flatten(2)
        targets = targets.flatten(2)
        mask = mask.flatten(2)
        
        # Compute intersection and union with masking
        intersection = ((probs * targets) * mask).sum(dim=2)
        union = ((probs + targets) * mask).sum(dim=2)
        
        # Dice coefficient per class
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss
        dice_loss = 1.0 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'none':
            return dice_loss
        else:
            return dice_loss.sum()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Features:
    - Supports per-class weights (pos_weight) for multi-class segmentation
    - Down-weights easy examples, focuses training on hard negatives
    - Handles NaN values in targets (masked regions)
    
    Parameters
    ----------
    alpha : float or torch.Tensor
        Balance factor. If tensor, should be shape (C,) for C classes.
        Default 0.25 works well for most cases.
    gamma : float
        Focus parameter. Higher values = more focus on hard examples.
        gamma=0 is equivalent to BCE. gamma=2.0 is standard.
    pos_weight : torch.Tensor, optional
        Per-class positive weights, shape (C, 1, 1) or (C, 1, 1, 1) for 3D.
        If provided, applies class-specific weighting to positive examples.
    reduction : str
        'mean', 'sum', or 'none'
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # Register pos_weight as buffer so it moves with model to GPU
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Model logits, shape (B, C, H, W) or (B, C, D, H, W)
        targets : torch.Tensor
            Ground truth, shape (B, C, H, W) or (B, C, D, H, W)
            
        Returns
        -------
        torch.Tensor
            Focal loss value
        """
        probs = torch.sigmoid(predictions)
        
        # Handle NaN values (masked regions)
        mask = ~torch.isnan(targets)
        targets = targets.nan_to_num(0)
        
        # Base cross-entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        
        # Focal weight: (1 - p_t)^gamma
        # p_t = p if y=1, else (1-p)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha balance factor
        loss = self.alpha * focal_weight * ce_loss
        
        # Apply per-class positive weights if provided
        if self.pos_weight is not None:
            # pos_weight should be broadcastable: (C, 1, 1) or (C, 1, 1, 1)
            # Apply higher weight to positive examples of rare classes
            weight = targets * self.pos_weight + (1 - targets)
            loss = loss * weight
        
        # Apply mask for NaN regions
        loss = loss * mask
        
        if self.reduction == 'mean':
            return loss.sum() / mask.sum().clamp(min=1)
        elif self.reduction == 'none':
            return loss
        else:
            return loss.sum()


class MetricsTracker:
    """
    Comprehensive metrics tracker for model comparison experiments.
    
    Tracks:
    - Training loss (BCE, Dice, combined)
    - Validation loss
    - Per-class metrics (Dice, IoU, precision, recall)
    - Timing information
    - Learning rate schedule
    """
    
    def __init__(
        self,
        classes: List[str],
        save_path: str,
        model_name: str,
        track_per_class: bool = True
    ):
        """
        Initialize metrics tracker.
        
        Parameters
        ----------
        classes : List[str]
            List of class names
        save_path : str
            Path to save metrics
        model_name : str
            Name of the model being tracked
        track_per_class : bool
            Whether to track per-class metrics
        """
        self.classes = classes
        self.n_classes = len(classes)
        self.save_path = Path(save_path)
        self.model_name = model_name
        self.track_per_class = track_per_class
        
        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric storage
        self.reset()
        
        # Loss functions for additional metrics
        self.dice_loss = DiceLoss(reduction='none')
        
        # Timing
        self.epoch_start_time = None
        self.total_training_time = 0
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            # Per-epoch metrics
            'epoch': [],
            'train_loss_bce': [],
            'train_loss_dice': [],
            'train_loss_combined': [],
            'val_loss_bce': [],
            'val_loss_dice': [],
            'val_loss_combined': [],
            
            # Aggregated metrics
            'train_dice_mean': [],
            'train_iou_mean': [],
            'train_accuracy': [],
            'val_dice_mean': [],
            'val_iou_mean': [],
            'val_accuracy': [],
            
            # Learning rate
            'learning_rate': [],
            
            # Timing
            'epoch_time': [],
            'total_time': [],
            
            # Iteration-level metrics (for plotting)
            'iter_loss': [],
            'iter_num': [],
        }
        
        # Per-class metrics
        if self.track_per_class:
            self.per_class_metrics = {
                'train_dice': defaultdict(list),
                'train_iou': defaultdict(list),
                'val_dice': defaultdict(list),
                'val_iou': defaultdict(list),
                'val_precision': defaultdict(list),
                'val_recall': defaultdict(list),
            }
    
    def start_epoch(self) -> None:
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
        
        # Reset batch accumulators
        self._batch_losses = []
        self._batch_dice_scores = []
        self._batch_predictions = []
        self._batch_targets = []
    
    def log_iteration(
        self,
        iteration: int,
        loss: float,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> None:
        """Log metrics for a single training iteration."""
        self.metrics['iter_loss'].append(loss)
        self.metrics['iter_num'].append(iteration)
        self._batch_losses.append(loss)
        
        if predictions is not None and targets is not None:
            dice = self._compute_dice_scores(predictions, targets)
            self._batch_dice_scores.append(dice)
    
    def end_epoch(
        self,
        epoch: int,
        val_predictions: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
        learning_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Finalize metrics for an epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        val_predictions : torch.Tensor, optional
            Validation predictions
        val_targets : torch.Tensor, optional
            Validation targets
        learning_rate : float, optional
            Current learning rate
            
        Returns
        -------
        Dict[str, float]
            Summary metrics for the epoch
        """
        epoch_time = time.time() - self.epoch_start_time
        self.total_training_time += epoch_time
        
        self.metrics['epoch'].append(epoch)
        self.metrics['epoch_time'].append(epoch_time)
        self.metrics['total_time'].append(self.total_training_time)
        
        if learning_rate is not None:
            self.metrics['learning_rate'].append(learning_rate)
        
        # Aggregate training metrics
        train_loss = np.mean(self._batch_losses) if self._batch_losses else 0
        self.metrics['train_loss_bce'].append(train_loss)
        
        if self._batch_dice_scores:
            dice_scores = np.stack(self._batch_dice_scores)
            mean_dice = np.nanmean(dice_scores)
            self.metrics['train_dice_mean'].append(mean_dice)
            self.metrics['train_loss_dice'].append(1 - mean_dice)
            self.metrics['train_loss_combined'].append(train_loss + (1 - mean_dice))
        else:
            self.metrics['train_dice_mean'].append(0)
            self.metrics['train_loss_dice'].append(1)
            self.metrics['train_loss_combined'].append(train_loss + 1)
        
        # Validation metrics
        if val_predictions is not None and val_targets is not None:
            val_metrics = self._compute_validation_metrics(
                val_predictions, val_targets
            )
            for key, value in val_metrics.items():
                if key.startswith('val_'):
                    self.metrics[key].append(value)
        
        # Create summary
        summary = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': self.metrics['train_dice_mean'][-1],
            'epoch_time': epoch_time,
        }
        
        if 'val_loss_bce' in self.metrics and self.metrics['val_loss_bce']:
            summary['val_loss'] = self.metrics['val_loss_bce'][-1]
            summary['val_dice'] = self.metrics['val_dice_mean'][-1]
        
        return summary
    
    def _compute_dice_scores(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> np.ndarray:
        """Compute per-class Dice scores."""
        with torch.no_grad():
            probs = torch.sigmoid(predictions)
            
            # Handle NaN
            mask = ~torch.isnan(targets)
            targets = targets.nan_to_num(0)
            
            # Binarize predictions at 0.5 threshold
            preds_binary = (probs > 0.5).float()
            
            # Compute per-class Dice
            dice_scores = []
            for c in range(predictions.shape[1]):
                pred_c = preds_binary[:, c].flatten()
                target_c = targets[:, c].flatten()
                mask_c = mask[:, c].flatten()
                
                pred_c = pred_c[mask_c]
                target_c = target_c[mask_c]
                
                if len(pred_c) == 0:
                    dice_scores.append(np.nan)
                    continue
                
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                
                if union == 0:
                    dice = 1.0 if intersection == 0 else 0.0
                else:
                    dice = (2 * intersection / union).item()
                
                dice_scores.append(dice)
            
            return np.array(dice_scores)
    
    def _compute_validation_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute comprehensive validation metrics."""
        with torch.no_grad():
            probs = torch.sigmoid(predictions)
            preds_binary = (probs > 0.5).float()
            
            # Handle NaN
            mask = ~torch.isnan(targets)
            targets_clean = targets.nan_to_num(0)
            
            metrics = {}
            
            # BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(
                predictions, targets_clean, reduction='none'
            )
            bce_loss = (bce_loss * mask).sum() / mask.sum().clamp(min=1)
            metrics['val_loss_bce'] = bce_loss.item()
            
            # Per-class metrics
            dice_scores = []
            iou_scores = []
            precision_scores = []
            recall_scores = []
            
            for c in range(predictions.shape[1]):
                pred_c = preds_binary[:, c].flatten()
                target_c = targets_clean[:, c].flatten()
                mask_c = mask[:, c].flatten()
                
                pred_c = pred_c[mask_c]
                target_c = target_c[mask_c]
                
                if len(pred_c) == 0:
                    continue
                
                tp = (pred_c * target_c).sum()
                fp = (pred_c * (1 - target_c)).sum()
                fn = ((1 - pred_c) * target_c).sum()
                
                # Dice
                dice = (2 * tp / (2 * tp + fp + fn + 1e-6)).item()
                dice_scores.append(dice)
                
                # IoU
                iou = (tp / (tp + fp + fn + 1e-6)).item()
                iou_scores.append(iou)
                
                # Precision
                precision = (tp / (tp + fp + 1e-6)).item()
                precision_scores.append(precision)
                
                # Recall
                recall = (tp / (tp + fn + 1e-6)).item()
                recall_scores.append(recall)
                
                # Store per-class metrics
                if self.track_per_class:
                    class_name = self.classes[c] if c < len(self.classes) else f'class_{c}'
                    self.per_class_metrics['val_dice'][class_name].append(dice)
                    self.per_class_metrics['val_iou'][class_name].append(iou)
                    self.per_class_metrics['val_precision'][class_name].append(precision)
                    self.per_class_metrics['val_recall'][class_name].append(recall)
            
            metrics['val_dice_mean'] = np.mean(dice_scores) if dice_scores else 0
            metrics['val_iou_mean'] = np.mean(iou_scores) if iou_scores else 0
            metrics['val_loss_dice'] = 1 - metrics['val_dice_mean']
            metrics['val_loss_combined'] = metrics['val_loss_bce'] + metrics['val_loss_dice']
            
            # Overall accuracy
            correct = ((preds_binary == targets_clean) * mask).sum()
            total = mask.sum()
            metrics['val_accuracy'] = (correct / total.clamp(min=1)).item()
            
            return metrics
    
    def save(self) -> None:
        """Save all metrics to disk."""
        # Save main metrics as JSON
        metrics_path = self.save_path / f'{self.model_name}_metrics.json'
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for key, value in self.metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                else:
                    serializable_metrics[key] = value
            json.dump(serializable_metrics, f, indent=2)
        
        # Save per-class metrics
        if self.track_per_class:
            per_class_path = self.save_path / f'{self.model_name}_per_class_metrics.json'
            with open(per_class_path, 'w') as f:
                serializable = {}
                for metric_type, class_dict in self.per_class_metrics.items():
                    serializable[metric_type] = dict(class_dict)
                json.dump(serializable, f, indent=2)
        
        # Save as numpy arrays for easy loading
        npz_path = self.save_path / f'{self.model_name}_metrics.npz'
        np.savez_compressed(
            npz_path,
            **{k: np.array(v) for k, v in self.metrics.items() if v}
        )
        
        print(f"Saved metrics to {self.save_path}")
    
    def load(self, model_name: Optional[str] = None) -> None:
        """Load metrics from disk."""
        if model_name is None:
            model_name = self.model_name
        
        metrics_path = self.save_path / f'{model_name}_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
        
        per_class_path = self.save_path / f'{model_name}_per_class_metrics.json'
        if per_class_path.exists():
            with open(per_class_path, 'r') as f:
                loaded = json.load(f)
                self.per_class_metrics = {
                    k: defaultdict(list, v) for k, v in loaded.items()
                }
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        summary = {
            'model_name': self.model_name,
            'total_epochs': len(self.metrics['epoch']),
            'total_training_time': self.total_training_time,
        }
        
        if self.metrics['val_dice_mean']:
            summary['best_val_dice'] = max(self.metrics['val_dice_mean'])
            summary['best_val_dice_epoch'] = self.metrics['epoch'][
                np.argmax(self.metrics['val_dice_mean'])
            ]
            summary['final_val_dice'] = self.metrics['val_dice_mean'][-1]
        
        if self.metrics['val_loss_bce']:
            summary['best_val_loss'] = min(self.metrics['val_loss_bce'])
            summary['final_val_loss'] = self.metrics['val_loss_bce'][-1]
        
        return summary


class ModelComparator:
    """
    Compare metrics across multiple models.
    """
    
    def __init__(self, metrics_dir: str):
        """
        Initialize comparator.
        
        Parameters
        ----------
        metrics_dir : str
            Directory containing saved metrics from different models
        """
        self.metrics_dir = Path(metrics_dir)
        self.models: Dict[str, Dict] = {}
    
    def load_all_models(self) -> None:
        """Load metrics from all models in the directory."""
        for metrics_file in self.metrics_dir.glob('*_metrics.json'):
            model_name = metrics_file.stem.replace('_metrics', '')
            with open(metrics_file, 'r') as f:
                self.models[model_name] = json.load(f)
        
        print(f"Loaded metrics for {len(self.models)} models: {list(self.models.keys())}")
    
    def compare_metric(
        self,
        metric_name: str,
        models: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare a specific metric across models.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric to compare
        models : List[str], optional
            Specific models to compare. If None, compares all.
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Statistics for each model
        """
        if models is None:
            models = list(self.models.keys())
        
        comparison = {}
        for model_name in models:
            if model_name not in self.models:
                continue
            
            values = self.models[model_name].get(metric_name, [])
            if not values:
                continue
            
            comparison[model_name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'final': float(values[-1]),
            }
        
        return comparison
    
    def rank_models(
        self,
        metric_name: str,
        higher_is_better: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Rank models by a specific metric (using best value).
        
        Parameters
        ----------
        metric_name : str
            Metric to rank by
        higher_is_better : bool
            Whether higher values are better
            
        Returns
        -------
        List[Tuple[str, float]]
            Sorted list of (model_name, best_value) tuples
        """
        scores = []
        for model_name, metrics in self.models.items():
            values = metrics.get(metric_name, [])
            if values:
                best = max(values) if higher_is_better else min(values)
                scores.append((model_name, best))
        
        scores.sort(key=lambda x: x[1], reverse=higher_is_better)
        return scores
    
    def generate_comparison_table(self) -> str:
        """Generate a markdown table comparing all models."""
        metrics_to_compare = [
            ('val_dice_mean', 'Val Dice', True),
            ('val_iou_mean', 'Val IoU', True),
            ('val_loss_bce', 'Val BCE Loss', False),
            ('train_loss_bce', 'Train Loss', False),
        ]
        
        # Header
        table = "| Model | " + " | ".join([m[1] for m in metrics_to_compare]) + " |\n"
        table += "|-------|" + "|".join(["-------" for _ in metrics_to_compare]) + "|\n"
        
        for model_name in self.models.keys():
            row = f"| {model_name} |"
            for metric_name, _, higher_better in metrics_to_compare:
                values = self.models[model_name].get(metric_name, [])
                if values:
                    best = max(values) if higher_better else min(values)
                    row += f" {best:.4f} |"
                else:
                    row += " N/A |"
            table += row + "\n"
        
        return table


if __name__ == "__main__":
    # Test the metrics tracker
    print("Testing MetricsTracker...")
    
    classes = ['ecs', 'pm', 'mito_mem', 'mito_lum', 'nuc']
    tracker = MetricsTracker(classes, '/tmp/test_metrics', 'test_model')
    
    # Simulate training
    for epoch in range(5):
        tracker.start_epoch()
        
        for i in range(10):
            loss = 1.0 / (epoch + 1) + np.random.random() * 0.1
            tracker.log_iteration(epoch * 10 + i, loss)
        
        # Simulate validation
        val_preds = torch.randn(2, 5, 64, 64)
        val_targets = torch.randint(0, 2, (2, 5, 64, 64)).float()
        
        summary = tracker.end_epoch(
            epoch, val_preds, val_targets, learning_rate=0.001 / (epoch + 1)
        )
        print(f"Epoch {epoch}: {summary}")
    
    tracker.save()
    print(f"\nFinal summary: {tracker.get_summary()}")
