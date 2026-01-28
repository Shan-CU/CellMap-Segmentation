# Feature Extraction Module for Model Comparison
# Extracts intermediate feature maps from different model architectures

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional, Any
from collections import OrderedDict
import numpy as np


class FeatureExtractor:
    """
    Hook-based feature extractor for CNN and Transformer models.
    
    Registers forward hooks on specified layers to capture intermediate
    feature maps during forward pass. Works with any PyTorch model.
    
    Usage:
        extractor = FeatureExtractor(model, layer_names=['encoder.layer1', 'decoder.up1'])
        features = extractor(input_tensor)
        # features['encoder.layer1'] contains the feature map from that layer
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        layer_names: Optional[List[str]] = None,
        return_input: bool = True,
        return_output: bool = True
    ):
        """
        Initialize feature extractor.
        
        Parameters
        ----------
        model : nn.Module
            The PyTorch model to extract features from
        layer_names : List[str], optional
            Names of layers to extract features from. If None, uses default
            layers based on model architecture
        return_input : bool
            Whether to include model input in features dict
        return_output : bool
            Whether to include model output in features dict
        """
        self.model = model
        self.return_input = return_input
        self.return_output = return_output
        self.features: Dict[str, torch.Tensor] = OrderedDict()
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Auto-detect layers if not specified
        if layer_names is None:
            layer_names = self._auto_detect_layers()
        
        self.layer_names = layer_names
        self._register_hooks()
    
    def _auto_detect_layers(self) -> List[str]:
        """
        Auto-detect meaningful layers based on model architecture.
        """
        model_class = self.model.__class__.__name__
        
        # Handle DDP wrapped models
        if hasattr(self.model, 'module'):
            inner_model = self.model.module
            model_class = inner_model.__class__.__name__
        else:
            inner_model = self.model
            
        if 'UNet' in model_class:
            return self._get_unet_layers(inner_model)
        elif 'ResNet' in model_class or 'Resnet' in model_class:
            return self._get_resnet_layers(inner_model)
        elif 'Swin' in model_class:
            return self._get_swin_layers(inner_model)
        elif 'ViT' in model_class:
            return self._get_vit_layers(inner_model)
        else:
            # Return first few layers as fallback
            layers = []
            for name, _ in inner_model.named_modules():
                if name and '.' not in name:  # Top-level modules only
                    layers.append(name)
                if len(layers) >= 5:
                    break
            return layers
    
    def _get_unet_layers(self, model: nn.Module) -> List[str]:
        """Get key layers for UNet architecture."""
        prefix = 'module.' if hasattr(self.model, 'module') else ''
        return [
            f'{prefix}inc',          # Initial conv
            f'{prefix}down1',        # First encoder stage
            f'{prefix}down2',        # Second encoder stage  
            f'{prefix}down3',        # Third encoder stage
            f'{prefix}down4',        # Bottleneck
            f'{prefix}up1',          # First decoder stage
            f'{prefix}up2',          # Second decoder stage
            f'{prefix}up3',          # Third decoder stage
            f'{prefix}up4',          # Fourth decoder stage
            f'{prefix}outc',         # Output conv
        ]
    
    def _get_resnet_layers(self, model: nn.Module) -> List[str]:
        """Get key layers for ResNet architecture."""
        prefix = 'module.' if hasattr(self.model, 'module') else ''
        # ResNet model structure varies, detect available layers
        layers = []
        for name, _ in model.named_modules():
            if name in ['model.0', 'model.3', 'model.6', 'model.9', 'model.12']:
                layers.append(f'{prefix}{name}')
        if not layers:
            # Fallback for different ResNet implementations
            layers = [f'{prefix}model']
        return layers
    
    def _get_swin_layers(self, model: nn.Module) -> List[str]:
        """Get key layers for Swin Transformer architecture (2D and 3D)."""
        prefix = 'module.' if hasattr(self.model, 'module') else ''
        
        # Check if it's a 3D Swin Transformer
        model_class = model.__class__.__name__
        if '3D' in model_class or '3d' in model_class:
            # SwinTransformer3D has encoder_layers structure
            return [
                f'{prefix}encoder_layers.0',  # Patch embedding
                f'{prefix}encoder_layers.1',  # Stage 1
                f'{prefix}encoder_layers.3',  # Stage 2
                f'{prefix}encoder_layers.5',  # Stage 3
                f'{prefix}encoder_layers.7',  # Stage 4
                f'{prefix}up_blocks',         # Decoder blocks
                f'{prefix}head',              # Segmentation head
            ]
        else:
            # 2D Swin Transformer
            return [
                f'{prefix}encoder_layers.0',  # Patch embedding
                f'{prefix}encoder_layers.1',  # Stage 1
                f'{prefix}encoder_layers.3',  # Stage 2
                f'{prefix}encoder_layers.5',  # Stage 3
                f'{prefix}encoder_layers.7',  # Stage 4
                f'{prefix}norm',              # Final norm
                f'{prefix}up_blocks',         # Decoder (if exists)
            ]
    
    def _get_vit_layers(self, model: nn.Module) -> List[str]:
        """Get key layers for ViT architecture (2D and 3D)."""
        prefix = 'module.' if hasattr(self.model, 'module') else ''
        
        # Check if it's a 2D ViT
        model_class = model.__class__.__name__
        if '2D' in model_class or '2d' in model_class:
            # ViTVNet2D structure
            return [
                f'{prefix}transformer.embeddings',  # Embeddings
                f'{prefix}transformer.encoder',     # Transformer encoder
                f'{prefix}decoder',                 # Decoder
                f'{prefix}seg_head',                # Segmentation head
            ]
        else:
            # 3D ViT-V-Net structure
            return [
                f'{prefix}transformer.embeddings',  # Embeddings
                f'{prefix}transformer.encoder',     # Transformer encoder
                f'{prefix}decoder',                 # Decoder
            ]
    
    def _get_hook(self, name: str) -> Callable:
        """Create a forward hook that stores the output."""
        def hook(module: nn.Module, input: Any, output: Any) -> None:
            # Handle tuple outputs (common in transformers)
            if isinstance(output, tuple):
                output = output[0]
            self.features[name] = output.detach()
        return hook
    
    def _register_hooks(self) -> None:
        """Register forward hooks on specified layers."""
        # Get the actual model (handle DDP)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        for name in self.layer_names:
            # Remove module prefix for lookup if present
            lookup_name = name.replace('module.', '')
            try:
                layer = dict(model.named_modules())[lookup_name]
                hook = layer.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)
            except KeyError:
                print(f"Warning: Layer '{name}' not found in model, skipping.")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from input.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping layer names to feature tensors
        """
        self.features.clear()
        
        if self.return_input:
            self.features['input'] = x.detach()
        
        with torch.no_grad():
            output = self.model(x)
        
        if self.return_output:
            self.features['output'] = output.detach()
        
        return self.features.copy()
    
    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()


class AttentionExtractor:
    """
    Extract attention maps from transformer-based models.
    Useful for visualizing where the model "attends" to.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_maps: Dict[str, torch.Tensor] = OrderedDict()
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_attention_hooks()
    
    def _register_attention_hooks(self) -> None:
        """Register hooks to capture attention weights."""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        for name, module in model.named_modules():
            # Swin Transformer attention
            if 'ShiftedWindowAttention' in module.__class__.__name__:
                hook = module.register_forward_hook(self._attention_hook(name))
                self.hooks.append(hook)
            # Standard transformer attention  
            elif 'Attention' in module.__class__.__name__ and hasattr(module, 'softmax'):
                hook = module.register_forward_hook(self._attention_hook(name))
                self.hooks.append(hook)
    
    def _attention_hook(self, name: str) -> Callable:
        """Create hook for attention modules."""
        def hook(module, input, output):
            # Different attention modules return attention weights differently
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]  # Usually second element
                if attn_weights is not None:
                    self.attention_maps[name] = attn_weights.detach()
        return hook
    
    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention maps."""
        self.attention_maps.clear()
        with torch.no_grad():
            _ = self.model(x)
        return self.attention_maps.copy()
    
    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def visualize_feature_maps(
    features: Dict[str, torch.Tensor],
    batch_idx: int = 0,
    max_channels: int = 16,
    figsize: tuple = (20, 10)
) -> 'matplotlib.figure.Figure':
    """
    Create a visualization of feature maps from multiple layers.
    
    Parameters
    ----------
    features : Dict[str, torch.Tensor]
        Feature maps from FeatureExtractor
    batch_idx : int
        Which batch element to visualize
    max_channels : int
        Maximum number of channels to show per layer
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    
    n_layers = len(features)
    fig, axes = plt.subplots(n_layers, max_channels, figsize=figsize)
    
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for row, (name, feat) in enumerate(features.items()):
        if feat.dim() == 5:  # 3D: B, C, D, H, W
            # Take middle slice
            mid_d = feat.shape[2] // 2
            feat = feat[:, :, mid_d, :, :]
        elif feat.dim() == 4:  # 2D: B, C, H, W
            pass
        else:
            continue
            
        # Select batch element
        feat = feat[batch_idx]
        
        # Normalize across all channels for this layer
        feat_min = feat.min()
        feat_max = feat.max()
        if feat_max - feat_min > 1e-6:
            feat = (feat - feat_min) / (feat_max - feat_min)
        
        n_channels = min(feat.shape[0], max_channels)
        
        for col in range(max_channels):
            ax = axes[row, col]
            if col < n_channels:
                arr = feat[col].cpu().numpy()
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32, copy=False)
                ax.imshow(arr, cmap='viridis')
                if col == 0:
                    ax.set_ylabel(name, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    return fig


def extract_and_save_features(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    save_path: str,
    device: torch.device,
    num_samples: int = 10,
    layer_names: Optional[List[str]] = None
) -> None:
    """
    Extract features from a fixed set of samples and save to disk.
    
    Parameters
    ----------
    model : nn.Module
        The model to extract features from
    dataloader : DataLoader
        DataLoader providing input samples
    save_path : str
        Path to save extracted features (.npz file)
    device : torch.device
        Device to run inference on
    num_samples : int
        Number of samples to extract features for
    layer_names : List[str], optional
        Specific layers to extract from
    """
    model.eval()
    extractor = FeatureExtractor(model, layer_names)
    
    all_features = {name: [] for name in extractor.layer_names}
    all_features['input'] = []
    all_features['output'] = []
    
    samples_collected = 0
    
    for batch in dataloader:
        if isinstance(batch, dict):
            # Handle dict-style batches from cellmap dataloader
            x = list(batch.values())[0].to(device)
        else:
            x = batch[0].to(device)
        
        features = extractor(x)
        
        for name, feat in features.items():
            arr = feat.cpu().numpy()
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            all_features[name].append(arr)
        
        samples_collected += x.shape[0]
        if samples_collected >= num_samples:
            break
    
    # Concatenate and save
    save_dict = {}
    for name, feat_list in all_features.items():
        if feat_list:
            concatenated = np.concatenate(feat_list, axis=0)[:num_samples]
            if concatenated.dtype != np.float32:
                concatenated = concatenated.astype(np.float32, copy=False)
            save_dict[name] = concatenated
    
    np.savez_compressed(save_path, **save_dict)
    print(f"Saved features to {save_path}")
    
    extractor.remove_hooks()


if __name__ == "__main__":
    # Test the feature extractor
    from cellmap_segmentation_challenge.models import UNet_2D, SwinTransformer
    
    # Test with UNet
    print("Testing with UNet_2D...")
    model = UNet_2D(1, 14)
    extractor = FeatureExtractor(model)
    
    x = torch.randn(1, 1, 256, 256)
    features = extractor(x)
    
    print("Extracted features from layers:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    extractor.remove_hooks()
    
    # Test with Swin
    print("\nTesting with SwinTransformer...")
    model = SwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        num_classes=14
    )
    extractor = FeatureExtractor(model)
    
    features = extractor(x)
    
    print("Extracted features from layers:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    extractor.remove_hooks()
