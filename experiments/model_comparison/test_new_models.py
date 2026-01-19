#!/usr/bin/env python3
"""Quick test of the new model implementations."""

import torch
import sys
sys.path.insert(0, '/home/spuser/ws/CellMap-Segmentation')

from src.cellmap_segmentation_challenge.models import SwinTransformer3D, ViTVNet2D

def test_swin3d():
    print('Testing SwinTransformer3D...')
    model_3d = SwinTransformer3D(
        patch_size=[2, 4, 4],
        embed_dim=48,  # Smaller for testing
        depths=[2, 2, 2, 2],
        num_heads=[2, 4, 8, 16],
        window_size=[4, 7, 7],
        num_classes=14
    )
    x_3d = torch.randn(1, 1, 32, 64, 64)  # Smaller for testing
    print('  Input shape:', x_3d.shape)
    with torch.no_grad():
        y_3d = model_3d(x_3d)
    print('  Output shape:', y_3d.shape)
    params_3d = sum(p.numel() for p in model_3d.parameters())
    print('  Parameters:', params_3d)
    return True

def test_vit2d():
    print('\nTesting ViTVNet2D...')
    config = {
        'img_size': 64,
        'patch_size': 8,
        'hidden_size': 192,
        'num_layers': 4,
        'num_heads': 4,
        'mlp_dim': 768,
        'decoder_channels': (128, 64, 32, 16),
        'dropout_rate': 0.1,
        'attention_dropout_rate': 0.0,
        'down_factor': 2,
    }
    model_2d = ViTVNet2D(config=config, in_channels=1, num_classes=14)
    x_2d = torch.randn(1, 1, 64, 64)
    print('  Input shape:', x_2d.shape)
    with torch.no_grad():
        y_2d = model_2d(x_2d)
    print('  Output shape:', y_2d.shape)
    params_2d = sum(p.numel() for p in model_2d.parameters())
    print('  Parameters:', params_2d)
    return True

if __name__ == '__main__':
    try:
        test_swin3d()
        test_vit2d()
        print('\n✓ Both models working correctly!')
    except Exception as e:
        print(f'\n✗ Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
