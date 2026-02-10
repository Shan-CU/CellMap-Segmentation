"""Debug script to check if dice metric is working correctly."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import torch
import numpy as np

# Load one of the saved checkpoints and run validation
from config_shenron import *
from train_local import compute_metrics, create_dataloaders, create_model

classes = QUICK_TEST_CLASSES
print(f"Classes: {classes}")

# Create val loader
print("\nCreating dataloaders...")
_, val_loader = create_dataloaders(
    classes=classes,
    batch_size=4,
    iterations_per_epoch=10,
    input_shape=(1, 256, 256),
)

# Get one batch
print("Getting validation batch...")
batch = next(iter(val_loader))
inputs = batch['input']
targets = batch['output']

print(f"\n{'='*60}")
print(f"INPUT shape: {inputs.shape}, dtype: {inputs.dtype}")
print(f"TARGET shape: {targets.shape}, dtype: {targets.dtype}")
print(f"{'='*60}")

# Analyze targets
print(f"\n--- TARGET ANALYSIS ---")
print(f"Target min: {targets.min():.4f}")
print(f"Target max: {targets.max():.4f}")
print(f"Target mean: {targets.mean():.4f}")
print(f"Target NaN count: {targets.isnan().sum().item()} / {targets.numel()}")
print(f"Target NaN fraction: {targets.isnan().float().mean():.4f}")

for i, c in enumerate(classes):
    t = targets[:, i]
    valid = ~t.isnan()
    t_clean = t[valid]
    print(f"\n  Class '{c}' (channel {i}):")
    print(f"    Shape: {t.shape}")
    print(f"    Valid pixels: {valid.sum().item()} / {t.numel()} ({valid.float().mean()*100:.1f}%)")
    if t_clean.numel() > 0:
        print(f"    Min: {t_clean.min():.4f}, Max: {t_clean.max():.4f}, Mean: {t_clean.mean():.4f}")
        print(f"    Unique values: {torch.unique(t_clean).tolist()[:20]}")
        print(f"    Positive (>0.5): {(t_clean > 0.5).sum().item()} / {t_clean.numel()} ({(t_clean > 0.5).float().mean()*100:.2f}%)")
        print(f"    Any ones: {(t_clean == 1.0).sum().item()}")
        print(f"    Any zeros: {(t_clean == 0.0).sum().item()}")
    else:
        print(f"    ALL NaN!")

# Now simulate what compute_metrics does with FAKE predictions
print(f"\n{'='*60}")
print("--- METRIC DEBUG WITH RANDOM MODEL OUTPUT ---")
print(f"{'='*60}")

# Simulate random model output (like an untrained model)
fake_pred = torch.randn_like(targets) * 0.3  # Small random logits

pred_sigmoid = torch.sigmoid(fake_pred)
print(f"\nFake sigmoid stats: min={pred_sigmoid.min():.4f}, max={pred_sigmoid.max():.4f}, mean={pred_sigmoid.mean():.4f}")

# Test with different thresholds
for thresh in [0.3, 0.5]:
    pred_binary = (pred_sigmoid > thresh).float()
    valid_mask = ~targets.isnan()
    target_clean = targets.nan_to_num(0)
    
    print(f"\n--- Threshold = {thresh} ---")
    print(f"Pred positive fraction: {pred_binary.mean():.4f}")
    
    for c_idx in range(len(classes)):
        pred_c = pred_binary[:, c_idx] * valid_mask[:, c_idx]
        target_c = target_clean[:, c_idx] * valid_mask[:, c_idx]
        
        intersection = (pred_c * target_c).sum()
        pred_sum = pred_c.sum()
        target_sum = target_c.sum()
        
        dice = (2 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)
        
        print(f"  {classes[c_idx]:12s}: intersection={intersection:.0f}, pred_sum={pred_sum:.0f}, target_sum={target_sum:.0f}, dice={dice:.6f}")

# Now test with ACTUAL model checkpoint
print(f"\n{'='*60}")
print("--- METRIC DEBUG WITH ACTUAL MODEL ---")
print(f"{'='*60}")

# Find a checkpoint
ckpt_dir = CHECKPOINT_DIR
ckpts = sorted(ckpt_dir.glob("*.pt"))
print(f"\nFound {len(ckpts)} checkpoints:")
for cp in ckpts[:10]:
    print(f"  {cp.name} ({cp.stat().st_size / 1e6:.1f} MB)")

if ckpts:
    # Load the first checkpoint
    ckpt = torch.load(ckpts[0], map_location='cpu', weights_only=False)
    print(f"\nLoaded: {ckpts[0].name}")
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    if 'loss_name' in ckpt:
        print(f"Loss: {ckpt['loss_name']}")
    if 'epoch' in ckpt:
        print(f"Epoch: {ckpt['epoch']}")
    
    # Create model and load weights
    model = create_model(len(classes), input_channels=1)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    model.eval()
    
    # Run inference
    with torch.no_grad():
        # Handle input dims
        inp = inputs
        if inp.dim() == 5 and inp.shape[1] == 1:
            inp = inp.squeeze(1)
        
        outputs = model(inp)
        print(f"\nModel output shape: {outputs.shape}")
        
        sigmoid_out = torch.sigmoid(outputs)
        print(f"Sigmoid min={sigmoid_out.min():.4f}, max={sigmoid_out.max():.4f}, mean={sigmoid_out.mean():.4f}")
        
        for i, c in enumerate(classes):
            vals = sigmoid_out[:, i].flatten()
            print(f"  {c}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}, >0.3: {(vals > 0.3).float().mean()*100:.1f}%, >0.5: {(vals > 0.5).float().mean()*100:.1f}%")
        
        # Compute metrics
        metrics = compute_metrics(outputs, targets)
        print(f"\ncompute_metrics result:")
        print(f"  dice_mean: {metrics['dice_mean']:.6f}")
        for i, c in enumerate(classes):
            print(f"  {c}: {metrics['dice_per_class'][i]:.6f}")
        
        # Manual check with thresh=0.3
        pred_binary_03 = (sigmoid_out > 0.3).float()
        pred_binary_05 = (sigmoid_out > 0.5).float()
        valid_mask = ~targets.isnan()
        target_clean = targets.nan_to_num(0)
        
        print(f"\n--- Manual dice check ---")
        for c_idx in range(len(classes)):
            for thresh, pb in [('0.3', pred_binary_03), ('0.5', pred_binary_05)]:
                pred_c = pb[:, c_idx] * valid_mask[:, c_idx]
                target_c = target_clean[:, c_idx] * valid_mask[:, c_idx]
                
                intersection = (pred_c * target_c).sum()
                pred_sum = pred_c.sum()
                target_sum = target_c.sum()
                
                dice = (2 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)
                
                print(f"  {classes[c_idx]:12s} @{thresh}: inter={intersection:.0f}, pred={pred_sum:.0f}, tgt={target_sum:.0f}, dice={dice:.6f}")

print("\nDone!")
