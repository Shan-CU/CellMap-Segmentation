# SwinTransformer Baseline Training Config
# Run with: python train_swin_baseline.py
# Monitor with: tensorboard --logdir tensorboard

from upath import UPath
from cellmap_segmentation_challenge.models import SwinTransformer
from cellmap_segmentation_challenge.utils import get_tested_classes

# %% Hyperparameters - IDENTICAL to UNet baseline for fair comparison
learning_rate = 0.0001
batch_size = 32
input_array_info = {
    "shape": (1, 256, 256),
    "scale": (8, 8, 8),
}
target_array_info = {
    "shape": (1, 256, 256),
    "scale": (8, 8, 8),
}
epochs = 10  # Same as UNet
iterations_per_epoch = 5000  # Same as UNet
random_seed = 42

# Use common classes that have good data coverage
# Full list causes EmptyImage errors due to some classes having no data
classes = [
    'ecs', 'pm', 'mito_mem', 'mito_lum', 'mito_ribo',
    'golgi_mem', 'golgi_lum', 'ves_mem', 'ves_lum',
    'endo_mem', 'endo_lum', 'er_mem', 'er_lum', 'nuc'
]
print(f"Training SwinTransformer baseline with {len(classes)} classes: {classes}")

# Model - SwinTransformer
model_name = "swin_baseline"
model_to_load = "swin_baseline"
model = SwinTransformer(
    patch_size=[4, 4],
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=[7, 7],
    num_classes=len(classes)
)

load_model = "latest"

# Paths
logs_save_path = UPath("tensorboard/{model_name}").path
model_save_path = UPath("checkpoints/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit.csv"

# Transforms - IDENTICAL to UNet baseline
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

validation_time_limit = 120
filter_by_scale = True

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)
