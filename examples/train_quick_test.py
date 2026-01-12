# Quick test config - verify pipeline works (5 minutes)
# Run with: python train_quick_test.py

from upath import UPath
from cellmap_segmentation_challenge.models import UNet_2D
from cellmap_segmentation_challenge.utils import get_tested_classes

# %% Hyperparameters - REDUCED FOR QUICK TEST
learning_rate = 0.0001
batch_size = 8  # smaller batch for quick test
input_array_info = {
    "shape": (1, 256, 256),
    "scale": (8, 8, 8),
}
target_array_info = {
    "shape": (1, 256, 256),
    "scale": (8, 8, 8),
}
epochs = 1  # just 1 epoch
iterations_per_epoch = 50  # very few iterations
random_seed = 42

# Use subset of classes for quick test
classes = get_tested_classes()[:5]  # only first 5 classes
print(f"Quick test with {len(classes)} classes: {classes}")

# Model
model_name = "quick_test_unet"
model_to_load = "quick_test_unet"
model = UNet_2D(1, len(classes))

load_model = "latest"

# Paths
logs_save_path = UPath("tensorboard/{model_name}").path
model_save_path = UPath("checkpoints/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit.csv"

# Transforms
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
}

validation_time_limit = 30
filter_by_scale = True

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)
