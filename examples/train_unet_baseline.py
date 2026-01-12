# UNet Baseline Training Config
# Run with: python train_unet_baseline.py
# Monitor with: tensorboard --logdir tensorboard

# Monkey-patch to fix cellmap_data bug (EmptyImage doesn't accept 'device' param)
import cellmap_data.empty_image as _ei
_orig_init = _ei.EmptyImage.__init__
def _patched_init(self, *args, device=None, **kwargs):
    _orig_init(self, *args, **kwargs)
_ei.EmptyImage.__init__ = _patched_init

from upath import UPath
from cellmap_segmentation_challenge.models import UNet_2D
from cellmap_segmentation_challenge.utils import get_tested_classes

# %% Hyperparameters
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
epochs = 10  # Full training run
iterations_per_epoch = 5000  # 5000 iterations per epoch
random_seed = 42

# Use common classes that have good data coverage
# Full list causes EmptyImage errors due to some classes having no data
classes = [
    'ecs', 'pm', 'mito_mem', 'mito_lum', 'mito_ribo',
    'golgi_mem', 'golgi_lum', 'ves_mem', 'ves_lum',
    'endo_mem', 'endo_lum', 'er_mem', 'er_lum', 'nuc'
]
print(f"Training UNet baseline with {len(classes)} classes: {classes}")

# Model - UNet baseline
model_name = "unet_baseline"
model_to_load = "unet_baseline"
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
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

validation_time_limit = 120
filter_by_scale = True

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)
