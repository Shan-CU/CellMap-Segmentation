# 3D UNet training config optimized for RTX 4090 (24GB VRAM)
from upath import UPath
from cellmap_segmentation_challenge.models import UNet_3D
from cellmap_segmentation_challenge.utils import get_tested_classes

learning_rate = 0.0001
batch_size = 1

input_array_info = {
    "shape": (32, 128, 128),
    "scale": (8, 8, 8),
}
target_array_info = {
    "shape": (32, 128, 128),
    "scale": (8, 8, 8),
}

epochs = 100
iterations_per_epoch = 500
random_seed = 42

classes = get_tested_classes()

model_name = "3d_unet_4090"
model_to_load = "3d_unet_4090"
model = UNet_3D(1, len(classes))

load_model = "latest"

logs_save_path = UPath("tensorboard/{model_name}").path
model_save_path = UPath("checkpoints/{model_name}_{epoch}.pth").path

# USE S3 STREAMING - bypasses corrupted local data
use_s3 = True

spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.1}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

validation_time_limit = 60
filter_by_scale = True