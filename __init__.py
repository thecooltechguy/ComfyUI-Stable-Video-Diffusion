import folder_paths
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "libs"))

os.makedirs(os.path.join(folder_paths.models_dir, "svd"), exist_ok=True)

folder_paths.add_model_folder_path("svd", os.path.join(folder_paths.models_dir, "svd"))

svd_checkpoints = folder_paths.get_filename_list("svd")
svd_configs = os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "svd_configs"))

assert len(svd_checkpoints) > 0, "ERROR: No Stable Video Diffusion checkpoints found. Please download & place them in the ComfyUI/models/svd folder, and restart ComfyUI."
assert len(svd_configs) > 0

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]