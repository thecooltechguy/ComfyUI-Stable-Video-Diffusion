import folder_paths
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "libs"))

os.makedirs(os.path.join(folder_paths.models_dir, "svd"), exist_ok=True)
os.makedirs(os.path.join(folder_paths.models_dir, "svd_configs"), exist_ok=True)

folder_paths.add_model_folder_path("svd", os.path.join(folder_paths.models_dir, "svd"))
folder_paths.add_model_folder_path("svd_configs", os.path.join(folder_paths.models_dir, "svd_configs"))

svd_checkpoints = folder_paths.get_filename_list("svd")
svd_configs = folder_paths.get_filename_list("svd_configs")

if not svd_checkpoints:
    print("WARNING: No Stable Video Diffusion checkpoints found. Please download & place them in the models/svd folder, and restart ComfyUI.")

if not svd_configs:
    print("WARNING: No Stable Video Diffusion configs found. Please download & place them in the models/svd_configs folder, and restart ComfyUI.")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]