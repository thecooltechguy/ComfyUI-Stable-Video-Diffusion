import os
import folder_paths
import sys

svd_models_dir = os.path.join(folder_paths.models_dir, "svd")
os.makedirs(svd_models_dir, exist_ok=True)

SVD_MODEL_URLS = {
    "svd.safetensors" : "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors?download=true",
    "svd_image_decoder.safetensors" : "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd_image_decoder.safetensors?download=true",
    "svd_xt.safetensors" : "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors?download=true",
    "svd_xt_image_decoder.safetensors" : "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt_image_decoder.safetensors?download=true"
}

existing_svd_models = os.listdir(svd_models_dir)

for model_name, model_url in SVD_MODEL_URLS.items():
    if model_name not in existing_svd_models:
        print(f"Downloading {model_name}...")
        os.system(f'wget -O {os.path.join(svd_models_dir, model_name)} "{model_url}"')