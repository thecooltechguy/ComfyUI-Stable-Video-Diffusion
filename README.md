# ComfyUI Stable Video Diffusion
Easily use Stable Video Diffusion inside ComfyUI!

## Examples

### Image to video generation



### Image to video generation (with frame interpolation for high FPS)


## Installation
```
cd custom_nodes/
git clone https://github.com/thecooltechguy/ComfyUI-Stable-Video-Diffusion
cd ComfyUI-Stable-Video-Diffusion/
pip install -r requirements.txt
```

### Models
#### Download & place these files in `ComfyUI/models/svd/`
 - svd.safetensors - [Download](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors?download=true)
 - svd_image_decoder.safetensors - [Download](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd_image_decoder.safetensors?download=true)
  - svd_xt.safetensors - [Download](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors?download=true)
 - svd_xt_image_decoder.safetensors - [Download](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt_image_decoder.safetensors?download=true)

### Configs
#### Download & place these files in `ComfyUI/models/svd_configs/`
 - svd.yaml - [Download](https://raw.githubusercontent.com/Stability-AI/generative-models/main/scripts/sampling/configs/svd.yaml)
 - svd_image_decoder.yaml - [Download](https://raw.githubusercontent.com/Stability-AI/generative-models/main/scripts/sampling/configs/svd_image_decoder.yaml)
  - svd_xt.yaml - [Download](https://raw.githubusercontent.com/Stability-AI/generative-models/main/scripts/sampling/configs/svd_xt.yaml)
 - svd_xt_image_decoder.yaml - [Download](https://raw.githubusercontent.com/Stability-AI/generative-models/main/scripts/sampling/configs/svd_xt_image_decoder.yaml)

