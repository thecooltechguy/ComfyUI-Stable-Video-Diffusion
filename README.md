# ComfyUI Stable Video Diffusion
Easily use Stable Video Diffusion inside ComfyUI!

[![](https://dcbadge.vercel.app/api/server/MfVCahkc2y)](https://discord.gg/MfVCahkc2y)

## Example workflows

### Image to video
[https://comfyworkflows.com/workflows/5a4cd9fd-9685-4985-adb8-7be84e8636ad](https://comfyworkflows.com/workflows/5a4cd9fd-9685-4985-adb8-7be84e8636ad)

![workflow graph](./svd_workflow_graph.png)
![sample output](./svd_workflow.gif)


### Image to video generation (high FPS w/ frame interpolation)
[https://comfyworkflows.com/workflows/bf3b455d-ba13-4063-9ab7-ff1de0c9fa75](https://comfyworkflows.com/workflows/bf3b455d-ba13-4063-9ab7-ff1de0c9fa75)

![workflow graph](./svd_rife_workflow_graph.png)
![sample output](./svd_rife.gif)

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

## Need help?
Join our Discord! https://discord.gg/MfVCahkc2y