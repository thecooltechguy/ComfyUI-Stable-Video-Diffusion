from .svd import load_model, get_unique_embedder_keys_from_conditioner, get_batch
from torchvision.transforms import ToTensor, ToPILImage
from einops import rearrange, repeat
import gc
import folder_paths
import torch
import os
import math
import numpy as np

class SVDModelLoader:
    def __init__(self):
        self.svd_model = None
        
    @classmethod
    def INPUT_TYPES(s):
        checkpoints = folder_paths.get_filename_list("svd")

        devices = []
        if True: #torch.cuda.is_available():
            devices.append("cuda")
        devices.append("cpu")

        return {
            "required": {
                "checkpoint" : (checkpoints, {
                    "default" : checkpoints[0],
                }),
                "num_frames" : ("INT", {
                    "default": 14,
                    "min" : 0,
                }),
                "num_steps" : ("INT", {
                    "default" : 25,
                    "min" : 0,
                }),
                "device" : (devices,),
            },
        }

    RETURN_TYPES = ("MODEL",)

    FUNCTION = "load_svd_model"

    CATEGORY = "ComfyUI Stable Video Diffusion"

    def load_svd_model(self, checkpoint, num_frames, num_steps, device):
        if self.svd_model is not None:
            del self.svd_model
            gc.collect()
            self.svd_model = None
        checkpoint_filename_without_extension = os.path.splitext(os.path.basename(checkpoint))[0]
        config = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "svd_configs"), f"{checkpoint_filename_without_extension}.yaml")
        checkpoint = os.path.join(folder_paths.get_folder_paths("svd")[0], checkpoint)
        self.svd_model = load_model(
            config=config,
            device=device,
            num_frames=num_frames,
            num_steps=num_steps,
            checkpoint=checkpoint,
        )
        return (self.svd_model,)

class SVDSampler:
    @classmethod
    def INPUT_TYPES(s):
        devices = []
        if True: #torch.cuda.is_available():
            devices.append("cuda")
        devices.append("cpu")
        return {
            "required": {
                "image": ("IMAGE",),
                "model" : ("MODEL",),
                "motion_bucket_id" : ("INT", {
                    "default": 127,
                }),
                "fps_id" : ("INT", {
                    "default": 6,
                }),
                "cond_aug" : ("FLOAT", {
                    "default": 0.02,
                }),
                "seed" : ("INT", {
                    "default": 23,
                }),
                "device" : (devices,),
            },
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "sample_video"

    CATEGORY = "ComfyUI Stable Video Diffusion"

    def sample_video(self, image, model, motion_bucket_id, fps_id, cond_aug, seed, device):
        # convert image torch tensor to PIL image
        # image shape: (1, H, W, C)
        image = image.squeeze(0)
        image = image.permute(2, 0, 1)

        image = ToPILImage()(image)

        if image.mode == "RGBA":
            image = image.convert("RGB")
        
        w, h = image.size

        if h % 64 != 0 or w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (w, h))
            image = image.resize((width, height))
            print(
                f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
            )

        image = ToTensor()(image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        num_frames = model.sampler.guider.num_frames
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024):
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug

        all_samples_z = []

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )

                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
        
        return (samples_z,)


class SVDDecoder:
    @classmethod
    def INPUT_TYPES(s):
        devices = []
        if True: #torch.cuda.is_available():
            devices.append("cuda")
        devices.append("cpu")
        return {
            "required": {
                "samples_z": ("LATENT",),
                "model" : ("MODEL",),
                "decoding_t" : ("INT", {
                    "default": 14,
                }),
                "device" : (devices,),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "decode"

    CATEGORY = "ComfyUI Stable Video Diffusion"

    def decode(self, samples_z, model, decoding_t, device):
        with torch.no_grad():
            with torch.autocast(device):
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 1)
                    .cpu()
                )
        return (vid,)

class SVDSimpleImg2Vid:
    def __init__(self):
        self.svd_model = None
        self.svd_config = None
        self.device = None
        self.num_frames = None
        self.num_steps = None
        self.checkpoint = None

    def is_model_config_different(self, config, device, num_frames, num_steps, checkpoint):
        return self.svd_config != config or self.device != device or self.num_frames != num_frames or self.num_steps != num_steps or self.checkpoint != checkpoint
    
    """
    Combines the SVDModelLoader, SVDSampler, and SVDDecoder nodes into one node.
    """
    @classmethod
    def INPUT_TYPES(s):
        checkpoints = folder_paths.get_filename_list("svd")
        configs = os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "svd_configs"))

        devices = []
        if True: #torch.cuda.is_available():
            devices.append("cuda")
        devices.append("cpu")

        return {
            "required": {
                "image": ("IMAGE",),
                "checkpoint" : (checkpoints, {
                    "default" : checkpoints[0],
                }),
                "num_frames" : ("INT", {
                    "default": 14,
                    "min" : 0,
                }),
                "num_steps" : ("INT", {
                    "default" : 25,
                    "min" : 0,
                }),

                "motion_bucket_id" : ("INT", {
                    "default": 127,
                }),
                "fps_id" : ("INT", {
                    "default": 6,
                }),
                "cond_aug" : ("FLOAT", {
                    "default": 0.02,
                }),
                "seed" : ("INT", {
                    "default": 23,
                }),

                "decoding_t" : ("INT", {
                    "default": 14,
                }),

                "device" : (devices,),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "img2vid"

    CATEGORY = "ComfyUI Stable Video Diffusion"

    def img2vid(self, image, checkpoint, num_frames, num_steps, motion_bucket_id, fps_id, cond_aug, seed, decoding_t, device):
        checkpoint_filename_without_extension = os.path.splitext(os.path.basename(checkpoint))[0]
        config = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "svd_configs"), f"{checkpoint_filename_without_extension}.yaml")
        checkpoint = os.path.join(folder_paths.get_folder_paths("svd")[0], checkpoint)
        
        if self.svd_model is None or self.is_model_config_different(config, device, num_frames, num_steps, checkpoint):
            if self.svd_model:
                del self.svd_model
                gc.collect()
                self.svd_model = None
            
            self.svd_model = load_model(
                config=config,
                device=device,
                num_frames=num_frames,
                num_steps=num_steps,
                checkpoint=checkpoint,
            )
            self.svd_config = config
            self.device = device
            self.num_frames = num_frames
            self.num_steps = num_steps
            self.checkpoint = checkpoint

        model = self.svd_model

        # convert image torch tensor to PIL image
        # image shape: (1, H, W, C)
        image = image.squeeze(0)
        image = image.permute(2, 0, 1)

        image = ToPILImage()(image)

        if image.mode == "RGBA":
            image = image.convert("RGB")
        
        w, h = image.size

        if h % 64 != 0 or w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (w, h))
            image = image.resize((width, height))
            print(
                f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
            )

        image = ToTensor()(image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        num_frames = self.svd_model.sampler.guider.num_frames
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024):
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug

        all_samples_z = []

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )

                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)

                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 1)
                    .cpu()
                )
        return (vid,)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "SVDModelLoader" : SVDModelLoader,
    "SVDSampler": SVDSampler,
    "SVDDecoder": SVDDecoder,
    "SVDSimpleImg2Vid": SVDSimpleImg2Vid,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDModelLoader" : "Load Stable Video Diffusion Model",
    "SVDSampler": "Stable Video Diffusion Sampler",
    "SVDDecoder": "Stable Video Diffusion Decoder",
    "SVDSimpleImg2Vid": "Stable Video Diffusion Simple Img2Vid",
}