from .svd import load_model, get_unique_embedder_keys_from_conditioner, get_batch
import gc
import folder_paths
import torch
import os
import math

class SVDModelLoader:
    def __init__(self):
        self.svd_model = None
    
    @classmethod
    def INPUT_TYPES(s):
        checkpoints = folder_paths.get_filename_list("svd")
        configs = folder_paths.get_filename_list("svd_configs")

        devices = []
        if torch.cuda.is_available():
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

    CATEGORY = "Comfy Stable Video Diffusion"

    def load_svd_model(self, checkpoint, num_frames, num_steps, device):
        if self.svd_model is not None:
            del self.svd_model
            gc.collect()
            self.svd_model = None
        print("Loading SVD model...")
        checkpoint_filename_without_extension = os.path.splitext(checkpoint)[0]
        config = os.path.join(folder_paths.get_folder_paths("svd_configs")[0], f"{checkpoint_filename_without_extension}.yaml")
        checkpoint = os.path.join(folder_paths.get_folder_paths("svd")[0], checkpoint)
        self.svd_model = load_model(
            config=config,
            device=device,
            num_frames=num_frames,
            num_steps=num_steps,
            checkpoint=checkpoint,
        )
        print("Loaded SVD model!")
        return (self.svd_model,)

class SVDSampler:
    @classmethod
    def INPUT_TYPES(s):
        devices = []
        if torch.cuda.is_available():
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
                "decoding_t" : ("INT", {
                    "default": 14,
                }),
                "device" : (devices,),
            },
        }

    RETURN_TYPES = ("LATENT",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "sample_video"

    #OUTPUT_NODE = False

    CATEGORY = "Comfy Stable Video Diffusion"

    def sample_video(self, image, model, motion_bucket_id, fps_id, cond_aug, seed, decoding_t, device):
        # convert image tensor to PIL image
        print(type(image))
        print(image.shape)
        1/0

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
        if torch.cuda.is_available():
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

    CATEGORY = "Comfy Stable Video Diffusion"

    def decode(self, samples_z, model, decoding_t, device):
        with torch.no_grad():
            with torch.autocast(device):
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
        return (samples,)


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "SVDModelLoader" : SVDModelLoader,
    "SVDSampler": SVDSampler,
    "SVDDecoder": SVDDecoder,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDModelLoader" : "Load SVD Model",
    "SVDSampler": "SVD Sampler",
    "SVDDecoder": "SVD Decoder",
}