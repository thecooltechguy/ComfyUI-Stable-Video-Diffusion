from .svd import load_model
import gc
import folder_paths
import torch
import os

class StableVideoDiffusion:
    """
    Node for applying Stable Video Diffusion

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        self.svd_model = None
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "model" : ("MODEL",),
                # "int_field": ("INT", {
                #     "default": 0, 
                #     "min": 0, #Minimum value
                #     "max": 4096, #Maximum value
                #     "step": 64, #Slider's step
                #     "display": "number" # Cosmetic only: display as "number" or "slider"
                # }),
                # "float_field": ("FLOAT", {
                #     "default": 1.0,
                #     "min": 0.0,
                #     "max": 10.0,
                #     "step": 0.01,
                #     "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                #     "display": "number"}),
                # "print_to_screen": (["enable", "disable"],),
                # "string_field": ("STRING", {
                #     "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                #     "default": "Hello World!"
                # }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "sample_video"

    #OUTPUT_NODE = False

    CATEGORY = "Comfy Stable Video Diffusion"

    def sample_video(self, image, model):
        print(type(image))
        print(image)
        print(type(model))
        print(model)
        return (image,)


class SVDModelLoader:
    """
    Node for applying Stable Video Diffusion

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        self.svd_model = None
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
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
                "config" : (configs, {
                    "default" : configs[0],
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

    RETURN_TYPES = ("MODEL", "CLIP")
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "load_svd_model"

    CATEGORY = "Comfy Stable Video Diffusion"

    def load_svd_model(self, checkpoint, config, num_frames, num_steps, device):
        if self.svd_model is not None:
            del self.svd_model
            gc.collect()
            self.svd_model = None
        print("Loading model...")
        config = os.path.join(folder_paths.get_folder_paths("svd_configs")[0], config)
        checkpoint = os.path.join(folder_paths.get_folder_paths("svd")[0], checkpoint)
        self.svd_model = load_model(
            config=config,
            device=device,
            num_frames=num_frames,
            num_steps=num_steps,
            checkpoint=checkpoint,
        )
        conditioner = self.svd_model.conditioner
        clip_model = conditioner.embedders[0].open_clip.model
        print("Model loaded!")
        return (self.svd_model, clip_model)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "SVDModelLoader" : SVDModelLoader,
    "StableVideoDiffusion": StableVideoDiffusion
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDModelLoader" : "Load Stable Video Diffusion Model",
    "StableVideoDiffusion": "Stable Video Diffusion"
}