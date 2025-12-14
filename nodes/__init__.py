# Node modules for ComfyUI-MLX-VLM

from .loaders import MLX_VLM_LoadModel, MLX_VLM_LoadLoRA, MLX_VLM_FreeCache
from .processors import MLX_VLM_ImageProcessor, MLX_VLM_VideoLoader, MLX_VLM_AudioProcessor
from .sampler import MLX_VLM_ChatTemplate, MLX_VLM_Sampler

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    # Loaders
    "MLX_VLM_LoadModel": MLX_VLM_LoadModel,
    "MLX_VLM_LoadLoRA": MLX_VLM_LoadLoRA,
    "MLX_VLM_FreeCache": MLX_VLM_FreeCache,

    # Processors
    "MLX_VLM_ImageProcessor": MLX_VLM_ImageProcessor,
    "MLX_VLM_VideoLoader": MLX_VLM_VideoLoader,
    "MLX_VLM_AudioProcessor": MLX_VLM_AudioProcessor,

    # Sampler
    "MLX_VLM_ChatTemplate": MLX_VLM_ChatTemplate,
    "MLX_VLM_Sampler": MLX_VLM_Sampler,
}

# Display names for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    # Loaders
    "MLX_VLM_LoadModel": "MLX VLM Load Model",
    "MLX_VLM_LoadLoRA": "MLX VLM Load LoRA",
    "MLX_VLM_FreeCache": "MLX VLM Free Cache",

    # Processors
    "MLX_VLM_ImageProcessor": "MLX VLM Image Processor",
    "MLX_VLM_VideoLoader": "MLX VLM Video Loader",
    "MLX_VLM_AudioProcessor": "MLX VLM Audio Processor",

    # Sampler
    "MLX_VLM_ChatTemplate": "MLX VLM Chat Template",
    "MLX_VLM_Sampler": "MLX VLM Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
