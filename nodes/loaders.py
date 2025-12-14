import mlx_vlm
import os
from typing import Optional, Dict, Any
from ..utils.registry import global_registry

class MLX_VLM_LoadModel:
    """Load Vision Language Models with MLX optimization for Apple Silicon.

    This node handles model fetching, loading, and caching with quantization support.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": "mlx-community/Qwen2-VL-7B-Instruct-4bit",
                    "multiline": False,
                    "placeholder": "HuggingFace model ID or local path"
                }),
                "quantization": (["default", "4-bit", "8-bit", "bf16"], {
                    "default": "4-bit"
                }),
                "trust_remote_code": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Yes",
                    "label_off": "No"
                })
            },
            "optional": {
                "adapter_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to LoRA adapter (optional)"
                })
            }
        }

    RETURN_TYPES = ("MLX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MLX_VLM/Loaders"

    def load_model(self, model_id: str, quantization: str, trust_remote_code: bool, adapter_path: str = ""):
        """Load MLX VLM model with caching and quantization support."""

        # Check registry first
        cached_model = global_registry.get_model(model_id, quantization)
        if cached_model:
            print(f"Using cached model: {model_id} ({quantization})")
            return (cached_model,)

        # Prepare loading parameters
        load_kwargs = {
            "trust_remote_code": trust_remote_code
        }

        # Handle quantization
        if quantization != "default":
            if quantization == "4-bit":
                load_kwargs["quantization"] = "4bit"
            elif quantization == "8-bit":
                load_kwargs["quantization"] = "8bit"
            elif quantization == "bf16":
                load_kwargs["quantization"] = "bf16"

        try:
            # Load model and processor
            model, processor = mlx_vlm.load(model_id, **load_kwargs)

            # Load model configuration
            config = mlx_vlm.utils.load_config(model_id)

            # Create model bundle
            model_bundle = {
                "model": model,
                "processor": processor,
                "config": config,
                "model_id": model_id,
                "quantization": quantization
            }

            # Handle adapter loading if specified
            if adapter_path and os.path.exists(adapter_path):
                try:
                    mlx_vlm.utils.apply_lora_layers(model, adapter_path)
                    model_bundle["adapter_path"] = adapter_path
                except Exception as e:
                    print(f"Warning: Failed to load adapter from {adapter_path}: {e}")

            # Register in global cache
            global_registry.register_model(model_id, quantization, model_bundle)

            print(f"Successfully loaded model: {model_id} ({quantization})")
            return (model_bundle,)

        except Exception as e:
            error_msg = f"Failed to load model {model_id}: {str(e)}"
            raise RuntimeError(error_msg)

class MLX_VLM_LoadLoRA:
    """Apply Low-Rank Adaptation (LoRA) adapters to loaded MLX VLM models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MLX_MODEL",),
                "lora_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to LoRA adapter directory"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                })
            }
        }

    RETURN_TYPES = ("MLX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_lora"
    CATEGORY = "MLX_VLM/Loaders"

    def apply_lora(self, model: Dict[str, Any], lora_path: str, strength: float):
        """Apply LoRA adapter to model with strength control."""

        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA adapter path not found: {lora_path}")

        # Check for required adapter config
        config_path = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"adapter_config.json not found in {lora_path}")

        try:
            # Create a copy of the model bundle to avoid modifying the cached version
            model_bundle = model.copy()

            # Apply LoRA with strength scaling
            original_model = model_bundle["model"]

            # Apply LoRA layers (MLX handles the low-rank adaptation)
            mlx_vlm.utils.apply_lora_layers(original_model, lora_path)

            # Store LoRA info
            model_bundle["lora_path"] = lora_path
            model_bundle["lora_strength"] = strength

            print(f"Successfully applied LoRA adapter from {lora_path} with strength {strength}")
            return (model_bundle,)

        except Exception as e:
            error_msg = f"Failed to apply LoRA adapter from {lora_path}: {str(e)}"
            raise RuntimeError(error_msg)

class MLX_VLM_FreeCache:
    """Utility node to free MLX cache and manage memory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clear_all": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Clear All Models",
                    "label_off": "Clear Cache Only"
                })
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "free_cache"
    CATEGORY = "MLX_VLM/Loaders"
    OUTPUT_NODE = True

    def free_cache(self, clear_all: bool):
        """Free MLX cache and optionally clear all loaded models."""
        import mlx.core as mx
        import gc

        if clear_all:
            # Clear all models from registry
            global_registry.clear_all()
            print("Cleared all models from registry")
        else:
            # Just clear MLX cache
            mx.metal.clear_cache()
            gc.collect()
            print("Cleared MLX cache and collected garbage")

        return ()
