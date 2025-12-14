import mlx.core as mx
import gc
from typing import Optional, Dict, Any

class GlobalModelRegistry:
    """Singleton-based Global Model Registry for MLX VLM models.

    Prevents redundant model loading and manages memory efficiently
    on Apple Silicon devices with Unified Memory Architecture.
    """

    _instance = None
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalModelRegistry, cls).__new__(cls)
        return cls._instance

    def get_model(self, model_id: str, quantization: str = "default") -> Optional[Dict[str, Any]]:
        """Retrieve a model from the registry if it matches the requested configuration."""
        key = f"{model_id}_{quantization}"
        return self._models.get(key)

    def register_model(self, model_id: str, quantization: str, model_bundle: Dict[str, Any]):
        """Register a loaded model in the registry."""
        key = f"{model_id}_{quantization}"
        # Clear existing model if present to prevent memory thrashing
        if key in self._models:
            self.clear_model(model_id, quantization)

        self._models[key] = model_bundle

    def clear_model(self, model_id: str, quantization: str = "default"):
        """Clear a specific model from memory and registry."""
        key = f"{model_id}_{quantization}"
        if key in self._models:
            # Explicitly delete the model references
            model_bundle = self._models[key]
            del model_bundle['model']  # Delete model weights
            del model_bundle  # Delete the bundle
            del self._models[key]

            # Clear MLX cache and force garbage collection
            mx.metal.clear_cache()
            gc.collect()

    def clear_all(self):
        """Clear all models from memory and registry."""
        for key in list(self._models.keys()):
            model_id, quantization = key.rsplit('_', 1)
            self.clear_model(model_id, quantization)

    def get_registry_info(self) -> Dict[str, Any]:
        """Get information about currently loaded models."""
        return {
            "loaded_models": list(self._models.keys()),
            "model_count": len(self._models)
        }

# Global instance
global_registry = GlobalModelRegistry()
