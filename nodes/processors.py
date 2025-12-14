import torch
from PIL import Image
from typing import List, Optional, Dict, Any
from ..utils.dtype_bridge import torch_tensor_to_pil_images, resize_image_with_max_pixels
from ..utils.omni_utils import extract_video_frames, validate_audio_file, create_audio_prompt_dict

class MLX_VLM_ImageProcessor:
    """Process ComfyUI images for MLX VLM input with dynamic resolution scaling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "resize_strategy": (["none", "max_pixels", "fixed_size"], {
                    "default": "max_pixels"
                }),
                "max_pixels": ("INT", {
                    "default": 1024 * 1024,  # 1MP default
                    "min": 10000,
                    "max": 4000000,  # 4MP max
                    "step": 10000
                })
            }
        }

    RETURN_TYPES = ("MLX_IMAGE_BATCH",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process_images"
    CATEGORY = "MLX_VLM/Processors"

    def process_images(self, images, resize_strategy: str, max_pixels: int):
        """Convert ComfyUI images to PIL images for MLX VLM processing."""

        # Convert torch tensors to PIL images
        pil_images = torch_tensor_to_pil_images(images)

        # Apply resize strategy
        if resize_strategy == "max_pixels":
            processed_images = []
            for pil_image in pil_images:
                resized_image = resize_image_with_max_pixels(pil_image, max_pixels)
                processed_images.append(resized_image)
            pil_images = processed_images
        elif resize_strategy == "fixed_size":
            # Could implement fixed size resizing here
            pass
        # "none" strategy leaves images unchanged

        print(f"Processed {len(pil_images)} images for MLX VLM input")
        return (pil_images,)

class MLX_VLM_VideoLoader:
    """Extract frames from video files for MLX VLM processing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to video file"
                }),
                "frame_extraction": (["uniform", "keyframe"], {
                    "default": "uniform"
                }),
                "num_frames": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 32,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("MLX_IMAGE_BATCH", "INT")
    RETURN_NAMES = ("frames", "frame_count")
    FUNCTION = "load_video"
    CATEGORY = "MLX_VLM/Processors"

    def load_video(self, video_path: str, frame_extraction: str, num_frames: int):
        """Extract video frames and return as MLX image batch."""

        try:
            # Extract frames using omni utilities
            frames = extract_video_frames(
                video_path=video_path,
                num_frames=num_frames,
                mode=frame_extraction
            )

            print(f"Extracted {len(frames)} frames from video: {video_path}")
            return (frames, len(frames))

        except Exception as e:
            error_msg = f"Failed to load video {video_path}: {str(e)}"
            raise RuntimeError(error_msg)

class MLX_VLM_AudioProcessor:
    """Process audio inputs for Omni models that support audio understanding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to audio file"
                })
            },
            "optional": {
                "segment_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "step": 0.1,
                    "display": "seconds"
                }),
                "segment_end": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "step": 0.1,
                    "display": "seconds"
                })
            }
        }

    RETURN_TYPES = ("MLX_AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_audio"
    CATEGORY = "MLX_VLM/Processors"

    def process_audio(self, audio_path: str, segment_start: float = 0.0, segment_end: float = 0.0):
        """Validate and prepare audio file for MLX VLM processing."""

        # Validate audio file
        if not validate_audio_file(audio_path):
            raise ValueError(f"Invalid or unsupported audio file: {audio_path}")

        # Create audio prompt dictionary
        audio_dict = create_audio_prompt_dict(audio_path)

        # Handle segment information (for future implementation)
        if segment_end > segment_start > 0:
            audio_dict["segment"] = {
                "start": segment_start,
                "end": segment_end
            }

        print(f"Processed audio file: {audio_path}")
        return (audio_dict,)
