import torch
import numpy as np
from PIL import Image
import mlx.core as mx
from typing import List, Union, Optional
import tempfile
import os

def torch_tensor_to_pil_images(torch_tensor) -> List[Image.Image]:
    """Convert ComfyUI torch.Tensor images to PIL Images for MLX processing.

    Args:
        torch_tensor: torch.Tensor with shape [B, H, W, C] and values [0.0, 1.0]

    Returns:
        List of PIL.Image objects
    """
    # Ensure tensor is on CPU and detached from computation graph
    if torch_tensor.is_cuda:
        torch_tensor = torch_tensor.cpu()
    if torch_tensor.requires_grad:
        torch_tensor = torch_tensor.detach()

    # Convert to numpy
    numpy_array = torch_tensor.numpy()

    # Handle different tensor shapes
    if len(numpy_array.shape) == 3:
        # Single image [H, W, C]
        numpy_array = np.expand_dims(numpy_array, axis=0)

    pil_images = []
    for img_array in numpy_array:
        # Convert from [0.0, 1.0] to [0, 255] and cast to uint8
        img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
        # Convert to PIL Image
        pil_image = Image.fromarray(img_array)
        pil_images.append(pil_image)

    return pil_images

def pil_images_to_torch_tensor(pil_images: List[Image.Image]) -> torch.Tensor:
    """Convert PIL Images back to ComfyUI torch.Tensor format.

    Args:
        pil_images: List of PIL.Image objects

    Returns:
        torch.Tensor with shape [B, H, W, C] and values [0.0, 1.0]
    """
    numpy_arrays = []
    for pil_image in pil_images:
        # Convert PIL to numpy array
        img_array = np.array(pil_image)
        # Normalize from [0, 255] to [0.0, 1.0]
        img_array = img_array.astype(np.float32) / 255.0
        numpy_arrays.append(img_array)

    # Stack into batch dimension
    batch_array = np.stack(numpy_arrays, axis=0)
    # Convert to torch tensor
    torch_tensor = torch.from_numpy(batch_array)

    return torch_tensor

def resize_image_with_max_pixels(pil_image: Image.Image, max_pixels: int) -> Image.Image:
    """Resize image to fit within maximum pixel constraint while maintaining aspect ratio.

    Args:
        pil_image: PIL Image to resize
        max_pixels: Maximum number of pixels allowed

    Returns:
        Resized PIL Image
    """
    width, height = pil_image.size
    current_pixels = width * height

    if current_pixels <= max_pixels:
        return pil_image

    # Calculate scaling factor
    scale_factor = (max_pixels / current_pixels) ** 0.5
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize using high-quality LANCZOS resampling
    resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image

def waveform_to_temp_file(waveform_tensor, sample_rate: int = 16000) -> str:
    """Convert audio waveform tensor to temporary WAV file for MLX processing.

    Args:
        waveform_tensor: Audio waveform as torch.Tensor or numpy array
        sample_rate: Sample rate for the audio file

    Returns:
        Path to temporary WAV file
    """
    import soundfile as sf

    # Convert tensor to numpy if needed
    if hasattr(waveform_tensor, 'numpy'):
        waveform = waveform_tensor.numpy()
    else:
        waveform = np.array(waveform_tensor)

    # Ensure waveform is 1D or 2D
    if len(waveform.shape) > 2:
        waveform = waveform.squeeze()

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_file.name
    temp_file.close()

    # Write audio file
    sf.write(temp_path, waveform, sample_rate)

    return temp_path

def cleanup_temp_file(file_path: str):
    """Safely remove temporary file."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception:
        pass  # Silently fail to avoid breaking workflows
