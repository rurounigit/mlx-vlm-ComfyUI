import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
import os

def extract_video_frames(video_path: str, num_frames: int = 8, mode: str = "uniform") -> List[Image.Image]:
    """Extract frames from video for MLX VLM processing.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        mode: Extraction mode - "uniform" or "keyframe"

    Returns:
        List of PIL Images representing video frames
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError("Video has no frames")

    frames = []

    if mode.lower() == "uniform":
        # Uniform frame extraction
        interval = max(1, total_frames // num_frames)
        frame_indices = [i * interval for i in range(num_frames)]
        frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]
    else:
        # Keyframe extraction (simplified - extract evenly spaced frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)

    cap.release()
    return frames

def validate_audio_file(audio_path: str) -> bool:
    """Validate that an audio file exists and is accessible.

    Args:
        audio_path: Path to audio file

    Returns:
        True if file is valid, False otherwise
    """
    if not os.path.exists(audio_path):
        return False

    # Check file extension for common audio formats
    valid_extensions = {'.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg'}
    _, ext = os.path.splitext(audio_path.lower())

    return ext in valid_extensions

def create_audio_prompt_dict(audio_path: str) -> dict:
    """Create the prompt dictionary structure for audio input.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary structure for MLX VLM audio input
    """
    return {
        "type": "input_audio",
        "input_audio": audio_path
    }

def get_video_info(video_path: str) -> dict:
    """Get basic information about a video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video information
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1)
    }

    cap.release()
    return info
