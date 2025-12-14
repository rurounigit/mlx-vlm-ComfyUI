# mlx-vlm-ComfyUI

Native Apple Silicon Integration for Vision Language Models in ComfyUI

## Overview

mlx-vlm-ComfyUI is a custom node pack based on mlx-vlm by @Blaizzy that brings the power of Apple's MLX framework to ComfyUI, enabling high-performance inference of Vision Language Models (VLMs) on Apple Silicon Macs. This integration leverages the Unified Memory Architecture (UMA) of M-series chips to deliver efficient, low-latency execution of massive multi-modal models without the traditional bottlenecks associated with PCI Express data transfer.

## Key Features

- **Native Apple Silicon Support**: Optimized for M1/M2/M3/M4 chips using MLX framework
- **Multi-Modal Processing**: Support for images, video, and audio inputs
- **Advanced Quantization**: 4-bit quantization for reduced memory usage (70% reduction)
- **LoRA Support**: Dynamic Low-Rank Adaptation adapter loading
- **Memory Management**: Global model registry with intelligent caching
- **Omni Model Support**: Audio and video understanding capabilities
- **Reasoning Extraction**: Support for Qwen2.5/Qwen3 thinking process extraction

## Architecture

The node pack implements a robust data-bridging layer that seamlessly converts ComfyUI's eager PyTorch tensors into MLX's lazy arrays without performance penalties. Key architectural components include:

- **Global Model Registry**: Singleton-based caching to prevent redundant model loading
- **Type Bridging Layer**: PyTorch tensor to MLX array conversion with PIL encapsulation
- **Hugging Face Integration**: Proper cache directory usage and model loading
- **Memory Management**: Automatic garbage collection and cache clearing

## Installation

1. Clone this repository to your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone <repository-url>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

## Node Types

### Loaders
- **MLX VLM Load Model**: Load Vision Language Models with quantization support
- **MLX VLM Load LoRA**: Apply Low-Rank Adaptation adapters dynamically
- **MLX VLM Free Cache**: Manage memory by clearing MLX cache

### Processors
- **MLX VLM Image Processor**: Convert ComfyUI images to PIL format with dynamic resizing
- **MLX VLM Video Loader**: Extract video frames for multi-image processing
- **MLX VLM Audio Processor**: Process audio inputs for Omni models

### Sampler
- **MLX VLM Chat Template**: Structure conversation history and media context
- **MLX VLM Sampler**: Execute model inference with advanced sampling controls

## Supported Models

The node pack works with various VLM architectures including:
- Qwen2-VL series
- LLaVA series
- Pixtral series
- SmolVLM2
- Gemma-3n (Omni models)

## Performance Benefits

- **Memory Efficiency**: 4-bit quantization reduces memory usage by ~70%
- **Unified Memory**: Eliminates PCIe transfer bottlenecks
- **Lazy Evaluation**: MLX's computation graph optimization
- **Hardware Acceleration**: Native Metal Performance Shaders backend

## Example Workflow

See `workflows/example_qwen2_vl.json` for a complete example using Qwen2-VL-7B with image input.

## Requirements

- macOS with M-series chip (M1, M2, M3, or M4)
- ComfyUI installation
- Python 3.8+
- mlx-vlm library
- Required Python packages (see requirements.txt)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- mlx-vlm package for inference and fine-tuning of Vision Language Models (VLMs) on your Mac using MLX by @Blaizzy alias Prince Canuma
- Apple Machine Learning Research for the MLX framework
- The mlx-vlm community for Vision Language Model implementations
- ComfyUI development team for the node-based interface framework
