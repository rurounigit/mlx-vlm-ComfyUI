import mlx.core as mx
import re
from typing import List, Dict, Any, Optional
from PIL import Image

class MLX_VLM_ChatTemplate:
    """Structure conversation history and media context for MLX VLM models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_message": ("STRING", {
                    "default": "You are a helpful AI assistant.",
                    "multiline": True
                }),
                "user_message": ("STRING", {
                    "default": "Describe the image.",
                    "multiline": True
                })
            },
            "optional": {
                "images": ("MLX_IMAGE_BATCH",),
                "audio": ("MLX_AUDIO",),
                "history": ("MLX_CHAT_HISTORY",)
            }
        }

    RETURN_TYPES = ("MLX_PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "create_prompt"
    CATEGORY = "MLX_VLM/Sampler"

    def create_prompt(self, system_message: str, user_message: str,
                     images: Optional[List[Image.Image]] = None,
                     audio: Optional[Dict[str, Any]] = None,
                     history: Optional[List[Dict[str, Any]]] = None):
        """Create structured prompt for MLX VLM models."""

        # Initialize messages list
        messages = []

        # Add system message
        if system_message.strip():
            messages.append({
                "role": "system",
                "content": system_message.strip()
            })

        # Add history if provided
        if history:
            messages.extend(history)

        # Build user message content
        user_content = []

        # Add images if present
        has_explicit_image_placeholder = "<image>" in user_message

        if images:
            if has_explicit_image_placeholder:
                # User wants explicit image placement - we'll handle this in the sampler
                pass
            else:
                # Auto-prepend images
                for _ in images:
                    user_content.append({"type": "image"})

        # Add audio if present
        if audio:
            user_content.append(audio)

        # Add text content
        text_content = user_message.strip()
        if text_content:
            user_content.append({"type": "text", "text": text_content})
        elif not user_content:  # Ensure we have some content
            user_content.append({"type": "text", "text": ""})

        # Add user message
        messages.append({
            "role": "user",
            "content": user_content if len(user_content) > 1 else user_content[0]["text"]
        })

        # Create prompt object
        prompt_obj = {
            "messages": messages,
            "images": images,
            "audio": audio,
            "has_images": images is not None and len(images) > 0,
            "has_audio": audio is not None
        }

        print(f"Created prompt with {len(messages)} messages")
        if images:
            print(f"  - {len(images)} images")
        if audio:
            print(f"  - Audio input included")

        return (prompt_obj,)

class MLX_VLM_Sampler:
    """Execute MLX VLM model inference with advanced sampling controls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MLX_MODEL",),
                "prompt": ("MLX_PROMPT",),
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1
                })
            },
            "optional": {
                "strip_thinking": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Strip Reasoning",
                    "label_off": "Keep Reasoning"
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text_response", "thought_process")
    FUNCTION = "generate"
    CATEGORY = "MLX_VLM/Sampler"

    def generate(self, model: Dict[str, Any], prompt: Dict[str, Any],
                 max_tokens: int, temperature: float, top_p: float, seed: int,
                 strip_thinking: bool = False):
        """Generate text response using MLX VLM model."""

        try:
            # Set random seed for reproducibility
            mx.random.seed(seed)

            # Extract model components
            mlx_model = model["model"]
            processor = model["processor"]
            config = model.get("config", {})

            # Extract prompt components
            messages = prompt["messages"]
            images = prompt.get("images")
            audio = prompt.get("audio")

            # Apply chat template
            formatted_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Prepare generation arguments
            generate_kwargs = {
                "model": mlx_model,
                "processor": processor,
                "prompt": formatted_prompt,
                "max_tokens": max_tokens,
                "temp": temperature,
                "top_p": top_p,
                "verbose": False
            }

            # Add media inputs if present
            if images and len(images) > 0:
                generate_kwargs["image"] = images[0] if len(images) == 1 else images

            if audio:
                generate_kwargs["audio"] = audio.get("input_audio")

            # Import mlx_vlm for generation
            import mlx_vlm

            # Generate response
            response = mlx_vlm.generate(**generate_kwargs)

            # Extract text from response
            if isinstance(response, dict):
                generated_text = response.get("text", str(response))
            else:
                generated_text = str(response)

            # Process thinking/reasoning content
            thought_process = ""
            final_text = generated_text

            if strip_thinking and "<think>" in generated_text and "</think>" in generated_text:
                # Extract thinking process
                think_pattern = r"<think>(.*?)</think>"
                think_matches = re.findall(think_pattern, generated_text, re.DOTALL)
                thought_process = "\n".join(think_matches)

                # Remove thinking tags from final output
                final_text = re.sub(think_pattern, "", generated_text, flags=re.DOTALL)
                final_text = re.sub(r"<think>.*?</think>", "", final_text, flags=re.DOTALL)
                final_text = final_text.strip()

            print(f"Generated response ({len(generated_text)} chars)")
            if thought_process:
                print(f"  - Extracted thinking process ({len(thought_process)} chars)")

            return (final_text, thought_process)

        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            raise RuntimeError(error_msg)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MLX_VLM_ChatTemplate": MLX_VLM_ChatTemplate,
    "MLX_VLM_Sampler": MLX_VLM_Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLX_VLM_ChatTemplate": "MLX VLM Chat Template",
    "MLX_VLM_Sampler": "MLX VLM Sampler"
}
