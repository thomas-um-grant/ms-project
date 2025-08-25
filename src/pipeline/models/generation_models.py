import asyncio
import base64
import io
from typing import Any

import ollama
import torch
from PIL import Image

from pipeline.models.base_generator import BaseGenerationModel
from utils.device import DeviceConfig


class LlamaOllamaTextModel(BaseGenerationModel):
    """Simple text-only generation via Ollama llama3.2:3b model."""

    def __init__(self, device_config: DeviceConfig):
        self.model_name = "llama3.2:3b"
        self.device_config = device_config

    @property
    def device(self) -> str:  # pragma: no cover - simple property
        return self.device_config.device_str

    @property
    def dtype(self) -> torch.dtype:  # pragma: no cover
        return self.device_config.dtype

    def _load_model(self) -> None:
        """Load model via Ollama. No explicit loading needed as Ollama handles this."""

    async def generate(self, query: str, context: list[Any] | None = None) -> str:
        prompt = "You are a helpful assistant. Use the provided context to answer the question.\n\n"
        if context:
            for item in context:
                if item.get("type") == "text":
                    prompt += f"Context: {item['text']}\n"
        prompt += f"Question: {query}\nAnswer:"

        loop = asyncio.get_event_loop()

        def _gen():
            return ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.7, "num_predict": 512},
            )

        result = await loop.run_in_executor(None, _gen)
        if isinstance(result, dict) and "response" in result:
            return result["response"].strip()
        return str(result)


class ColQwen2OllamaModel(BaseGenerationModel):
    """Ollama-based Qwen2-VL model for faster inference on ARM Macs."""

    def __init__(self, device_config: DeviceConfig):
        self.model_name = "qwen2.5vl:7b"  # Ollama model name
        self.device_config = device_config

    @property
    def device(self) -> str:
        """Get the device from device config."""
        return self.device_config.device_str

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype from device config."""
        return self.device_config.dtype

    def _load_model(self) -> None:
        """Load model via Ollama. No explicit loading needed as Ollama handles this."""

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()

        # Handle RGBA images by converting to RGB or saving as PNG
        if image.mode in ("RGBA", "LA"):
            # For images with transparency, save as PNG to preserve transparency
            image.save(buffered, format="PNG")
        elif image.mode == "P" and "transparency" in image.info:
            # For palette images with transparency, convert to RGBA then save as PNG
            image = image.convert("RGBA")
            image.save(buffered, format="PNG")
        else:
            # For other modes, convert to RGB and save as JPEG for smaller size
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffered, format="JPEG", quality=95)

        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    async def generate(self, query: str, context: list[Any] | None = None) -> str:
        """Generate an answer using Ollama Python library."""
        # Build the prompt
        prompt = "You are a helpful assistant that can analyze both text and images to answer questions. Use the provided context to give accurate and helpful responses.\n\n"

        # Add context
        if context:
            for item in context:
                if item.get("type") == "text":
                    prompt += f"Context: {item['text']}\n"

        prompt += f"Question: {query}\nAnswer:"

        # Handle images separately
        images = []
        if context:
            for item in context:
                if item.get("type") == "image":
                    if isinstance(item["image"], str):
                        image = Image.open(item["image"])
                    else:
                        image = item["image"]
                    images.append(self._image_to_base64(image))

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()

            def _generate_sync():
                if images:
                    # For vision models with images
                    response = ollama.generate(
                        model=self.model_name,
                        prompt=prompt,
                        images=images,
                        options={"temperature": 0.8, "num_predict": 512},
                    )
                else:
                    # For text-only
                    response = ollama.generate(
                        model=self.model_name,
                        prompt=prompt,
                        options={"temperature": 0.8, "num_predict": 512},
                    )
                return response

            result = await loop.run_in_executor(None, _generate_sync)

            if "response" in result:
                return result["response"].strip()

            return f"Unexpected response format: {result}"

        except Exception as e:
            error_msg = f"Failed to generate with Ollama: {e}"
            raise RuntimeError(error_msg) from e


def setup_generation_model(
    model_name: str,
    device_config: DeviceConfig,
) -> BaseGenerationModel:
    """Setup the generation model based on the model name and device."""
    if model_name == "colqwen2_ollama_gen":
        return ColQwen2OllamaModel(device_config)
    if model_name == "llama_text_gen":
        return LlamaOllamaTextModel(device_config)

    error_msg = f"Unsupported generation model: {model_name}"
    raise ValueError(error_msg)
