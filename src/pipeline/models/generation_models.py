import asyncio
import base64
import io
from typing import Any, ClassVar

import ollama
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from retrieval_pipeline.device import DeviceConfig
from retrieval_pipeline.models.base_generator import BaseGenerationModel


class ColQwen2Model(BaseGenerationModel):
    _model_cache: ClassVar[
        dict[str, tuple[Qwen2VLForConditionalGeneration, AutoProcessor]]
    ] = {}

    def __init__(self, device_config: DeviceConfig):
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        self.device_config = device_config
        self._load_model()

    @property
    def device(self) -> str:
        """Get the device from device config."""
        return self.device_config.device_str

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype from device config."""
        return self.device_config.dtype

    def _load_model(self):
        """Load the Qwen2VL model with device-specific configuration and caching."""
        cache_key = f"{self.model_name}_{self.device_config.device_str}_{self.device_config.dtype}"

        if cache_key in self._model_cache:
            print(f"Using cached model for {cache_key}")
            self.model, self.processor = self._model_cache[cache_key]
            # Move model to current device if needed
            if (
                hasattr(self.model, "device")
                and str(self.model.device) != self.device_config.device_str
            ):
                self.model = self.model.to(self.device_config.device_str)
        else:
            print(f"Loading new model for {cache_key}")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.device_config.dtype,
                device_map=self.device_config.device_map_str,
            ).eval()

            # Bound image size to boost perf
            # github.com/QwenLM/Qwen2.5-VL/blob/main/README.md#image-resolution-for-performance-boost
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

            # Cache the loaded model
            self._model_cache[cache_key] = (self.model, self.processor)

    def _load_images(self, messages: list[dict[str, Any]]) -> list[Any]:
        """
        Process image information from messages for the model.

        Args:
            messages: List of conversation messages containing images

        Returns:
            List of processed image inputs for the model

        """
        images = []
        for message in messages:
            if message.get("type") == "image":
                images.append(Image.open(message["image"]))

        return images

    def _generate_sync(
        self,
        query: str,
        context: list[dict[str, Any]] | None = None,
    ) -> str:
        """Synchronous generation method."""
        self.model.eval()

        # Prepare the conversation messages
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant that can analyze both text and images to answer questions. "
                        "Use the provided context to give accurate and helpful responses.",
                    },
                ],
            },
        ]

        # Prepare user message with query and context
        if context:
            user_content = [*context, {"type": "text", "text": f"Question: {query}"}]
        else:
            user_content = [{"type": "text", "text": f"Question: {query}"}]

        messages.append(
            {
                "role": "user",
                "content": user_content,
            },
        )

        image_inputs = self._load_images(context) if context else []

        # Apply chat template and prepare inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device=self.device, dtype=self.dtype)

        # Generate response
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
            )

        # Decode the generated response
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]

        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return response

    async def generate(self, query: str, context: list[Any] | None = None) -> str:
        """Generate an answer based on the input query and context."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate_sync(query, context),
        )


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
    if model_name == "colqwen2_gen":
        return ColQwen2Model(device_config)
    if model_name == "colqwen2_ollama_gen":
        return ColQwen2OllamaModel(device_config)

    error_msg = f"Unsupported generation model: {model_name}"
    raise ValueError(error_msg)
