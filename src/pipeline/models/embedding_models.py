import asyncio
from typing import ClassVar

import aiohttp
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image

from pipeline.models.base_embedder import BaseEmbeddingModel
from utils.device import DeviceConfig


# Local Nomic Model (text embedding only)
class NomicOllamaModel(BaseEmbeddingModel):
    def __init__(
        self,
        ollama_url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    ):
        self.ollama_url = ollama_url
        self.model_name = model_name

    async def embed_images(self, images: list):
        msg = "NomicOllamaModel does not support image embedding."
        raise NotImplementedError(msg)

    async def embed_texts(self, texts: list[str]):
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model_name,
                "prompt": texts if isinstance(texts, list) else [texts],
            }
            async with session.post(self.ollama_url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        return torch.tensor(data["embeddings"], dtype=torch.float32)


# Local ColQwen2 Model (multi-modal embedding)
class ColQwen2Model(BaseEmbeddingModel):
    _model_cache: ClassVar[dict[str, tuple[ColQwen2, ColQwen2Processor]]] = {}

    def __init__(self, device_config: DeviceConfig):
        self.model_name = "vidore/colqwen2-v1.0"
        self.device_config = device_config
        self._load_model()

    @property
    def device(self) -> str:
        """Get the device for tensor operations."""
        return self.device_config.device_str

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype from device config."""
        return self.device_config.dtype

    def _embed_images_sync(
        self,
        images: list[Image.Image],
        batch_size: int,
    ) -> torch.Tensor:
        self.model.eval()
        all_embeddings = []

        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i : i + batch_size]

                # Process images and prepare inputs
                inputs = self.processor.process_images(batch_imgs)

                # Handle device placement carefully for device-mapped models
                try:
                    # Try to move inputs to the expected device
                    inputs = inputs.to(device=self.device, dtype=self.dtype)
                except (RuntimeError, ValueError) as e:
                    print(
                        f"Warning: Could not move inputs to {self.device}, using CPU: {e}",
                    )
                    inputs = inputs.to(device="cpu", dtype=self.dtype)

                embeddings = self.model(**inputs)
                # Move to CPU with consistent dtype for concatenation
                all_embeddings.append(embeddings.to(dtype=torch.float32, device="cpu"))

        return torch.cat(all_embeddings, dim=0)

    async def embed_images(self, images: list[Image.Image], batch_size: int = 8):
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(
            None,
            lambda: self._embed_images_sync(images, batch_size),
        )

    def _embed_texts_sync(
        self,
        texts: list[str],
        batch_size: int,
    ) -> torch.Tensor:
        self.model.eval()
        all_embeddings = []

        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                )

                # Handle device placement carefully for device-mapped models
                try:
                    inputs = inputs.to(device=self.device, dtype=self.dtype)
                except (RuntimeError, ValueError) as e:
                    print(
                        f"Warning: Could not move text inputs to {self.device}, using CPU: {e}",
                    )
                    inputs = inputs.to(device="cpu", dtype=self.dtype)

                embeddings = self.model(**inputs)
                # Move to CPU with consistent dtype for concatenation
                all_embeddings.append(embeddings.to(dtype=torch.float32, device="cpu"))

        return torch.cat(all_embeddings, dim=0)

    async def embed_texts(self, texts: list[str], batch_size: int = 8):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._embed_texts_sync(texts, batch_size),
        )

    def _load_model(self):
        """Load the ColQwen model with device-specific configuration and caching."""
        cache_key = f"{self.model_name}_{self.device_config.device_str}_{self.device_config.dtype}"

        if cache_key in self._model_cache:
            print(f"Using cached model for {cache_key}")
            self.model, self.processor = self._model_cache[cache_key]
            # Don't try to move device-mapped models
        else:
            print(f"Loading new model for {cache_key}")
            self.model = ColQwen2.from_pretrained(
                self.model_name,
                torch_dtype=self.device_config.dtype,
            ).eval()

            # Move to device
            try:
                self.model = self.model.to(self.device_config.device_str)
                print(f"Successfully moved model to {self.device_config.device_str}")
            except RuntimeError as e:
                if "offloaded to cpu or disk" in str(e):
                    print(
                        f"Model already device-mapped, keeping current placement: {e}",
                    )
                else:
                    print(f"Failed to move model to device: {e}")
                    # Continue anyway, the model might still work on CPU

            self.processor = ColQwen2Processor.from_pretrained(self.model_name)

            # Cache the loaded model
            self._model_cache[cache_key] = (self.model, self.processor)


def setup_embedding_model(
    model_name: str,
    device_config: DeviceConfig,
) -> BaseEmbeddingModel:
    """
    Setup the embedding model based on the provided model name and device.

    Args:
        model_name (str): Name of the embedding model.
        device_config (DeviceConfig): Device configuration containing device, dtype, and device_map settings.

    Returns:
        BaseEmbeddingModel: An instance of the embedding model.

    """
    if model_name == "nomic_embed":
        return NomicOllamaModel(
            ollama_url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text",
        )

    if model_name == "colqwen2_embed":
        return ColQwen2Model(
            device_config=device_config,
        )

    msg = f"Unsupported model name: {model_name}"
    raise ValueError(msg)
