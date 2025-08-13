import asyncio
import logging
from typing import ClassVar

import aiohttp
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image

from pipeline.models.base_embedder import BaseEmbeddingModel
from utils.device import DeviceConfig

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""

    def __init__(self, operation: str, details: str):
        self.operation = operation
        self.details = details
        super().__init__(f"{operation} failed: {details}")


class NomicOllamaModel(BaseEmbeddingModel):
    """
    Nomic embedding model using local Ollama API for text-only embeddings.

    This model only supports text embeddings and raises NotImplementedError for images.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/embeddings",
        model_name: str = "nomic-embed-text",
    ):
        self.ollama_url = ollama_url
        self.model_name = model_name
        logger.info(f"Initialized NomicOllamaModel with {model_name} at {ollama_url}")

    async def embed_images(
        self,
        images: list[Image.Image],
        dtype: torch.dtype = None,
    ) -> list[torch.Tensor]:
        """Not supported by Nomic text-only model."""
        msg = "NomicOllamaModel does not support image embedding."
        raise NotImplementedError(msg)

    async def embed_texts(self, texts: list[str]) -> list[torch.Tensor]:
        """
        Embed texts using Ollama API.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding tensors, one per input text

        """
        if not texts:
            return []

        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model_name,
                "prompt": texts,
            }

            try:
                async with session.post(self.ollama_url, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                # Convert to tensor and split by text
                embeddings_tensor = torch.tensor(
                    data["embeddings"],
                    dtype=torch.float32,
                )
                return [embeddings_tensor[i] for i in range(embeddings_tensor.shape[0])]

            except Exception as e:
                logger.exception("Failed to get embeddings from Ollama")
                raise EmbeddingError("Text embedding via Ollama", str(e)) from e


class ColQwen2Model(BaseEmbeddingModel):
    """
    ColQwen2 multimodal embedding model with proper batching and error handling.

    This model supports both text and image embeddings using the ColQwen2 architecture.
    It processes items individually to avoid padding-related NaN issues while maintaining
    efficient batching for device operations.
    """

    _model_cache: ClassVar[dict[str, tuple[ColQwen2, ColQwen2Processor]]] = {}

    def __init__(self, device_config: DeviceConfig):
        self.model_name = "vidore/colqwen2-v1.0"
        self.device_config = device_config
        self.model = None
        self.processor = None
        self._load_model()
        logger.info(f"Initialized ColQwen2Model on {device_config.device_str}")

    @property
    def device(self) -> str:
        """Get the device string for tensor operations."""
        return self.device_config.device_str

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype from device config."""
        return self.device_config.dtype

    def _process_single_image(
        self,
        image: Image.Image,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Process a single image and return its multi-vector embedding tensor."""
        try:
            processed_image = self.processor.process_images([image])
            processed_image = self._move_to_device_with_dtype(processed_image, dtype)

            with torch.inference_mode():
                embedding = self.model(**processed_image)

            return self._check_and_clean_tensor(embedding[0].to("cpu"), "image")

        except Exception as e:
            logger.exception("Failed to process image")
            msg = f"Image embedding failed: {e}"
            raise EmbeddingError("Image embedding", msg) from e

    def _process_single_text(self, text: str) -> torch.Tensor:
        """Process a single text and return its multi-vector embedding tensor."""
        try:
            processed_query = self.processor.process_queries([text])
            processed_query = self._move_to_device_without_dtype(processed_query)

            with torch.inference_mode():
                embedding = self.model(**processed_query)

            # Return the single multi-vector embedding (shape: [seq_len, emb_dim])
            context = f"text: '{text[:50]}...'"
            return self._check_and_clean_tensor(embedding[0].to("cpu"), context)

        except Exception as e:
            logger.exception("Failed to process text '%s'", text[:50])
            msg = f"Text embedding failed: {e}"
            raise EmbeddingError("Text embedding", msg) from e

    def _move_to_device_with_dtype(self, inputs: dict, dtype: torch.dtype) -> dict:
        """Move input tensors to device with dtype conversion for float tensors."""
        try:
            return {
                k: v.to(self.model.device).to(
                    self.model.dtype if v.dtype.is_floating_point else v.dtype,
                )
                for k, v in inputs.items()
            }
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "Could not move inputs to %s, using CPU: %s",
                self.model.device,
                e,
            )
            return {
                k: v.to("cpu").to(dtype if v.dtype.is_floating_point else v.dtype)
                for k, v in inputs.items()
            }

    def _move_to_device_without_dtype(self, inputs: dict) -> dict:
        """
        Move input tensors to device without dtype conversion.

        Args:
            inputs: Dictionary of input tensors

        Returns:
            Dictionary with tensors moved to device

        """
        try:
            return {k: v.to(self.model.device) for k, v in inputs.items()}
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "Could not move text inputs to %s, falling back to CPU: %s",
                self.model.device,
                e,
            )
            return {k: v.to("cpu") for k, v in inputs.items()}

    def _check_and_clean_tensor(
        self,
        tensor: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        """
        Check for NaN values in tensor and replace with zeros if found.

        Args:
            tensor: Tensor to check
            context: Context string for logging

        Returns:
            Clean tensor (original or zeros if NaN was found)

        """
        if torch.isnan(tensor).any():
            logger.warning(
                "NaN detected in embedding for %s, replacing with zeros",
                context,
            )
            return torch.zeros_like(tensor)
        return tensor

    async def embed_images(
        self,
        images: list[Image.Image],
        dtype: torch.dtype = None,
    ) -> list[torch.Tensor]:
        """Embed a list of images asynchronously."""
        if not images:
            return []

        if dtype is None:
            dtype = self.device_config.dtype

        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self._process_single_image, img, dtype)
            for img in images
        ]

        try:
            embeddings = await asyncio.gather(*tasks)
            logger.debug("Successfully embedded %d images", len(embeddings))
        except Exception:
            logger.exception("Failed to embed images")
            raise
        else:
            return embeddings

    async def embed_texts(self, texts: list[str]) -> list[torch.Tensor]:
        """Embed a list of texts asynchronously."""
        if not texts:
            return []

        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self._process_single_text, text)
            for text in texts
        ]

        try:
            embeddings = await asyncio.gather(*tasks)
            logger.debug("Successfully embedded %d texts", len(embeddings))
        except Exception:
            logger.exception("Failed to embed texts")
            raise
        else:
            return embeddings

    def _load_model(self) -> None:
        """
        Load the ColQwen model with device-specific configuration and caching.

        Uses a class-level cache to avoid reloading the same model configuration.
        """
        cache_key = f"{self.model_name}_{self.device_config.device_str}_{self.device_config.dtype}"

        if cache_key in self._model_cache:
            logger.info("Using cached model for %s", cache_key)
            self.model, self.processor = self._model_cache[cache_key]
        else:
            logger.info("Loading new model for %s", cache_key)

            try:
                # Load model with specified dtype
                self.model = ColQwen2.from_pretrained(
                    self.model_name,
                    torch_dtype=self.device_config.dtype,
                ).eval()

                # Attempt to move to target device
                try:
                    self.model = self.model.to(self.device_config.device_str)
                    logger.info(
                        "Successfully moved model to %s",
                        self.device_config.device_str,
                    )
                except RuntimeError as e:
                    if "offloaded to cpu or disk" in str(e).lower():
                        logger.info(
                            "Model already device-mapped, keeping current placement",
                        )
                    else:
                        logger.warning("Failed to move model to device: %s", e)

                # Load processor
                self.processor = ColQwen2Processor.from_pretrained(self.model_name)

                # Cache the loaded components
                self._model_cache[cache_key] = (self.model, self.processor)
                logger.info("Model loaded and cached successfully")

            except Exception:
                logger.exception("Failed to load ColQwen2 model")
                raise


def setup_embedding_model(
    model_name: str,
    device_config: DeviceConfig,
) -> BaseEmbeddingModel:
    """
    Factory function to create embedding model instances.

    Args:
        model_name: Name of the embedding model to create
        device_config: Device configuration for the model

    Returns:
        Configured embedding model instance

    Raises:
        ValueError: If model_name is not supported

    """
    if model_name == "nomic_embed":
        return NomicOllamaModel(
            ollama_url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text",
        )

    if model_name == "colqwen2_embed":
        return ColQwen2Model(device_config=device_config)

    supported_models = ["nomic_embed", "colqwen2_embed"]
    msg = f"Unsupported model name: {model_name}. Supported models: {supported_models}"
    raise ValueError(msg)
