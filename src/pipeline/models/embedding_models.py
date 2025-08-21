import asyncio
import logging
from typing import ClassVar

import aiohttp
import torch
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
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
        dtype: torch.dtype | None = None,
    ) -> list[torch.Tensor]:
        """Not supported by Nomic text-only model."""
        msg = "NomicOllamaModel does not support image embedding."
        raise NotImplementedError(msg)

    async def embed_texts(self, texts: list[str]) -> list[torch.Tensor]:  # type: ignore[override]
        if not texts:
            return []

        async def _embed_one(session: aiohttp.ClientSession, text: str, idx: int):
            payload = {"model": self.model_name, "prompt": text}
            async with session.post(self.ollama_url, json=payload) as resp:
                body = await resp.text()
                if resp.status != 200:
                    logger.exception(
                        "Ollama embedding request failed (status=%s, idx=%d): %.200s",
                        resp.status,
                        idx,
                        body,
                    )
                    op = "Text embedding via Ollama"
                    details = f"status={resp.status} body={body[:200]}"
                    raise EmbeddingError(op, details)
                data = await resp.json()
                if "embedding" not in data:
                    op = "Text embedding via Ollama"
                    details = "Missing 'embedding' key in response"
                    raise EmbeddingError(op, details)
                emb = torch.tensor(data["embedding"], dtype=torch.float32)
                return idx, emb

        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(_embed_one(session, t, i))
                for i, t in enumerate(texts)
            ]
            gathered = await asyncio.gather(*tasks, return_exceptions=True)

        first_error = next((g for g in gathered if isinstance(g, Exception)), None)
        if first_error:
            raise first_error
        ordered = sorted(gathered, key=lambda x: x[0])  # type: ignore[arg-type]
        return [emb for _i, emb in ordered]  # type: ignore[misc]


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
        assert self.processor is not None and self.model is not None, "Model not loaded"
        try:
            processed_image = self.processor.process_images([image])
            processed_image = self._move_to_device_with_dtype(processed_image, dtype)
            with torch.inference_mode():
                embedding = self.model(**processed_image)
            return self._check_and_clean_tensor(embedding[0].to("cpu"), "image")
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to process image")
            op = "Image embedding"
            msg = f"Image embedding failed: {e}"
            raise EmbeddingError(op, msg) from e

    def _process_single_text(self, text: str) -> torch.Tensor:
        assert self.processor is not None and self.model is not None, "Model not loaded"
        try:
            processed_query = self.processor.process_queries([text])
            processed_query = self._move_to_device_without_dtype(processed_query)
            with torch.inference_mode():
                embedding = self.model(**processed_query)
            context = f"text: '{text[:50]}...'"
            return self._check_and_clean_tensor(embedding[0].to("cpu"), context)
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to process text '%s'", text[:50])
            op = "Text embedding"
            msg = f"Text embedding failed: {e}"
            raise EmbeddingError(op, msg) from e

    def _move_to_device_with_dtype(self, inputs: dict, dtype: torch.dtype) -> dict:
        if self.model is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)
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
        if self.model is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)
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
        dtype: torch.dtype | None = None,
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


class ColPaliModel(BaseEmbeddingModel):
    """
    ColPali multimodal embedding model with proper batching and error handling.

    This model supports both text and image embeddings using the original ColPali architecture
    as described in the ColPali paper. It processes items individually to avoid padding-related
    NaN issues while maintaining efficient batching for device operations.
    """

    _model_cache: ClassVar[dict[str, tuple[ColPali, ColPaliProcessor]]] = {}

    def __init__(self, device_config: DeviceConfig):
        self.model_name = "vidore/colpali-v1.2"
        self.device_config = device_config
        self.model = None
        self.processor = None
        self._load_model()
        logger.info(f"Initialized ColPaliModel on {device_config.device_str}")

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
        if self.processor is None or self.model is None:
            msg = "ColPaliModel not loaded before image processing"
            raise RuntimeError(msg)
        try:
            # Let processor handle resizing / preprocessing
            processed_image = self.processor.process_images([image])
            processed_image = self._move_to_device_with_dtype(processed_image, dtype)
            with torch.inference_mode():
                embedding = self.model(**processed_image)
            emb = self._check_and_clean_tensor(embedding[0].to("cpu"), "image")
            return emb.to(torch.float32)
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to process image")
            op = "Image embedding"
            msg = f"Image embedding failed: {e}"
            raise EmbeddingError(op, msg) from e

    def _process_single_text(self, text: str) -> torch.Tensor:
        if self.processor is None or self.model is None:
            msg = "ColPaliModel not loaded before text processing"
            raise RuntimeError(msg)
        try:
            processed_query = self.processor.process_queries([text])
            processed_query = self._move_to_device_without_dtype(processed_query)
            with torch.inference_mode():
                embedding = self.model(**processed_query)
            context = f"text: '{text[:50]}...'"
            emb = self._check_and_clean_tensor(embedding[0].to("cpu"), context)
            return emb.to(torch.float32)
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to process text '%s'", text[:50])
            op = "Text embedding"
            msg = f"Text embedding failed: {e}"
            raise EmbeddingError(op, msg) from e

    def _move_to_device_with_dtype(self, inputs: dict, dtype: torch.dtype) -> dict:
        if self.model is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)
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
        if self.model is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)
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
        Load the ColPali model with device-specific configuration and caching.

        Uses a class-level cache to avoid reloading the same model configuration.
        Includes fallback to older model versions if loading issues occur.
        """
        cache_key = f"{self.model_name}_{self.device_config.device_str}_{self.device_config.dtype}"

        if cache_key in self._model_cache:
            logger.info("Using cached model for %s", cache_key)
            self.model, self.processor = self._model_cache[cache_key]
        else:
            logger.info("Loading new model for %s", cache_key)

            # Try loading the model with fallback versions if needed (added v1.3 first)
            models_to_try = [
                "vidore/colpali-v1.3",
                "vidore/colpali-v1.2",
                "vidore/colpali-v1.0",
                "vidore/colpali-v0.3",
            ]

            model_loaded = False
            for model_name in models_to_try:
                try:
                    logger.info(
                        "Attempting to load %s (mask_non_image_embeddings=True)",
                        model_name,
                    )

                    # Load model with masking of non-image embeddings to reduce noise from prompt tokens
                    self.model = ColPali.from_pretrained(
                        model_name,
                        torch_dtype=self.device_config.dtype,
                        mask_non_image_embeddings=True,
                    ).eval()

                    self.model_name = model_name  # Update to successful model
                    logger.info("✅ Successfully loaded %s", model_name)
                    model_loaded = True
                    break

                except (OSError, ValueError, RuntimeError) as e:
                    logger.warning("❌ Failed to load %s: %s", model_name, str(e)[:120])
                    continue

            if not model_loaded:
                msg = "Failed to load any ColPali model version"
                raise RuntimeError(msg)

            try:
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
                self.processor = ColPaliProcessor.from_pretrained(self.model_name)

                # Cache the loaded components
                self._model_cache[cache_key] = (self.model, self.processor)
                logger.info(
                    "Model loaded and cached successfully (ColPali masking active)",
                )

            except Exception:
                logger.exception("Failed to finalize ColPali model load")
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

    if model_name == "colpali_embed":
        return ColPaliModel(device_config=device_config)

    supported_models = ["nomic_embed", "colqwen2_embed", "colpali_embed"]
    msg = f"Unsupported model name: {model_name}. Supported models: {supported_models}"
    raise ValueError(msg)
