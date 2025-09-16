from abc import abstractmethod

import torch
from PIL import Image

from src.pipeline.models.base_model import BaseModel


class BaseEmbeddingModel(BaseModel):
    """Abstract base class for all embedding models. Defines the interface for embedding images and texts."""

    @abstractmethod
    async def embed_images(
        self,
        images: list[Image.Image],
        dtype: torch.dtype | None,
    ) -> list[torch.Tensor]:
        """Embed a list of images asynchronously and in parallel."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[torch.Tensor]:
        """Embed a list of texts asynchronously and in parallel."""

    @abstractmethod
    def _load_model(self) -> None:
        """Abstract method to load the embedding model. Each concrete model will implement its specific loading logic here."""
