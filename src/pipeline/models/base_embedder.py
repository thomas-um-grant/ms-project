from abc import ABC, abstractmethod

import torch
from PIL import Image


class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed_images(self, images: list[Image.Image], dtype: torch.dtype = None):
        """Embed a list of images asynchronously and in parallel."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]):
        """Embed a list of texts asynchronously and in parallel."""
