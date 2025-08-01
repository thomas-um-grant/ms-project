from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed_images(self, images: list):
        """Embed a list of images asynchronously and in parallel."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]):
        """Embed a list of texts asynchronously and in parallel."""
