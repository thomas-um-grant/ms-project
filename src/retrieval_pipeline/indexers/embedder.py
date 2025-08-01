from retrieval_pipeline import utils
from retrieval_pipeline.models.base_embedder import BaseEmbeddingModel


class Embedder:
    def __init__(
        self,
        model: BaseEmbeddingModel,
        device=None,
    ):
        device = device or utils.get_torch_device("auto")
        self.model = model
        self.device = device

    async def embed_images(self, images: list):
        return await self.model.embed_images(images)

    async def embed_texts(self, texts: list[str]):
        return await self.model.embed_texts(texts)
