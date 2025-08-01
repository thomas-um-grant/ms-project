import asyncio

import aiohttp
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from retrieval_pipeline.models.base_embedder import BaseEmbeddingModel


# Nomic Model via Ollama API (text embedding only)
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
    def __init__(self, device, dtype="auto"):
        self.model_name = "vidore/colqwen2-v1.0"
        # Convert string device to torch.device if needed
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = (
            torch.bfloat16
            if dtype == "auto" and torch.cuda.is_available()
            else getattr(torch, dtype)
            if dtype != "auto"
            else torch.float32
        )

        self.model = (
            ColQwen2.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2"
                if is_flash_attn_2_available()
                else None,
            )
            .to(self.device)
            .eval()
        )
        self.processor = ColQwen2Processor.from_pretrained(self.model_name)

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
                inputs = self.processor.process_images(batch_imgs).to(self.device)
                embeddings = self.model(**inputs)
                all_embeddings.append(embeddings.cpu())

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
                ).to(self.device)

                outputs = self.model(**inputs)
                embeddings = outputs.embeddings
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    async def embed_texts(self, texts: list[str], batch_size: int = 8):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._embed_texts_sync(texts, batch_size),
        )


def setup_embedding_model(model_name: str, device: str = "auto") -> BaseEmbeddingModel:
    """
    Setup the embedding model based on the provided model name and device.

    Args:
        model_name (str): Name of the embedding model.
        device (str): Device to run the model on, can be 'auto', 'cuda', 'mps', or 'cpu'.

    Returns:
        BaseEmbeddingModel: An instance of the embedding model.

    """
    if model_name == "nomic-text":
        return NomicOllamaModel(
            ollama_url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text",
        )

    if model_name == "colqwen2":
        return ColQwen2Model(
            device=device,
        )

    msg = f"Unsupported model name: {model_name}"
    raise ValueError(msg)
