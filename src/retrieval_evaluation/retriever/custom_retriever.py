from typing import Any

import torch
from config import settings
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever

from retrieval_evaluation.utils.score_utils import score_multi_vector
from src.repositories.embeddings_repository import EmbeddingsRepository
from src.utils.gpu_utils import get_torch_device


class CustomVisionRetriever(BaseVisionRetriever):
    """
    A retriever that leverages EmbeddingsRepository for embeddings and scoring.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str,
        device: str,
        num_workers: int,
    ):
        super().__init__(use_visual_embedding=True)
        self.device = get_torch_device(device)
        self.num_workers = num_workers

        # Setup EmbeddingsRepository
        self.embeddings_repository = EmbeddingsRepository(
            x_api_key=settings.embeddings_secret_key,
            base_url=settings.embeddings_base_url,
        )

    async def forward_queries(
        self,
        queries: list[str] | str,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Embed queries using EmbeddingsRepository."""
        enriched_queries = [queries] if isinstance(queries, str) else queries
        query_embeddings = await self.embeddings_repository.embed_queries(
            enriched_queries,
        )

        assert len(query_embeddings) == len(enriched_queries), (
            f"Query embeddings and enriched queries must be the same length, {len(query_embeddings)} != {len(enriched_queries)}"
        )

        return query_embeddings

    async def forward_passages(
        self,
        passages: list[Any],
        **kwargs,
    ) -> list[torch.Tensor]:
        """Embed passages (images) using EmbeddingsRepository."""
        passage_embeddings = await self.embeddings_repository.embed_images(passages)

        return passage_embeddings

    def get_scores(
        self,
        query_embeddings: list[torch.Tensor] | torch.Tensor,
        passage_embeddings: list[torch.Tensor] | torch.Tensor,
        batch_size: int = 128,
    ) -> torch.Tensor:
        """
        Get similarity scores between queries and passages.

        Args:
                query_embeddings: Query embeddings from forward_queries
                passage_embeddings: Passage embeddings from forward_passages
                batch_size: Batch size for scoring computation
                score_method: Scoring method to use ("multi_vector" or "colpali")

        Returns:
                Tensor of shape (n_queries, n_passages) containing similarity scores

        """
        return score_multi_vector(
            qs=query_embeddings,
            ps=passage_embeddings,
            batch_size=batch_size,
            device=self.device,
        )
