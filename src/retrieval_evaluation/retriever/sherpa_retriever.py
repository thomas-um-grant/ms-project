from typing import Any

import torch
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever

from evaluation.utils.score_utils import score_multi_vector
from src.repositories.colpali_repository import ColPaliRepository
from src.services.colpali_service import ColPaliService
from src.utils.gpu_utils import get_torch_device


class SherpaVisionRetriever(BaseVisionRetriever):
    """
    A retriever that leverages ColPaliService for embeddings and scoring.
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

        # Setup ColPali
        colpali_repo = ColPaliRepository(
            model_name=model_name, dtype=dtype, device=device
        )
        self.colpali_service = ColPaliService(colpali_repo)

    async def forward_queries(
        self, queries: list[str] | str, **kwargs
    ) -> list[torch.Tensor]:
        """Embed queries using ColPaliService."""
        enriched_queries = [queries] if isinstance(queries, str) else queries
        query_embeddings = await self.colpali_service.embed_queries(enriched_queries)

        assert len(query_embeddings) == len(enriched_queries), (
            f"Query embeddings and enriched queries must be the same length, {len(query_embeddings)} != {len(enriched_queries)}"
        )

        return query_embeddings

    async def forward_passages(
        self, passages: list[Any], **kwargs
    ) -> list[torch.Tensor]:
        """Embed passages (images) using ColPaliService."""
        passage_embeddings = await self.colpali_service.embed_images(passages)

        return passage_embeddings

    def get_scores(
        self,
        query_embeddings: list[torch.Tensor] | torch.Tensor,
        passage_embeddings: list[torch.Tensor] | torch.Tensor,
        batch_size: int = 128,
        score_method: str = "multi_vector",
        **kwargs,
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
        if score_method == "multi_vector":
            return score_multi_vector(
                qs=query_embeddings,
                ps=passage_embeddings,
                batch_size=batch_size,
                device=self.device,
            )
        elif score_method == "colpali":
            return self.colpali_service.get_scores(
                query_embeddings=query_embeddings,
                passage_embeddings=passage_embeddings,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Unknown score method: {score_method}")
