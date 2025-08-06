import os
from typing import Any

import httpx
import torch
from dotenv import load_dotenv
from vespa.application import VespaAsync

load_dotenv()

from src.config import settings
from src.repositories.embeddings_repository import EmbeddingsRepository
from src.repositories.vespa_repository import VespaRepository
from src.services.retrieval_service import RetrievalService
from src.services.vespa_service import VespaService
from src.utils.gpu_utils import get_torch_device
from src.vespa_utils.vespa_schemas import VDBQueryResponse


class EvaluationRetriever:
    """
    A retriever that leverages ColPaliService for embeddings and scoring.
    """

    def __init__(
        self,
        device: str,
        num_workers: int,
    ):
        self.device = get_torch_device(device)
        self.num_workers = num_workers
        # Embedder
        self.embeddings_repo = EmbeddingsRepository(
            x_api_key=settings.embeddings_secret_key,
            base_url=settings.embeddings_base_url,
        )

        # Vespa
        vespa_url = os.getenv(
            "VESPA_EVALS_APP_URL",
            "https://dbevals.vespa-app.cloud",
        )
        vespa_app_name = os.getenv("VESPA_EVALS_APP_NAME", "dbevals")
        tenant_name = os.getenv("VESPA_TENANT_NAME", "vespa-dev")
        instance_name = os.getenv("VESPA_EVALS_INSTANCE_NAME", "default")

        self.vespa_repo = VespaRepository.connect_to_vespa_cloud(
            url=vespa_url,
            app_name=vespa_app_name,
            tenant_name=tenant_name,
            instance_name=instance_name,
        )

        if not self.vespa_repo:
            raise ValueError(
                f"Failed to connect to Vespa application {vespa_app_name} "
                f"for tenant {tenant_name} and instance {instance_name}.",
            )

    async def forward_queries(
        self,
        queries: list[str] | str,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Embed queries using the embedder."""
        enriched_queries = [queries] if isinstance(queries, str) else queries
        query_embeddings = await self.embeddings_repo.embed_queries(enriched_queries)

        assert len(query_embeddings) == len(enriched_queries), (
            f"Query embeddings and enriched queries must be the same length, {len(query_embeddings)} != {len(enriched_queries)}"
        )

        return query_embeddings

    async def forward_passages(
        self,
        passages: list[Any],
        **kwargs,
    ) -> list[torch.Tensor]:
        """Embed passages (images) using the embedder."""
        passage_embeddings = await self.embeddings_repo.embed_images(passages)

        return passage_embeddings

    async def retrieve(
        self,
        queries: list[str] | str,
        num_results: int = 10,
        variant: str = "ann_float",
        document_filters: list[tuple[str, str]] | None = None,
    ) -> list[VDBQueryResponse]:
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)

        async with VespaAsync(
            self.vespa_repo,
            connections=4,
            limits=limits,
            timeout=200,
        ) as vespa_session:
            vespa_repository = VespaRepository(vespa_session, self.vespa_repo)
            vespa_service = VespaService(vespa_repository)

            retrieval_service = RetrievalService(self.embeddings_repo, vespa_service)
            returned_responses = await retrieval_service.retrieve(
                queries=queries,
                num_results=num_results,
                document_filters=document_filters,
                variant=variant,
                schema_name="evaluation_dataset",
                kwargs={
                    "embedding_field_name": "embedding",
                    "binary_embedding_field_name": "binary_embedding",
                },
            )

            return returned_responses
