import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.strategy import (
        ChunkingStrategy,
        QueryEnhancement,
        RetrievalStrategy,
        RoutingStrategy,
        SimilarityMetric,
    )

from pipeline.models.embedding_models import setup_embedding_model
from pipeline.models.generation_models import setup_generation_model
from utils.device import DeviceConfig

logger = logging.getLogger(__name__)


class BaseRAG(ABC):
    @classmethod
    def _load_defaults(cls) -> dict:
        """Load default configuration values."""
        defaults_path = (
            Path(__file__).parent.parent.parent / "configs" / "defaults.json"
        )
        with defaults_path.open() as f:
            return json.load(f)

    def __init__(
        self,
        name: str,
        data_dir: Path,
        configs: dict,
    ):
        # Load defaults for fallback values
        defaults = self._load_defaults()
        processing_defaults = defaults["processing"]

        self.name = name
        knowledge_base = configs.get("knowledge_base", "default")
        for prefix in ["vidore/", "sherpa/"]:
            knowledge_base = knowledge_base.removeprefix(prefix)
        self.knowledge_base = knowledge_base

        self.data_dir = data_dir / self.name / self.knowledge_base
        self.device_config = DeviceConfig.auto_detect(configs.get("preferred_device"))

        # Prepare store structure
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.data_dir / "metadata.json"

        self.corpuses_dir = self.data_dir / "corpuses"
        self.corpuses_dir.mkdir(parents=True, exist_ok=True)

        self.store_dir = self.data_dir / "store"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_ids_path = self.store_dir / "embeddings_ids.jsonl"
        self.embeddings_path = self.store_dir / "embeddings.pt"

        # Setup models
        self.embedding_model = setup_embedding_model(
            configs.get("embedding_model"),
            device_config=self.device_config,
        )
        self.generation_model = setup_generation_model(
            configs.get("generation_model"),
            device_config=self.device_config,
        )

        # Prepare configurations
        if not configs:  # If no configs provided, use defaults
            configs = {}

        self.chunking_strategy: ChunkingStrategy = configs.get("chunking_strategy")
        self.query_enhancement: QueryEnhancement = configs.get("query_enhancement")
        self.retrieval_strategy: RetrievalStrategy = configs.get("retrieval_strategy")
        self.similarity_metric: SimilarityMetric = configs.get("similarity_metric")
        self.routing_strategy: RoutingStrategy = configs.get("routing_strategy")

        # Use configuration defaults with user overrides
        self.top_k = configs.get("top_k", processing_defaults["retrieval"]["top_k"])
        self.pruning_threshold = configs.get(
            "pruning_threshold",
            processing_defaults["retrieval"]["pruning_threshold"],
        )
        self.batch_size = configs.get(
            "batch_size",
            processing_defaults["indexing"]["batch_size"],
        )

        # Store full configuration for passing to helper classes
        self.config = configs

    @abstractmethod
    async def extract(
        self,
        documents: list[Path],
        *,
        preprocessed: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """Extract relevant corpuses for retrieval."""

    @abstractmethod
    async def index(self) -> None:
        """Index corpuses for retrieval."""

    @abstractmethod
    async def retrieve(
        self,
        queries: str | list[str],
        top_k: int | None = None,
    ) -> list[list[tuple[dict, float]]]:
        """Retrieve the most relevant corpuses for the given queries."""

    @abstractmethod
    async def answer(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[str, list[tuple[dict, float]]]:
        """Generate answers based on the retrieved corpuses."""

    async def rerank(
        self,
        queries: str | list[str],
        retrieved_corpuses: list[list[tuple[dict, float]]],
        method: str = "llm",  # llm or jina
    ) -> list[list[tuple[dict, float]]]:
        """Rerank the retrieved corpuses based on the queries."""
        if method == "jina":
            return await self._rerank_with_jina(queries, retrieved_corpuses)
        if method == "llm":
            return await self._rerank_with_llm(queries, retrieved_corpuses)
        return retrieved_corpuses

    async def _rerank_with_jina(
        self,
        queries: str | list[str],
        retrieved_corpuses: list[list[tuple[dict, float]]],
    ) -> list[list[tuple[dict, float]]]:
        """Rerank using Jina embeddings."""
        try:
            from pipeline.rerankers import JinaRerankerFactory

            # Create reranker if not exists
            if not hasattr(self, "_jina_reranker"):
                self._jina_reranker = JinaRerankerFactory.create_reranker(
                    rag_type=self.name.split("_")[0],  # Extract RAG type from name
                    corpus_dir=self.corpuses_dir,
                    device_config=self.device_config,
                )

                # Pre-compute corpus embeddings if needed
                metadata = self.metadata_manager.load_metadata()
                await self._jina_reranker.precompute_corpus_embeddings(metadata)

            return await self._jina_reranker.rerank(queries, retrieved_corpuses)

        except ImportError:
            logger.warning("Jina reranker not available, returning original ranking")
            return retrieved_corpuses
        except Exception as e:
            logger.error(f"Error in Jina reranking: {e}")
            return retrieved_corpuses

    async def _rerank_with_llm(
        self,
        queries: str | list[str],
        retrieved_corpuses: list[list[tuple[dict, float]]],
    ) -> list[list[tuple[dict, float]]]:
        """Rerank using LLM-based scoring."""
        # TODO: Implement LLM-based reranking
        logger.info("LLM reranking not yet implemented, returning original ranking")
        return retrieved_corpuses
