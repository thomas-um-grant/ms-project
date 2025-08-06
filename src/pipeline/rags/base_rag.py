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


class BaseRAG(ABC):
    def __init__(
        self,
        name: str,
        data_dir: Path,
        configs: dict,
    ):
        self.name = name
        knowledge_base = configs.get("knowledge_base", "default")
        for prefix in ["vidore/", "sherpa/"]:
            knowledge_base = knowledge_base.removeprefix(prefix)
        self.knowledge_base = knowledge_base

        self.data_dir = data_dir / name
        self.device_config = DeviceConfig.auto_detect(configs.get("preferred_device"))

        # Prepare store structure
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store_dir = self.data_dir / self.knowledge_base / "store"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.data_dir / self.knowledge_base / "metadata.json"
        self.embeddings_ids_path = (
            self.store_dir / self.knowledge_base / "embeddings_ids.jsonl"
        )
        self.embeddings_path = self.store_dir / self.knowledge_base / "embeddings.pt"

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
        self.top_k = configs.get("top_k", 5)  # Number of top results to retrieve
        self.pruning_threshold = configs.get(
            "pruning_threshold",
            0.0,
        )  # % of results to prune
        self.batch_size = configs.get("batch_size", 8)

    @abstractmethod
    async def extract(self):
        """Extract relevant corpuses for retrieval."""

    @abstractmethod
    async def index(self, corpuses: list):
        """Index corpuses for retrieval."""

    @abstractmethod
    async def retrieve(self, queries: list, top_k: int | None = None):
        """Retrieve the most relevant corpuses for the given queries."""

    @abstractmethod
    async def answer(self, queries: list):
        """Generate answers based on the retrieved corpuses."""
