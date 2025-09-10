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
        *,
        disable_generation: bool = False,
    ):
        self.name = name

        if "multi" in configs.get("type", ""):
            # Mark and exit early; children will perform their own setup (MultiRAG orchestrates sub-systems).
            self.is_multi_parent = True
            logger.debug(
                "Skipping BaseRAG storage/model setup for multi parent (type=%s)",
                configs.get("type"),
            )
            return

        self.is_multi_parent = False

        # Load defaults for fallback values
        defaults = self._load_defaults()
        processing_defaults = defaults["processing"]

        knowledge_base = configs.get("knowledge_base", "default")
        for prefix in ["vidore/", "sherpa/", "beir/"]:
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

        # Determine embedding model tag
        self.embedding_model_tag = configs.get("embedding_model", "unknown")

        # Embedding storage
        self.embeddings_path = (
            self.store_dir / f"embeddings_{self.embedding_model_tag}.pt"
        )
        self.embeddings_ids_path = (
            self.store_dir / f"embeddings_ids_{self.embedding_model_tag}.jsonl"
        )
        logger.info(
            "Embedding files: %s / %s",
            self.embeddings_path.name,
            self.embeddings_ids_path.name,
        )

        # Setup models (optionally disable generation model to save memory during pure retrieval workflows)
        self.embedding_model = setup_embedding_model(
            configs.get("embedding_model"),
            device_config=self.device_config,
        )
        self.generation_model = None
        self.generation_disabled = disable_generation or configs.get(
            "disable_generation",
            False,
        )
        if not self.generation_disabled:
            try:
                self.generation_model = setup_generation_model(
                    configs.get("generation_model"),
                    device_config=self.device_config,
                )
            except Exception:  # pragma: no cover - defensive fallback
                logger.exception(
                    "Failed to initialize generation model; continuing without it.",
                )
                self.generation_disabled = True
        else:
            logger.info("Generation model loading disabled (disable_generation=True)")

        # Prepare configurations
        if not configs:  # If no configs provided, use defaults
            configs = {}

        self.chunking_strategy: ChunkingStrategy = configs.get("chunking_strategy")
        self.query_enhancement: QueryEnhancement = configs.get("query_enhancement")
        self.retrieval_strategy: RetrievalStrategy = configs.get("retrieval_strategy")
        self.similarity_metric: SimilarityMetric = configs.get("similarity_metric")
        self.routing_strategy: RoutingStrategy = configs.get("routing_strategy")

        # ------------------------------------------------------------------
        # Reranker configuration:
        # { "reranker": { "name": "jina", "configs": { "embedding_model": "jina_embed" } } }
        # or
        # { "reranker": { "name": "llm", "configs": { "content_mode": "full", ... } } }
        # ------------------------------------------------------------------
        self.reranker_method: str | None = None
        self.reranker_configs: dict = {}

        rer_conf = configs.get("reranker", {})
        self.reranker_method = rer_conf.get("name")
        self.reranker_configs = rer_conf.get("configs", {}) or {}
        # Allow specifying auto_rerank inside the reranker dict
        self.auto_rerank = rer_conf.get(
            "auto_rerank",
            True,
        )

        # Derive Jina reranker embedding
        if self.reranker_method == "jina":
            self.jina_reranker_tag = (
                self.reranker_configs.get("embedding_model") or "jina_embed"
            )

            self.jina_rerank_embeddings_path = (
                self.store_dir / f"embeddings_{self.jina_reranker_tag}.pt"
            )
            self.jina_rerank_embeddings_ids_path = (
                self.store_dir / f"embeddings_ids_{self.jina_reranker_tag}.jsonl"
            )
            self._jina_reranker_missing_logged = False

            if self.reranker_method and self.auto_rerank:
                logger.info(
                    "Auto-reranking enabled using method '%s' (new-style=%s)",
                    self.reranker_method,
                    isinstance(rer_conf, dict) and bool(rer_conf.get("name")),
                )
            elif self.reranker_method:
                logger.info(
                    "Reranker '%s' configured but auto_rerank disabled (manual rerank required)",
                    self.reranker_method,
                )

            try:  # pragma: no cover
                from pipeline.models.embedding_models import (
                    JinaV4Model,  # type: ignore[attr-defined]
                )

                if isinstance(getattr(self, "embedding_model", None), JinaV4Model):
                    logger.info(
                        "Skipping external Jina reranker: retrieval embedding model is already Jina.",
                    )
                    # Disable automatic rerank to avoid redundant scoring
                    self.auto_rerank = False
            except ImportError:  # pragma: no cover
                pass

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
        method: str = "jina",  # jina or llm
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
            from pipeline.models.embedding_models import JinaV4Model
            from pipeline.rerankers import JinaRerankerFactory
        except ImportError:  # pragma: no cover
            logger.warning("Jina reranker not available; skipping rerank")
            return retrieved_corpuses

        # If retrieval model already Jina, no external reranking needed
        if isinstance(getattr(self, "embedding_model", None), JinaV4Model):
            logger.debug(
                "Retrieval model is Jina; external Jina reranker bypassed.",
            )
            return retrieved_corpuses

        # Lazy init dedicated reranker bound to Jina embedding store
        if not hasattr(self, "_jina_reranker"):
            self._jina_reranker = JinaRerankerFactory.create_reranker(
                embeddings_path=self.jina_rerank_embeddings_path,
                embeddings_ids_path=self.jina_rerank_embeddings_ids_path,
                device_config=self.device_config,
                embedding_model=None,  # let factory/model init its own Jina model
            )

        # Ensure Jina reranker embeddings are available
        if not self._jina_reranker.load_store_embeddings():
            if not self._jina_reranker_missing_logged:
                logger.warning(
                    "Jina reranker embeddings missing (%s); skipping rerank.",
                    self.jina_rerank_embeddings_path.name,
                )
                self._jina_reranker_missing_logged = True
            return retrieved_corpuses

        try:
            return await self._jina_reranker.rerank(queries, retrieved_corpuses)
        except Exception:  # pragma: no cover - fail open
            logger.exception(
                "Jina reranking failed; returning original results",
            )
            return retrieved_corpuses

    async def _rerank_with_llm(
        self,
        queries: str | list[str],
        retrieved_corpuses: list[list[tuple[dict, float]]],
    ) -> list[list[tuple[dict, float]]]:
        """
        Rerank using a Gemini LLM-based scoring model.

        Uses `pipeline.rerankers.LLMReranker` which calls Gemini to obtain a
        JSON ranking for up to the top 20 (default) items per query. Falls back
        to original ordering on any error.
        """
        try:
            from pipeline.rerankers import LLMReranker, LLMRerankerFactory
        except ImportError:  # pragma: no cover
            logger.warning("LLM reranker package not available")
            return retrieved_corpuses

        # Lazy init (single instance per BaseRAG)
        if not hasattr(self, "_llm_reranker"):
            llm_conf = self.reranker_configs
            try:
                self._llm_reranker: LLMReranker = LLMRerankerFactory.create_reranker(
                    **llm_conf,
                )
            except Exception:  # pragma: no cover - fail open
                logger.exception("Failed to initialize LLM reranker")
                return retrieved_corpuses

        try:
            return await self._llm_reranker.rerank(queries, retrieved_corpuses)
        except Exception:  # pragma: no cover - fail open
            logger.exception("LLM reranking failed; returning original ranking")
            return retrieved_corpuses

    async def ensure_jina_reranker_embeddings(self) -> None:
        """
        Prime external Jina reranker by loading its dedicated embedding store.

        Skips when:
          * Retrieval embedding model already Jina (no external rerank used), or
          * Jina reranker not configured.

        Does NOT build embeddings; assumes prior indexing run with the Jina
        embedding model generated:
            embeddings_{tag}.pt
            embeddings_ids_{tag}.jsonl
        where tag = self.jina_reranker_tag (default 'jina_embed').
        """
        if self.reranker_method != "jina":
            return
        try:  # pragma: no cover
            from pipeline.models.embedding_models import (
                JinaV4Model,  # type: ignore[attr-defined]
            )
            from pipeline.rerankers import (
                JinaRerankerFactory,  # type: ignore[attr-defined]
            )
        except ImportError:  # pragma: no cover
            return

        # If retrieval model already Jina, nothing to prime
        if isinstance(getattr(self, "embedding_model", None), JinaV4Model):
            return

        if not hasattr(self, "_jina_reranker"):
            self._jina_reranker = JinaRerankerFactory.create_reranker(
                embeddings_path=self.jina_rerank_embeddings_path,
                embeddings_ids_path=self.jina_rerank_embeddings_ids_path,
                device_config=self.device_config,
            )
        self._jina_reranker.load_store_embeddings()
