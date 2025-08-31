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
            # Mark and exit early; children will perform their own setup.
            self.is_multi_parent = True
            logger.debug(
                "Skipping BaseRAG storage/model setup for multi parent (type=%s, skip_storage=%s)",
                rag_type,
                skip_storage_flag,
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

        # New suffixed file names
        suffixed_embeddings_file = f"embeddings_{self.embedding_model_tag}.pt"
        suffixed_ids_file = f"embeddings_ids_{self.embedding_model_tag}.jsonl"

        new_embeddings_path = self.store_dir / suffixed_embeddings_file
        new_embeddings_ids_path = self.store_dir / suffixed_ids_file

        legacy_embeddings_path = self.store_dir / "embeddings.pt"
        legacy_ids_path = self.store_dir / "embeddings_ids.jsonl"

        # Backward compatibility: if legacy files exist and no suffixed files yet,
        # reuse legacy paths so we don't duplicate or orphan existing data.
        if (
            legacy_embeddings_path.exists()
            and legacy_ids_path.exists()
            and not new_embeddings_path.exists()
            and not new_embeddings_ids_path.exists()
        ):
            self.embeddings_path = legacy_embeddings_path
            self.embeddings_ids_path = legacy_ids_path
            self.using_legacy_embedding_files = True  # diagnostic flag
            logger.info(
                "Using legacy embedding files (no suffixed files present) for model tag '%s'",
                self.embedding_model_tag,
            )
        else:
            self.embeddings_path = new_embeddings_path
            self.embeddings_ids_path = new_embeddings_ids_path
            self.using_legacy_embedding_files = False
            logger.info(
                "Embedding files set to suffixed variants: %s / %s",
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
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "Failed to initialize generation model (%s); continuing without it.",
                    exc,
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

        # Reranker configuration (e.g. 'jina'). If provided, we enable auto-rerank by default
        self.reranker_method: str | None = configs.get("reranker")
        # Allow user override; default True when a reranker is defined
        self.auto_rerank: bool = configs.get(
            "auto_rerank",
            bool(self.reranker_method),
        )
        if self.reranker_method and self.auto_rerank:
            logger.info(
                "Auto-reranking enabled using method '%s'",
                self.reranker_method,
            )
        elif self.reranker_method:
            logger.info(
                "Reranker '%s' configured but auto_rerank disabled (manual rerank required)",
                self.reranker_method,
            )

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

        # Only proceed if the retrieval embedding model itself is Jina
        if not isinstance(getattr(self, "embedding_model", None), JinaV4Model):
            logger.debug(
                "Rerank method 'jina' requested but current embedding model is not Jina; bypassing.",
            )
            return retrieved_corpuses

        if not hasattr(self, "_jina_reranker"):
            self._jina_reranker = JinaRerankerFactory.create_reranker(
                embeddings_path=self.embeddings_path,
                embeddings_ids_path=self.embeddings_ids_path,
                device_config=self.device_config,
                embedding_model=self.embedding_model,  # reuse instance to save memory
            )

        # Guard: ensure embeddings exist; if not, bypass silently
        if not self._jina_reranker.load_store_embeddings():
            logger.warning(
                "Jina store embeddings missing; skipping rerank (indexing with Jina required).",
            )
            return retrieved_corpuses

        try:
            return await self._jina_reranker.rerank(queries, retrieved_corpuses)
        except Exception as exc:  # pragma: no cover - fail open
            logger.exception(
                "Jina reranking failed; returning original results: %s",
                exc,
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
            # Allow future config injection via self.config.get('llm_reranker', {})
            llm_conf = (
                self.config.get("llm_reranker", {}) if hasattr(self, "config") else {}
            )
            try:
                self._llm_reranker: LLMReranker = LLMRerankerFactory.create_reranker(
                    **llm_conf,
                )
            except Exception as exc:  # pragma: no cover - fail open
                logger.warning("Failed to initialize LLM reranker: %s", exc)
                return retrieved_corpuses

        try:
            return await self._llm_reranker.rerank(queries, retrieved_corpuses)
        except Exception:  # pragma: no cover - fail open
            logger.exception("LLM reranking failed; returning original ranking")
            return retrieved_corpuses

    async def ensure_jina_reranker_embeddings(self) -> None:
        """
        Compatibility no-op: reranker now directly reuses retrieval embeddings.

        Kept for external callers; will lazily validate presence of store embeddings
        and initialize reranker instance if possible.
        """
        try:
            from pipeline.models.embedding_models import JinaV4Model
            from pipeline.rerankers import JinaRerankerFactory
        except ImportError:  # pragma: no cover
            return

        if not isinstance(getattr(self, "embedding_model", None), JinaV4Model):
            return

        if not hasattr(self, "_jina_reranker"):
            self._jina_reranker = JinaRerankerFactory.create_reranker(
                embeddings_path=self.embeddings_path,
                embeddings_ids_path=self.embeddings_ids_path,
                device_config=self.device_config,
                embedding_model=self.embedding_model,
            )
        # Attempt a load (no error if missing); this primes internal index map.
        self._jina_reranker.load_store_embeddings()
