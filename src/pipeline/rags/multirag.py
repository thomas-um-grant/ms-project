import asyncio
import logging
from pathlib import Path
from typing import Any

from pipeline.rags.base_rag import BaseRAG
from pipeline.rags.multimodal import MultiModalRAG
from pipeline.rags.traditional import TraditionalRAG

logger = logging.getLogger(__name__)


class MultiRAG(BaseRAG):
    """Device-agnostic Multi RAG system."""

    _EPS = 1e-12

    def __init__(
        self,
        name: str,
        data_dir: Path,
        configs: dict | None = None,
        *,
        disable_generation: bool = False,
    ):
        super().__init__(name, data_dir, configs)
        # Extract configuration for each RAG component
        self.configs = configs or {}
        traditional_config = self.configs.get("traditional", {})
        multimodal_config = self.configs.get("multimodal", {})

        # Instanciate Traditional RAG
        self.traditional_rag = TraditionalRAG(
            name=traditional_config.get("name", "traditional"),
            data_dir=data_dir,
            configs=traditional_config.get("configs", {}),
            disable_generation=disable_generation,
        )

        # Instanciate Multimodal RAG
        self.multimodal_rag = MultiModalRAG(
            name=multimodal_config.get("name", "multimodal"),
            data_dir=data_dir,
            configs=multimodal_config.get("configs", {}),
        )

    async def extract(
        self,
        documents: list[Path],
        *,
        preprocessed: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """Extract relevant corpuses for retrieval."""
        # Run sub-RAG extracts in parallel
        await asyncio.gather(
            self.traditional_rag.extract(
                documents,
                preprocessed=preprocessed,
                batch_size=batch_size,
            ),
            self.multimodal_rag.extract(
                documents,
                preprocessed=preprocessed,
                batch_size=batch_size,
            ),
        )

    async def index(self) -> None:
        """Index documents for retrieval."""
        await asyncio.gather(
            self.traditional_rag.index(),
            self.multimodal_rag.index(),
        )

    async def retrieve(
        self,
        queries: str | list[str],
        top_k: int | None = None,
    ) -> list[list[tuple[dict, float]]]:
        query_texts = [queries] if isinstance(queries, str) else queries
        if not query_texts:
            return []
        buffer_k = min(top_k, 100) if top_k else 100

        trad_results, multi_results = await self._retrieve_subsystems(
            query_texts,
            buffer_k,
        )

        # Sub-RAGs perform their own reranking inside their retrieve() methods.
        # So we can safely merge by scores here since the same embedding model is used in each sub-RAG.
        merged = self._merge_results(query_texts, [trad_results, multi_results])

        if top_k is not None:
            return [res[:top_k] for res in merged]
        return merged

    async def _retrieve_subsystems(
        self,
        query_texts: list[str],
        buffer_k: int,
    ) -> tuple[list[list[tuple[dict, float]]], list[list[tuple[dict, float]]]]:
        """Retrieve from traditional & multimodal subsystems in parallel."""
        trad, multi = await asyncio.gather(
            self.traditional_rag.retrieve(query_texts, top_k=buffer_k),
            self.multimodal_rag.retrieve(query_texts, top_k=buffer_k),
        )
        return trad, multi

    def _merge_results(
        self,
        query_texts: list[str],
        systems_results: list[list[list[tuple[dict, float]]]],
    ) -> list[list[tuple[dict, float]]]:
        """Merge results keeping max score for duplicate corpus-ids and provenance."""
        merged_all: list[list[tuple[dict, float]]] = []
        num_queries = len(query_texts)
        for qi in range(num_queries):
            accumulator: dict[str, tuple[dict, float]] = {}
            for sys_idx, sys_results in enumerate(systems_results):
                source_label = "traditional" if sys_idx == 0 else "multimodal"
                query_list = sys_results[qi] if qi < len(sys_results) else []
                for metadata, score in query_list:
                    corpus_id = metadata.get("corpus-id")
                    doc_id = metadata.get("doc-id", "")
                    cid = f"{corpus_id}{f'_{doc_id}' if doc_id else ''}"
                    if not cid:
                        continue

                    metadata.setdefault("source_rag", source_label)
                    if cid in accumulator:
                        if score > accumulator[cid][1]:
                            accumulator[cid] = (metadata, score)
                    else:
                        # Warning, should never get in here
                        logger.warning("Unexpected document ID format: %s", metadata)
                        if "corpus-id" not in metadata:
                            metadata["corpus-id"] = metadata.get("name", "")
                        accumulator[cid] = (metadata, score)

            merged_sorted = sorted(
                accumulator.values(),
                key=lambda x: x[1],
                reverse=True,
            )
            merged_all.append(merged_sorted)
        return merged_all

    async def answer(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[str, list[tuple[dict, float]]]:
        """Generate answer based on the retrieved images or documents."""
        # Use configured top_k if not provided
        if top_k is None:
            top_k = self.top_k

        # Retrieve the most relevant images or documents
        results = await self.retrieve(query, top_k=top_k)

        # Generate answers from the retrieved results
        context: list[Any] = []
        for metadata, _ in results[0]:
            # Use 'corpus-id' for consistent file access
            doc_id = metadata.get("corpus-id", metadata.get("name", ""))
            if not doc_id:
                logger.warning(
                    "Document metadata missing 'corpus-id' and 'name'. Skipping context generation for this document.",
                )
                continue

            if doc_id.endswith((".txt", ".md")):
                item_type = "text"
            elif doc_id.endswith((".jpg", ".jpeg", ".png")):
                item_type = "image"
            else:
                item_type = "unknown"

            if item_type == "text":
                text_path = self.traditional_rag.corpuses_dir / doc_id
                try:
                    context_chunks = []
                    context_chunks.append(text_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError) as exc:
                    logger.debug("Failed adding context %s: %s", text_path, exc)

                # Join context and truncate if too long
                joined_context = "\n\n".join(context_chunks)[:15000]

                context.append(
                    {
                        "type": "text",
                        "text": joined_context,
                    },
                )
            elif item_type == "image":
                context.append(
                    {
                        "type": "image",
                        "image": str(
                            self.multimodal_rag.corpuses_dir / doc_id,
                        ),
                    },
                )

        if self.generation_disabled or self.generation_model is None:
            msg = "Generation disabled: only retrieval results returned. Enable generation to produce answers."
            return msg, results[0]

        response = await self.generation_model.generate(query, context=context)
        return response, results[0]
