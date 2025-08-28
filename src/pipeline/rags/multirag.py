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

    def __init__(
        self,
        name: str,
        data_dir: Path,
        configs: dict | None = None,
    ):
        # Extract configuration for each RAG component
        self.configs = configs or {}
        traditional_config = self.configs.get("traditional", {})
        multimodal_config = self.configs.get("multimodal", {})
        # graph_config = self.configs.get("graph", {})

        # Instanciate Traditional RAG
        self.traditional_rag = TraditionalRAG(
            name,
            data_dir,
            configs=traditional_config,
        )

        # Instanciate Multimodal RAG
        self.multimodal_rag = MultiModalRAG(
            name,
            data_dir,
            configs=multimodal_config,
        )

        # Instanciate Graph RAG
        # self.graph_rag = GraphRAG(name, data_dir, configs=graph_config)

    async def extract(
        self,
        documents: list[Path],
        *,
        preprocessed: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """Extract relevant corpuses for retrieval."""
        # Run in parallel
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
        # Convert single query to list for uniform processing
        query_texts = [queries] if isinstance(queries, str) else queries

        # Run in parallel
        results = await asyncio.gather(
            self.traditional_rag.retrieve(queries, top_k=top_k),
            self.multimodal_rag.retrieve(queries, top_k=top_k),
        )

        # TODO: Routing -> Advantage a rag system based on the query type checked against templates

        # TODO: Prune down irrelevant results (lower than 0.3 match)
        # TODO: Remove duplicates across rags
        all_results = []
        existing_ids = set()
        for sublist in results:
            query_results = []
            for metadata, score in sublist:
                if score > 0.3 and metadata["name"] not in existing_ids:
                    query_results.append((metadata, score))
                    existing_ids.add(metadata["name"])

            all_results.append(query_results)

        # Rerank, use default if none found
        try:
            return await self.rerank(
                queries=query_texts,
                retrieved_corpuses=all_results,
                method=self.reranker_method,
            )
        except Exception:  # pragma: no cover - fail open
            logger.exception(
                "Automatic rerank failed; returning raw retrieval results",
            )

        return all_results[:top_k]

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
            if metadata.get("name", "").endswith((".txt", ".md")):
                item_type = "text"
            elif metadata.get("name", "").endswith((".jpg", ".jpeg", ".png")):
                item_type = "image"
            else:
                item_type = "unknown"

            if item_type == "text":
                text_path = self.traditional_rag.corpuses_dir / metadata["name"]
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
                            self.multimodal_rag.corpuses_dir / metadata["name"],
                        ),
                    },
                )

        if self.generation_disabled or self.generation_model is None:
            msg = "Generation disabled: only retrieval results returned. Enable generation to produce answers."
            return msg, results[0]

        response = await self.generation_model.generate(query, context=context)
        return response, results[0]
