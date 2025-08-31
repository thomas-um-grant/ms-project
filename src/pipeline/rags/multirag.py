import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.rags.base_rag import BaseRAG
from pipeline.rags.multimodal import MultiModalRAG
from pipeline.rags.traditional import TraditionalRAG

logger = logging.getLogger(__name__)


class MultiRAG(BaseRAG):
    """
    Device-agnostic Multi RAG system.

    Fusion strategies (set via configs["fusion"]).
    Supported methods:
        - "max" (default): keep the max score for duplicate ids
        - "normalize_average": L1/L2 normalise per modality then average scores
        - "rerank_fuse": L1/L2 normalise per modality then interleave by rank

    Config example:
    {
        "fusion": {
            "method": "normalize_average" | "rerank_fuse" | "max",
            "norm": "l1" | "l2"
        }
    }
    """

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
        # Fusion configuration
        fusion_cfg = self.configs.get("fusion", {})
        self.fusion_method: str = fusion_cfg.get("method", "max")
        self.fusion_norm: str = fusion_cfg.get("norm", "l1")

        # Instantiate Traditional RAG
        self.traditional_rag = TraditionalRAG(
            name=traditional_config.get("name", "traditional"),
            data_dir=data_dir,
            configs=traditional_config.get("configs", {}),
            disable_generation=disable_generation,
        )

        # Instantiate Multimodal RAG
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
        *,
        fusion_method: str | None = None,
        norm: str | None = None,
    ) -> list[list[tuple[dict, float]]]:
        query_texts = [queries] if isinstance(queries, str) else queries
        if not query_texts:
            return []
        buffer_k = min(top_k, 100) if top_k else 100

        trad_results, multi_results = await self._retrieve_subsystems(
            query_texts,
            buffer_k,
        )

        # Choose fusion strategy
        fused = self._fuse(
            query_texts,
            trad_results,
            multi_results,
            fusion_method,
            norm,
        )

        if top_k is not None:
            return [res[:top_k] for res in fused]
        return fused

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

    def _fuse(
        self,
        query_texts: list[str],
        trad_results: list[list[tuple[dict, float]]],
        multi_results: list[list[tuple[dict, float]]],
        fusion_method: str | None,
        norm: str | None,
    ) -> list[list[tuple[dict, float]]]:
        method = (fusion_method or self.fusion_method or "max").lower()
        norm_type = (norm or self.fusion_norm or "l2").lower()

        fused_all: list[list[tuple[dict, float]]] = []
        for qi, _ in enumerate(query_texts):
            trad_list = trad_results[qi] if qi < len(trad_results) else []
            multi_list = multi_results[qi] if qi < len(multi_results) else []

            if method == "normalize_average":
                fused = self._normalize_and_average(trad_list, multi_list, norm_type)
            elif method == "rerank_fuse":
                fused = self._rerank_and_fuse(trad_list, multi_list, norm_type)
            elif method == "max":
                fused = self._merge_max(trad_list, multi_list)
            else:
                logger.warning(
                    "Unknown fusion method '%s', falling back to 'max'",
                    method,
                )
                fused = self._merge_max(trad_list, multi_list)
            fused_all.append(fused)
        return fused_all

    @staticmethod
    def _extract_id(md: dict) -> str:
        corpus_id = md.get("corpus-id", "")
        doc_id = md.get("doc-id")
        if doc_id:
            return f"{corpus_id}_{doc_id}"
        return corpus_id

    def _merge_max(
        self,
        trad_list: list[tuple[dict, float]],
        multi_list: list[tuple[dict, float]],
    ) -> list[tuple[dict, float]]:
        accumulator: dict[str, tuple[dict, float]] = {}
        for source_label, results in (
            ("traditional", trad_list),
            ("multimodal", multi_list),
        ):
            for metadata, score in results:
                cid = self._extract_id(metadata)
                if not cid:
                    continue
                metadata.setdefault("source_rag", source_label)
                if cid in accumulator:
                    if score > accumulator[cid][1]:
                        accumulator[cid] = (metadata, score)
                else:
                    accumulator[cid] = (metadata, score)
        merged_sorted = sorted(accumulator.values(), key=lambda x: x[1], reverse=True)
        return merged_sorted

    def _normalize_and_average(
        self,
        traditional_scores: list[tuple[dict, float]],
        multimodal_scores: list[tuple[dict, float]],
        norm: str = "l1",
    ) -> list[tuple[dict, float]]:
        """
        Normalize two lists of scores per modality and compute fused scores by averaging. Matches items by metadata 'id' and handles missing modalities.

        Steps:
            1. Build id -> score mapping for each modality.
            2. Normalize scores per modality (L1 or L2).
            3. For each unique id:
                - Collect normalized scores from available modalities.
                - Compute the mean (fused score).
                - Preserve metadata and mark 'type' as traditional/multimodal/both.
            4. Sort the fused list descending by fused score.

        Args:
            traditional_scores: list of tuples (metadata, traditional/text score)
            multimodal_scores: list of tuples (metadata, multimodal/image score)
            norm: normalization method, "l1" or "l2"

        Returns:
            List of tuples (metadata, fused_score) sorted descending by score.

        """

        # Build id -> (metadata, score)
        def build_map(items: Iterable[tuple[dict, float]]):
            m = {}
            for md, sc in items:
                corpus_id = md.get("corpus-id")
                doc_id = md.get("doc-id")
                cid = f"{corpus_id}{f'_{doc_id}' if doc_id else ''}"

                if cid is None:
                    continue

                if cid not in m:
                    m[cid] = (md, float(sc))

            return m

        traditional_map = build_map(traditional_scores)
        multimodal_map = build_map(multimodal_scores)

        traditional_vals = np.array(
            [v[1] for v in traditional_map.values()],
            dtype=float,
        )
        multimodal_vals = np.array([v[1] for v in multimodal_map.values()], dtype=float)

        def normalize(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            if norm == "l1":
                return arr / (np.sum(arr) + 1e-12)
            if norm == "l2":
                return arr / (np.linalg.norm(arr) + 1e-12)
            raise ValueError(f"Unknown norm type: {norm}")

        traditional_norm_vals = normalize(traditional_vals)
        multimodal_norm_vals = normalize(multimodal_vals)

        traditional_norm_map = {
            k: traditional_norm_vals[i] for i, k in enumerate(traditional_map.keys())
        }
        multimodal_norm_map = {
            k: multimodal_norm_vals[i] for i, k in enumerate(multimodal_map.keys())
        }

        all_ids = set(traditional_map.keys()) | set(multimodal_map.keys())
        fused: list[tuple[dict, float]] = []
        for _id in all_ids:
            vals = []
            metadata = None
            if _id in multimodal_norm_map:
                vals.append(multimodal_norm_map[_id])
            if _id in traditional_norm_map:
                vals.append(traditional_norm_map[_id])
                metadata = traditional_map[_id][0]
                if metadata is None:
                    metadata = multimodal_map[_id][0]
            if metadata is None:
                continue
            metadata = metadata.copy()
            metadata["fusion_type"] = (
                "both"
                if _id in traditional_norm_map and _id in multimodal_norm_map
                else ("traditional" if _id in traditional_norm_map else "multimodal")
            )
            metadata["fusion_method"] = "normalize_average"
            fused_score = float(np.mean(vals)) if vals else 0.0
            fused.append((metadata, fused_score))

        return sorted(fused, key=lambda x: x[1], reverse=True)

    def _rerank_and_fuse(
        self,
        traditional_scores: list[tuple[dict, float]],
        multimodal_scores: list[tuple[dict, float]],
        norm: str = "l1",
    ) -> list[tuple[dict, float]]:
        """
        Rerank by assembling two streams of scores, taking the top-ranked items from each modality and skipping duplicates.

        Steps:
            1. Normalize each modality's scores (L1 or L2).
            2. Replace original scores with normalized values.
            3. Sort each modality descending by normalized score.
            4. Fuse lists by iterating through ranks (round-robin):
                - Add top-ranked item from each list in order.
                - Skip items already added (avoid duplicates by 'id').
            5. Return fused list up to the length of the traditional list.

        Args:
            traditional_scores: list of tuples (metadata, traditional/text score)
            multimodal_scores: list of tuples (metadata, multimodal/image score)
            norm: normalization method, "l1" or "l2"

        Returns:
            List of tuples (metadata, normalized score) fused by rank.

        """

        def build_list(items: list[tuple[dict, float]]):
            reduced: dict[str, tuple[dict, float]] = {}
            for md, sc in items:
                corpus_id = md.get("corpus-id")
                doc_id = md.get("doc-id")
                cid = f"{corpus_id}{f'_{doc_id}' if doc_id else ''}"

                if cid is None:
                    continue

                if cid not in reduced:
                    reduced[cid] = (md, float(sc))

            return list(reduced.values())

        traditional_unique = build_list(traditional_scores)
        multimodal_unique = build_list(multimodal_scores)

        trad_vals = np.array([sc for _, sc in traditional_unique], dtype=float)
        multi_vals = np.array([sc for _, sc in multimodal_unique], dtype=float)

        def normalize(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            if norm == "l1":
                return arr / (np.sum(arr) + 1e-12)
            if norm == "l2":
                return arr / (np.linalg.norm(arr) + 1e-12)
            raise ValueError(f"Unknown norm type: {norm}")

        trad_norm = normalize(trad_vals)
        multi_norm = normalize(multi_vals)

        traditional_sorted = sorted(
            [(md.copy(), trad_norm[i]) for i, (md, _) in enumerate(traditional_unique)],
            key=lambda x: x[1],
            reverse=True,
        )
        multimodal_sorted = sorted(
            [(md.copy(), multi_norm[i]) for i, (md, _) in enumerate(multimodal_unique)],
            key=lambda x: x[1],
            reverse=True,
        )

        fused: list[tuple[dict, float]] = []
        seen: set[str] = set()
        i = 0
        while i < max(len(traditional_sorted), len(multimodal_sorted)):
            # Compare rank i scores if both present, else append whichever exists
            pair: list[tuple[dict, float]] = []
            if i < len(traditional_sorted):
                pair.append(traditional_sorted[i])
            if i < len(multimodal_sorted):
                pair.append(multimodal_sorted[i])
            # Order by score desc for this rank pair so highest inserted first
            pair.sort(key=lambda x: x[1], reverse=True)
            for md, sc in pair:
                corpus_id = md.get("corpus-id")
                doc_id = md.get("doc-id")
                cid = f"{corpus_id}{f'_{doc_id}' if doc_id else ''}"

                if cid in seen:
                    continue
                seen.add(cid)
                md = md.copy()
                md["fusion_method"] = "rerank_fuse"
                fused.append((md, float(sc)))
            i += 1

        return fused

    async def answer(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[str, list[tuple[dict, float]]]:
        """Generate answer based on the retrieved images or documents."""
        if top_k is None:
            top_k = self.top_k

        results = await self.retrieve(query, top_k=top_k)
        # Always attempt to answer. If retrieval returns nothing, proceed with empty context.
        retrieval_results: list[tuple[dict, float]] = results[0] if results else []
        context = self._build_context(retrieval_results)

        if self.generation_disabled or self.generation_model is None:
            if retrieval_results:
                msg = "Generation disabled: only retrieval results returned. Enable generation to produce answers."
            else:
                msg = "Generation disabled and no retrieval results; enable generation or provide retrievable context."
            return msg, retrieval_results

        # Generate answer even if context is empty (pure LLM answer fallback)
        response = await self.generation_model.generate(query, context=context)
        return response, retrieval_results

    def _build_context(self, retrieval_results: list[tuple[dict, float]]):
        """Construct generation context from retrieval results (text + images)."""
        context: list[Any] = []
        for metadata, _ in retrieval_results:
            doc_name = metadata.get("name", "")
            if not doc_name:
                logger.warning(
                    "Document metadata missing 'corpus-id' and 'name'. Skipping context generation for this document.",
                )
                continue

            # Determine item type
            suffix = doc_name.lower()
            if suffix.endswith((".txt", ".md")):
                item_type = "text"
            elif suffix.endswith((".jpg", ".jpeg", ".png")):
                item_type = "image"
            else:
                item_type = "other"

            if item_type == "text":
                text_path = self.traditional_rag.corpuses_dir / doc_name
                try:
                    content = text_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    logger.debug("Failed adding context %s: %s", text_path, exc)
                    continue
                snippet = content[:15000]
                if snippet:
                    context.append({"type": "text", "text": snippet})
            elif item_type == "image":
                context.append(
                    {
                        "type": "image",
                        "image": str(self.multimodal_rag.corpuses_dir / doc_name),
                    },
                )
        return context
