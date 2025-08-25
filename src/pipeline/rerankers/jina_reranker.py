"""
Jina-based reranker implementation for RAG pipelines.

Only reranking logic lives here. The embedding model is implemented in
`pipeline.models.embedding_models.JinaV4Model` so it can be shared
across all RAG systems.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from pipeline.models.embedding_models import JinaV4Model
from utils.device import DeviceConfig

logger = logging.getLogger(__name__)


class JinaReranker:
    """
    Jina-based reranker for RAG pipelines.

    This reranker pre-computes embeddings for the entire corpus using Jina embeddings,
    then reranks retrieved candidates by computing similarity between query embeddings
    and the pre-stored corpus embeddings.
    """

    def __init__(
        self,
        corpus_dir: Path,
        device_config: DeviceConfig | None = None,
        embedding_model: JinaV4Model | None = None,
        embeddings_cache_file: str = "jina_corpus_embeddings.pt",
        embeddings_ids_cache_file: str = "jina_corpus_ids.json",
        cache_dir: Path | None = None,
    ):
        """
        Initialize the Jina reranker.

        Args:
            corpus_dir: Directory containing the corpus documents
            device_config: Device configuration
            embedding_model: Pre-initialized Jina embedding model (optional)
            embeddings_cache_file: Filename for cached corpus embeddings
            embeddings_ids_cache_file: Filename for cached corpus IDs

        """
        self.corpus_dir = Path(corpus_dir)
        self.device_config = device_config

        # Initialize embedding model
        if embedding_model is None:
            self.embedding_model = JinaV4Model(device_config=device_config)
        else:
            self.embedding_model = embedding_model

        # Cache directory (default: <dataset_root>/reranker/jina)
        if cache_dir is None:
            cache_dir = self.corpus_dir.parent / "reranker" / "jina"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self.embeddings_cache_path = cache_dir / embeddings_cache_file
        self.embeddings_ids_cache_path = cache_dir / embeddings_ids_cache_file

        # Cached embeddings and metadata
        self._corpus_embeddings: list[torch.Tensor] | None = None
        self._corpus_ids: list[str] | None = None
        self._corpus_metadata: dict[str, dict[str, Any]] | None = None

        logger.info(f"Initialized JinaReranker for corpus at {self.corpus_dir}")

    async def precompute_corpus_embeddings(
        self,
        corpus_metadata: dict[str, dict[str, Any]],
        force_recompute: bool = False,
        batch_size: int = 4,
    ) -> None:
        """
        Pre-compute embeddings for the entire corpus.

        Args:
            corpus_metadata: Dictionary mapping corpus IDs to metadata
            force_recompute: Whether to recompute embeddings even if cached
            batch_size: Batch size for embedding computation

        """
        # Check if cached embeddings exist and are valid
        if not force_recompute and self._load_cached_embeddings():
            # Verify that cached embeddings match current corpus
            if self._corpus_ids is not None and set(self._corpus_ids) == set(
                corpus_metadata.keys(),
            ):
                logger.info("Using cached corpus embeddings")
                self._corpus_metadata = corpus_metadata
                logger.info("Loaded cached Jina corpus embeddings")
                return
            logger.info("Cached embeddings don't match current corpus, recomputing")

        logger.info("Pre-computing Jina embeddings for corpus...")

        # Collect corpus entries preserving original order for final alignment
        original_order: list[str] = []
        image_entries: list[tuple[str, Path]] = []  # (corpus_id, image_path)
        text_entries: list[tuple[str, str]] = []  # (corpus_id, text_content)

        for corpus_id, metadata in corpus_metadata.items():
            original_order.append(corpus_id)
            try:
                corpus_path = self.corpus_dir / metadata["name"]
                suffix = corpus_path.suffix.lower()
                if suffix == ".txt":
                    # Text file -> read content
                    try:
                        content = corpus_path.read_text(encoding="utf-8")
                    except Exception as e:  # pragma: no cover
                        logger.warning(
                            "Failed reading text for %s (%s); falling back to description.",
                            corpus_id,
                            e,
                        )
                        content = metadata.get(
                            "description",
                            f"Document: {metadata['name']}",
                        )
                    text_entries.append((corpus_id, content))
                elif suffix in {".png", ".jpg", ".jpeg"}:
                    # Image file -> embed image directly
                    if corpus_path.exists():
                        image_entries.append((corpus_id, corpus_path))
                    else:  # fallback to description if path missing
                        desc = metadata.get(
                            "description",
                            f"Image Document: {metadata['name']}",
                        )
                        text_entries.append((corpus_id, desc))
                else:
                    # Other types -> fallback to description
                    desc = metadata.get(
                        "description",
                        f"Document: {metadata['name']}",
                    )
                    text_entries.append((corpus_id, desc))
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to prepare corpus %s: %s", corpus_id, e)
                continue

        if not image_entries and not text_entries:
            logger.warning("No corpus items found for embedding")
            return

        embedding_map: dict[str, torch.Tensor] = {}

        # --- Embed images in batches ---
        if image_entries:
            logger.info(
                "Embedding %d images with Jina (batch_size=%d)",
                len(image_entries),
                batch_size,
            )
            for i in range(0, len(image_entries), batch_size):
                batch = image_entries[i : i + batch_size]
                batch_ids = [cid for cid, _ in batch]
                batch_paths = [p for _, p in batch]
                try:
                    batch_images = [Image.open(p) for p in batch_paths]
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "Failed opening some images in batch starting %d: %s",
                        i,
                        e,
                    )
                    # Fallback: skip this batch
                    continue
                try:
                    img_embs = await self.embedding_model.embed_images(batch_images)
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "Image embedding batch failed (%d images): %s",
                        len(batch_images),
                        e,
                    )
                    continue
                for cid, emb in zip(batch_ids, img_embs, strict=False):
                    embedding_map[cid] = emb
                await asyncio.sleep(0.05)

        # --- Embed texts in batches ---
        if text_entries:
            logger.info(
                "Embedding %d text items with Jina (batch_size=%d)",
                len(text_entries),
                batch_size,
            )
            for i in range(0, len(text_entries), batch_size):
                batch = text_entries[i : i + batch_size]
                batch_ids = [cid for cid, _ in batch]
                batch_texts = [txt for _, txt in batch]
                try:
                    txt_embs = await self.embedding_model.embed_texts(batch_texts)
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "Text embedding batch failed (%d items): %s",
                        len(batch_texts),
                        e,
                    )
                    continue
                for cid, emb in zip(batch_ids, txt_embs, strict=False):
                    embedding_map[cid] = emb
                await asyncio.sleep(0.05)

        # Align embeddings to original order
        ordered_embeddings: list[torch.Tensor] = []
        missing = 0
        for cid in original_order:
            emb = embedding_map.get(cid)
            if emb is None:
                missing += 1
                continue
            ordered_embeddings.append(emb)
        if missing:
            logger.warning("%d corpus items missing embeddings (skipped).", missing)

        self._corpus_embeddings = ordered_embeddings
        self._corpus_ids = [cid for cid in original_order if cid in embedding_map]
        self._corpus_metadata = corpus_metadata

        # Cache the embeddings for future use
        self._save_cached_embeddings()

        logger.info(
            "Pre-computed %d corpus embeddings (images=%d, texts=%d, missing=%d)",
            len(self._corpus_embeddings) if self._corpus_embeddings else 0,
            len(image_entries),
            len(text_entries),
            missing,
        )

    def _load_cached_embeddings(self) -> bool:
        """Load cached embeddings from disk."""
        try:
            if not (
                self.embeddings_cache_path.exists()
                and self.embeddings_ids_cache_path.exists()
            ):
                return False

            # Load embeddings
            self._corpus_embeddings = torch.load(
                self.embeddings_cache_path,
                map_location="cpu",
                weights_only=True,
            )

            # Load IDs
            with self.embeddings_ids_cache_path.open(encoding="utf-8") as f:
                self._corpus_ids = json.load(f)

            if self._corpus_embeddings is not None:
                logger.info(f"Loaded {len(self._corpus_embeddings)} cached embeddings")
            return True

        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}")
            return False

    def _save_cached_embeddings(self) -> None:
        """Save embeddings to cache."""
        try:
            if self._corpus_embeddings and self._corpus_ids:
                # Save embeddings
                torch.save(self._corpus_embeddings, self.embeddings_cache_path)

                # Save IDs
                with open(self.embeddings_ids_cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._corpus_ids, f, indent=2)

                logger.info("Saved embeddings to cache")

        except Exception as e:
            logger.error(f"Failed to save cached embeddings: {e}")

    async def rerank(
        self,
        queries: str | list[str],
        retrieved_candidates: list[list[tuple[dict[str, Any], float]]],
        top_k: int | None = None,
    ) -> list[list[tuple[dict[str, Any], float]]]:
        """
        Rerank retrieved candidates using Jina embeddings.

        Args:
            queries: Single query or list of queries
            retrieved_candidates: List of retrieved candidates for each query
                Format: [[(metadata_dict, score), ...], ...]
            top_k: Number of top candidates to return (None = return all)

        Returns:
            Reranked candidates in the same format as input

        """
        if not self._corpus_embeddings or not self._corpus_ids:
            logger.warning(
                "JinaReranker: embeddings not computed yet – skipping rerank",
            )
            return retrieved_candidates

        # Normalize queries to list
        query_list = [queries] if isinstance(queries, str) else queries

        if len(query_list) != len(retrieved_candidates):
            msg = f"Number of queries ({len(query_list)}) must match number of candidate lists ({len(retrieved_candidates)})"
            raise ValueError(msg)

        # Embed all queries
        logger.info("Computing query embeddings for reranking")
        if hasattr(self.embedding_model, "embed_queries"):
            query_embeddings = await self.embedding_model.embed_queries(query_list)  # type: ignore[attr-defined]
        else:  # fallback
            query_embeddings = await self.embedding_model.embed_texts(query_list)

        # Rerank each query's candidates
        reranked_results: list[list[tuple[dict[str, Any], float]]] = []
        for i, (query_emb, candidates) in enumerate(
            zip(query_embeddings, retrieved_candidates, strict=True),
        ):
            if not candidates:
                reranked_results.append([])
                continue

            # Extract candidate IDs and find their embeddings
            candidate_scores = []
            for metadata, _original_score in candidates:
                corpus_id = metadata.get("corpus-id", "")

                # Find this document's embedding
                if corpus_id in self._corpus_ids:
                    idx = self._corpus_ids.index(corpus_id)
                    corpus_emb = self._corpus_embeddings[idx]
                    q = query_emb / (query_emb.norm(p=2) + 1e-9)
                    d = corpus_emb / (corpus_emb.norm(p=2) + 1e-9)
                    similarity = float(torch.dot(q, d).item())
                    candidate_scores.append((metadata, similarity))
                else:
                    logger.debug(
                        "JinaReranker: corpus id %s missing in embedding cache",
                        corpus_id,
                    )
                    candidate_scores.append((metadata, _original_score))

            # Sort by similarity score (descending)
            candidate_scores.sort(key=lambda x: x[1], reverse=True)

            # Apply top_k limit if specified
            if top_k is not None:
                candidate_scores = candidate_scores[:top_k]

            reranked_results.append(candidate_scores)

            logger.debug(f"Reranked {len(candidates)} candidates for query {i + 1}")

        logger.info(f"Completed reranking for {len(query_list)} queries")
        return reranked_results

    def get_embedding_stats(self) -> dict[str, Any]:
        """Get statistics about the cached embeddings."""
        if not self._corpus_embeddings:
            return {"status": "no_embeddings"}

        embedding_dim = (
            self._corpus_embeddings[0].shape[0] if self._corpus_embeddings else 0
        )

        return {
            "status": "ready",
            "num_embeddings": len(self._corpus_embeddings),
            "embedding_dimension": embedding_dim,
            "cache_path": str(self.embeddings_cache_path),
            "device": getattr(self.embedding_model, "device_config", None).device_str
            if getattr(self.embedding_model, "device_config", None)
            else "unknown",
        }


class JinaRerankerFactory:
    """Factory for creating Jina rerankers with different configurations."""

    @staticmethod
    def create_reranker(
        rag_type: str,
        corpus_dir: Path,
        device_config: DeviceConfig | None = None,
        **kwargs,
    ) -> JinaReranker:
        """
        Create a Jina reranker configured for a specific RAG type.

        Args:
            rag_type: Type of RAG system ("traditional", "multimodal", "graph", etc.)
            corpus_dir: Directory containing the corpus
            device_config: Device configuration
            **kwargs: Additional arguments for reranker initialization

        Returns:
            Configured JinaReranker instance

        """
        # Customize cache file names based on RAG type
        embeddings_cache_file = f"jina_{rag_type}_embeddings.pt"
        embeddings_ids_cache_file = f"jina_{rag_type}_ids.json"

        return JinaReranker(
            corpus_dir=corpus_dir,
            device_config=device_config,
            embeddings_cache_file=embeddings_cache_file,
            embeddings_ids_cache_file=embeddings_ids_cache_file,
            **kwargs,
        )
