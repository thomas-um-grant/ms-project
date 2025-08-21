"""
Jina-based reranker implementation for RAG pipelines.

This module provides a specialized reranker that uses Jina embeddings (jinaai/jina-embeddings-v4)
to rerank retrieved candidates based on semantic similarity between query and corpus embeddings.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from utils.device import DeviceConfig

logger = logging.getLogger(__name__)


class JinaEmbeddingModel:
    """
    Jina embedding model wrapper for jinaai/jina-embeddings-v4.

    This model provides high-quality embeddings specifically optimized for
    retrieval and reranking tasks.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device_config: DeviceConfig | None = None,
        trust_remote_code: bool = True,
    ):
        """
        Initialize the Jina embedding model.

        Args:
            model_name: HuggingFace model identifier for Jina embeddings
            device_config: Device configuration for model placement
            trust_remote_code: Whether to trust remote code (required for Jina models)

        """
        self.model_name = model_name
        self.device_config = device_config
        self.trust_remote_code = trust_remote_code

        self.tokenizer = None
        self.model = None
        self._device = None

        if device_config:
            self._device = device_config.device.value
        # Auto-detect best device
        elif torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        logger.info(f"Initializing Jina model {model_name} on {self._device}")

    def _lazy_load(self):
        """Lazy load the model and tokenizer to avoid initialization costs."""
        if self.model is None or self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=self.device_config.dtype
                    if self.device_config
                    else torch.float32,
                ).to(self._device)
                self.model.eval()
                logger.info(f"Successfully loaded Jina model on {self._device}")
            except Exception as e:
                logger.error(f"Failed to load Jina model: {e}")
                raise

    async def embed_texts(self, texts: list[str]) -> list[torch.Tensor]:
        """
        Embed a list of texts asynchronously.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding tensors, one per input text

        """
        if not texts:
            return []

        self._lazy_load()

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, self._embed_texts_sync, texts)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            raise

    def _embed_texts_sync(self, texts: list[str]) -> list[torch.Tensor]:
        """Synchronous text embedding implementation."""
        try:
            # Tokenize texts
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,  # Jina models typically work well with 512 tokens
            ).to(self._device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling over token embeddings
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                )
                # Normalize embeddings for cosine similarity
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Split into list of individual tensors
            return [embeddings[i] for i in range(embeddings.size(0))]

        except Exception as e:
            logger.error(f"Error in synchronous embedding: {e}")
            raise

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply mean pooling to token embeddings."""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1),
            min=1e-9,
        )


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
        embedding_model: JinaEmbeddingModel | None = None,
        embeddings_cache_file: str = "jina_corpus_embeddings.pt",
        embeddings_ids_cache_file: str = "jina_corpus_ids.json",
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
            self.embedding_model = JinaEmbeddingModel(device_config=device_config)
        else:
            self.embedding_model = embedding_model

        # Cache file paths
        self.embeddings_cache_path = self.corpus_dir.parent / embeddings_cache_file
        self.embeddings_ids_cache_path = (
            self.corpus_dir.parent / embeddings_ids_cache_file
        )

        # Cached embeddings and metadata
        self._corpus_embeddings: list[torch.Tensor] | None = None
        self._corpus_ids: list[str] | None = None
        self._corpus_metadata: dict[str, dict[str, Any]] | None = None

        logger.info(f"Initialized JinaReranker for corpus at {self.corpus_dir}")

    async def precompute_corpus_embeddings(
        self,
        corpus_metadata: dict[str, dict[str, Any]],
        force_recompute: bool = False,
        batch_size: int = 32,
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
                return
            logger.info("Cached embeddings don't match current corpus, recomputing")

        logger.info("Pre-computing Jina embeddings for corpus...")

        # Extract text content from all corpus documents
        corpus_texts = []
        corpus_ids = []

        for corpus_id, metadata in corpus_metadata.items():
            try:
                corpus_path = self.corpus_dir / metadata["name"]

                # Handle different file types
                if corpus_path.suffix.lower() == ".txt":
                    content = corpus_path.read_text(encoding="utf-8")
                else:
                    # For other file types, use the description from metadata if available
                    content = metadata.get(
                        "description",
                        f"Document: {metadata['name']}",
                    )

                corpus_texts.append(content)
                corpus_ids.append(corpus_id)

            except Exception as e:
                logger.warning(f"Failed to load corpus {corpus_id}: {e}")
                continue

        if not corpus_texts:
            logger.warning("No corpus texts found for embedding")
            return

        # Compute embeddings in batches
        all_embeddings = []
        for i in range(0, len(corpus_texts), batch_size):
            batch_texts = corpus_texts[i : i + batch_size]
            logger.info(
                f"Computing embeddings for batch {i // batch_size + 1}/{(len(corpus_texts) + batch_size - 1) // batch_size}",
            )

            batch_embeddings = await self.embedding_model.embed_texts(batch_texts)
            all_embeddings.extend(batch_embeddings)

            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)

        # Store embeddings and metadata
        self._corpus_embeddings = all_embeddings
        self._corpus_ids = corpus_ids
        self._corpus_metadata = corpus_metadata

        # Cache the embeddings for future use
        self._save_cached_embeddings()

        logger.info(f"Pre-computed {len(all_embeddings)} corpus embeddings")

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
                "Corpus embeddings not available, returning original ranking",
            )
            return retrieved_candidates

        # Normalize queries to list
        query_list = [queries] if isinstance(queries, str) else queries

        if len(query_list) != len(retrieved_candidates):
            msg = f"Number of queries ({len(query_list)}) must match number of candidate lists ({len(retrieved_candidates)})"
            raise ValueError(msg)

        # Embed all queries
        logger.info("Computing query embeddings for reranking")
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
                try:
                    idx = self._corpus_ids.index(corpus_id)
                    corpus_emb = self._corpus_embeddings[idx]

                    # Compute cosine similarity
                    if self.device_config:
                        query_device = query_emb.to(self.device_config.device.value)
                        corpus_device = corpus_emb.to(self.device_config.device.value)
                    else:
                        query_device = query_emb
                        corpus_device = corpus_emb

                    similarity = torch.cosine_similarity(
                        query_device.unsqueeze(0),
                        corpus_device.unsqueeze(0),
                    ).item()

                    candidate_scores.append((metadata, similarity))

                except ValueError:
                    logger.warning(
                        f"Corpus ID {corpus_id} not found in pre-computed embeddings",
                    )
                    # Use original score as fallback
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
            "device": self.embedding_model._device
            if self.embedding_model
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
