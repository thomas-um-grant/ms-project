from __future__ import annotations

import asyncio
import json
import logging
import re
from enum import Enum
from pathlib import Path

import nltk
import numpy as np
import pytesseract
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from pipeline.rags.base_rag import BaseRAG
from pipeline.rags.helpers import MetadataManager
from utils.device import cleanup_memory

logger = logging.getLogger(__name__)


class RetrievalMethod(Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class TraditionalRAG(BaseRAG):
    """Traditional text-only RAG implementation with vector, BM25, and hybrid retrieval."""

    def __init__(
        self,
        name: str,
        data_dir: Path,
        configs: dict | None = None,
        *,
        disable_generation: bool = False,
    ) -> None:
        configs = configs or {}
        super().__init__(name, data_dir, configs, disable_generation=disable_generation)

        # Load defaults
        defaults = self._load_defaults()

        # Configuration
        self.extraction_batch_size = configs.get(
            "extraction_batch_size",
            defaults["processing"]["extraction"]["batch_size"],
        )
        self.retrieval_method = RetrievalMethod(
            configs.get("retrieval_method", "vector"),
        )

        # BM25 setup
        self.bm25_data_path = self.store_dir / "bm25_data.json"
        self._bm25_index: BM25Okapi | None = None
        self._bm25_corpus_ids: list[str] | None = None
        self.stop_words: set[str] = set()

        # Initialize BM25 resources if needed
        if self.retrieval_method in [RetrievalMethod.BM25, RetrievalMethod.HYBRID]:
            self._setup_bm25_resources()

        # Metadata manager
        self.metadata_manager = MetadataManager(
            self.metadata_path,
            json_indent=configs.get(
                "json_indent",
                defaults["data_structure"]["metadata"]["json_indent"],
            ),
            fallback_corpus_id=configs.get(
                "fallback_corpus_id",
                defaults["data_structure"]["metadata"]["fallback_corpus_id"],
            ),
        )

        # Log embedding file selection (BaseRAG already decided paths)
        logger.info(
            "TraditionalRAG embedding storage -> %s / %s (legacy=%s)",
            self.embeddings_path.name,
            self.embeddings_ids_path.name,
            getattr(self, "using_legacy_embedding_files", False),
        )

    def _setup_bm25_resources(self) -> None:
        """Download and setup NLTK resources for BM25."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            self.stop_words = set()

    def _tokenize_text(self, text: str) -> list[str]:
        """Tokenize text for BM25 with fallback tokenization."""
        try:
            tokens = word_tokenize(text.lower())
        except LookupError:
            tokens = re.findall(r"[a-zA-Z]+", text.lower())

        tokens = [t for t in tokens if t.isalpha()]
        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]
        return tokens

    async def extract(
        self,
        documents: list[Path],
        *,
        preprocessed: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """Extract text from documents and save as corpus files."""
        if batch_size is None:
            batch_size = self.extraction_batch_size
        if not documents:
            print("No documents provided for extraction.")
            return

        corpus_entries: list[tuple[str, Path, str]] = []

        for doc_path in documents:
            if not doc_path.exists():
                continue

            suffix = doc_path.suffix.lower()
            stem = doc_path.stem

            if suffix == ".pdf":
                corpus_entries.extend(self._extract_pdf(doc_path, stem))
            elif suffix in {".png", ".jpg", ".jpeg"}:
                entry = self._extract_image(doc_path, stem)
                if entry:
                    corpus_entries.append(entry)
            elif suffix in {".txt", ".md"}:
                entry = self._extract_text(doc_path, stem, preprocessed)
                if entry:
                    corpus_entries.append(entry)

        if not corpus_entries:
            print("No new documents to process.")
            return

        await self._save_corpuses(corpus_entries, batch_size)

    def _extract_pdf(self, path: Path, stem: str) -> list[tuple[str, Path, str]]:
        """Extract text from PDF pages."""
        entries: list[tuple[str, Path, str]] = []
        try:
            reader = PdfReader(str(path))
        except (PdfReadError, FileNotFoundError, OSError, ValueError) as exc:
            logger.warning("Failed to read PDF '%s': %s", path, exc)
            return entries

        for page_idx, page in enumerate(reader.pages, start=1):
            corpus_id = f"{stem}_{page_idx}"
            if self.metadata_manager.is_document_processed(corpus_id):
                continue
            try:
                text = page.extract_text() or ""
            except (ValueError, TypeError, AttributeError) as exc:
                logger.debug(
                    "Text extraction failed for %s page %d: %s",
                    path.name,
                    page_idx,
                    exc,
                )
                text = ""
            entries.append((corpus_id, self.corpuses_dir / f"{corpus_id}.txt", text))
        return entries

    def _extract_image(self, path: Path, stem: str) -> tuple[str, Path, str] | None:
        """Extract text from image using OCR."""
        if self.metadata_manager.is_document_processed(stem):
            return None
        try:
            text = pytesseract.image_to_string(Image.open(path))
        except (OSError, ValueError):
            text = ""
        return (stem, self.corpuses_dir / f"{stem}.txt", text)

    def _extract_text(
        self,
        path: Path,
        stem: str,
        preprocessed: bool,
    ) -> tuple[str, Path, str] | None:
        """Extract text from text files."""
        corpus_id = stem if preprocessed else f"{stem}_1"
        if self.metadata_manager.is_document_processed(corpus_id):
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            text = ""
        return (corpus_id, self.corpuses_dir / f"{corpus_id}.txt", text)

    async def _save_corpuses(
        self,
        corpus_entries: list[tuple[str, Path, str]],
        batch_size: int,
    ) -> None:
        """Save corpus entries to disk and update metadata."""
        processed = 0
        total = len(corpus_entries)

        for i in range(0, total, batch_size):
            batch = corpus_entries[i : i + batch_size]
            batch_num = i // batch_size + 1

            metadata_batch: list[tuple[str, str, str, bool]] = []
            for corpus_id, text_path, content in batch:
                try:
                    text_path.write_text(content, encoding="utf-8")
                    description = content[:5000]  # Truncate for metadata
                    metadata_batch.append(
                        (corpus_id, text_path.name, description, False),
                    )
                    processed += 1
                except OSError:
                    continue

            if metadata_batch:
                self.metadata_manager.update_metadata_batch(metadata_batch)

            print(
                f"✓ Batch {batch_num}: {len(metadata_batch)}/{len(batch)} documents saved",
            )
            cleanup_memory()
            await asyncio.sleep(0.01)

        self.metadata_manager.flush()
        print(f"Extraction complete: {processed} documents processed")

    async def index(self) -> None:
        """Index documents for vector and/or BM25 retrieval."""
        if self.retrieval_method in [RetrievalMethod.VECTOR, RetrievalMethod.HYBRID]:
            await self._index_vectors()
        if self.retrieval_method in [RetrievalMethod.BM25, RetrievalMethod.HYBRID]:
            await self._index_bm25()

    async def _index_vectors(self) -> None:
        """Create vector embeddings for all documents."""
        print("Building vector index...")
        # Get unembedded documents for this embedding model
        unembedded = self.metadata_manager.get_unembedded_documents(
            self.embedding_model_tag,
        )
        # (metadata still needed later for reading corpus files)
        metadata = self.metadata_manager.load_metadata()

        if not unembedded:
            print("No documents to index.")
            return

        # Load existing embeddings
        embeddings, embedding_ids = self._load_embeddings()

        # Process new documents
        corpus_ids, texts = [], []
        for doc_id, meta in unembedded:
            text_path = self.corpuses_dir / meta["name"]
            try:
                content = text_path.read_text(encoding="utf-8").strip()
                if content:
                    corpus_ids.append(doc_id)
                    texts.append(content)
            except (OSError, UnicodeDecodeError):
                continue

        if not texts:
            print("No new texts to index.")
            return

        # Create embeddings in batches
        new_embeddings, new_ids = [], []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i : i + self.batch_size]
            batch_ids = corpus_ids[i : i + self.batch_size]
            try:
                batch_embs = await self.embedding_model.embed_texts(batch_texts)
                for doc_id, emb in zip(batch_ids, batch_embs, strict=False):
                    if isinstance(emb, torch.Tensor) and emb.numel() > 0:
                        new_embeddings.append(emb)
                        new_ids.append(doc_id)
            except (RuntimeError, ValueError) as exc:
                logger.warning("Embedding batch failed: %s", exc)

        # Save embeddings
        if new_embeddings:
            all_embeddings = embeddings + new_embeddings
            all_ids = embedding_ids + new_ids

            torch.save(all_embeddings, self.embeddings_path)
            with self.embeddings_ids_path.open("w") as f:
                for doc_id in all_ids:
                    f.write(json.dumps(doc_id) + "\n")

            self.metadata_manager.mark_as_embedded(new_ids, self.embedding_model_tag)
            print(f"Indexed {len(new_embeddings)} documents (total: {len(all_ids)})")

    def _load_embeddings(self) -> tuple[list[torch.Tensor], list[str]]:
        """Load existing embeddings from disk."""
        if not (self.embeddings_path.exists() and self.embeddings_ids_path.exists()):
            return [], []
        try:
            embeddings = torch.load(
                self.embeddings_path,
                map_location="cpu",
                weights_only=True,
            )
            with self.embeddings_ids_path.open() as f:
                ids = [json.loads(line) for line in f]
            return embeddings, ids
        except (OSError, json.JSONDecodeError, RuntimeError, ValueError):
            return [], []

    async def _index_bm25(self) -> None:
        """Create BM25 index from text documents."""
        print("Building BM25 index...")

        metadata = self.metadata_manager.load_metadata()
        if not metadata:
            print("No documents to index for BM25.")
            return

        documents, corpus_ids = [], []
        for doc_id, meta in metadata.items():
            text_path = self.corpuses_dir / meta["name"]
            try:
                text = text_path.read_text(encoding="utf-8")
                tokens = self._tokenize_text(text)
                if tokens:
                    documents.append(tokens)
                    corpus_ids.append(doc_id)
            except (OSError, UnicodeDecodeError):
                continue

        if not documents:
            print("No valid documents for BM25 indexing.")
            return

        # Build and save BM25 index
        self._bm25_index = BM25Okapi(documents)
        self._bm25_corpus_ids = corpus_ids

        # Save BM25 data
        bm25_data = {"documents": documents, "corpus_ids": corpus_ids}
        try:
            with self.bm25_data_path.open("w") as f:
                json.dump(bm25_data, f)
            print(f"BM25 index built with {len(documents)} documents")
        except OSError as exc:
            logger.warning("Failed saving BM25 data: %s", exc)

    def _load_bm25_index(self) -> bool:
        """Load BM25 index from disk."""
        if self._bm25_index is not None:
            return True

        if not self.bm25_data_path.exists():
            return False

        try:
            with self.bm25_data_path.open() as f:
                data = json.load(f)
            self._bm25_index = BM25Okapi(data["documents"])
            self._bm25_corpus_ids = data["corpus_ids"]
            return True
        except (OSError, json.JSONDecodeError, ValueError):
            return False

    async def retrieve(
        self,
        queries: str | list[str],
        top_k: int | None = None,
        method: RetrievalMethod | None = None,
    ) -> list[list[tuple[dict, float]]]:
        """Retrieve documents using the specified method."""
        if top_k is None:
            top_k = self.top_k
        if method is None:
            method = self.retrieval_method

        query_list = [queries] if isinstance(queries, str) else queries
        if not query_list:
            return []

        # Perform raw retrieval based on selected method
        if method == RetrievalMethod.VECTOR:
            raw_results = await self._retrieve_vectors(query_list, top_k)
        elif method == RetrievalMethod.BM25:
            raw_results = await self._retrieve_bm25(query_list, top_k)
        elif method == RetrievalMethod.HYBRID:
            raw_results = await self._retrieve_hybrid(query_list, top_k)
        else:
            msg = f"Unknown retrieval method: {method}"
            raise ValueError(msg)

        # If rerank is enabled, apply it to the results
        if self.auto_rerank and self.reranker_method:
            try:
                return await self.rerank(
                    queries=query_list,
                    retrieved_corpuses=raw_results,
                    method=self.reranker_method,
                )
            except Exception:  # pragma: no cover - fail open
                logger.exception(
                    "Automatic rerank failed; returning raw retrieval results",
                )

        return raw_results

    async def _retrieve_vectors(
        self,
        queries: list[str],
        top_k: int,
    ) -> list[list[tuple[dict, float]]]:
        """Retrieve documents using vector similarity."""
        embeddings, embedding_ids = self._load_embeddings()
        metadata = self.metadata_manager.load_metadata()

        if not embeddings:
            return [[] for _ in queries]

        # Filter valid embeddings
        valid_embeddings = [
            (emb, doc_id)
            for emb, doc_id in zip(embeddings, embedding_ids, strict=True)
            if isinstance(emb, torch.Tensor) and emb.numel() > 0
        ]

        if not valid_embeddings:
            return [[] for _ in queries]

        doc_embeddings, doc_ids = zip(*valid_embeddings, strict=True)

        # Get query embeddings
        query_embeddings = await self.embedding_model.embed_texts(queries)

        # Move to device
        doc_embeddings = [
            emb.to(self.device_config.device_str) for emb in doc_embeddings
        ]
        query_embeddings = [
            emb.to(self.device_config.device_str) for emb in query_embeddings
        ]

        # Compute similarities
        results = []
        for query_emb in query_embeddings:
            similarities = []
            for i, doc_emb in enumerate(doc_embeddings):
                similarity = torch.cosine_similarity(
                    query_emb.unsqueeze(0),
                    doc_emb.unsqueeze(0),
                ).item()
                similarities.append((similarity, i))

            # Get top-k results
            similarities.sort(reverse=True)
            top_results = []
            for sim_score, idx in similarities[:top_k]:
                doc_id = doc_ids[idx]
                if doc_id in metadata:
                    top_results.append((metadata[doc_id], sim_score))

            results.append(top_results)

        return results

    async def _retrieve_bm25(
        self,
        queries: list[str],
        top_k: int,
    ) -> list[list[tuple[dict, float]]]:
        """Retrieve documents using BM25."""
        if not self._load_bm25_index():
            # Try to build index on demand
            await self._index_bm25()
            if not self._load_bm25_index():
                logger.error("BM25 index not available")
                return [[] for _ in queries]

        metadata = self.metadata_manager.load_metadata()
        results = []

        for query in queries:
            tokens = self._tokenize_text(query)
            if not tokens:
                results.append([])
                continue

            scores = self._bm25_index.get_scores(tokens)
            top_indices = np.argsort(scores)[-top_k:][::-1]

            query_results = []
            for idx in top_indices:
                if idx < len(self._bm25_corpus_ids):
                    doc_id = self._bm25_corpus_ids[idx]
                    if doc_id in metadata:
                        query_results.append((metadata[doc_id], float(scores[idx])))

            results.append(query_results)

        return results

    async def _retrieve_hybrid(
        self,
        queries: list[str],
        top_k: int,
    ) -> list[list[tuple[dict, float]]]:
        """Hybrid retrieval using Reciprocal Rank Fusion (RRF)."""
        # Get results from both methods with larger k for better fusion
        retrieve_k = max(top_k * 2, 50)  # Retrieve more for better fusion

        vector_results = await self._retrieve_vectors(queries, retrieve_k)
        bm25_results = await self._retrieve_bm25(queries, retrieve_k)

        metadata = self.metadata_manager.load_metadata()
        combined_results = []

        for vec_res, bm25_res in zip(vector_results, bm25_results, strict=True):
            # Apply Reciprocal Rank Fusion
            doc_scores = {}

            # Add vector scores (RRF with k=60 is standard)
            for rank, (meta, _score) in enumerate(vec_res):
                # Use the document name without extension as the key
                doc_name = meta.get("name", "")
                if doc_name.endswith(".txt"):
                    doc_key = doc_name[:-4]  # Remove .txt extension
                    doc_scores[doc_key] = doc_scores.get(doc_key, 0) + 1 / (rank + 60)

            # Add BM25 scores
            for rank, (meta, _score) in enumerate(bm25_res):
                # Use the document name without extension as the key
                doc_name = meta.get("name", "")
                if doc_name.endswith(".txt"):
                    doc_key = doc_name[:-4]  # Remove .txt extension
                    doc_scores[doc_key] = doc_scores.get(doc_key, 0) + 1 / (rank + 60)

            # Sort by combined RRF score and take top-k
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[
                :top_k
            ]

            # Convert back to expected format
            query_results = []
            for doc_key, rrf_score in sorted_docs:
                if doc_key in metadata:
                    query_results.append((metadata[doc_key], rrf_score))

            combined_results.append(query_results)

        return combined_results

    async def answer(
        self,
        query: str,
        top_k: int | None = None,
        method: RetrievalMethod | None = None,
    ) -> tuple[str, list[tuple[dict, float]]]:
        """Generate an answer using retrieved context."""
        if top_k is None:
            top_k = self.top_k

        retrieved = await self.retrieve(query, top_k=top_k, method=method)
        context_chunks = []

        if retrieved and retrieved[0]:
            for meta, _score in retrieved[0]:
                text_path = self.corpuses_dir / meta["name"]
                try:
                    context_chunks.append(text_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError) as exc:
                    logger.debug("Failed adding context %s: %s", text_path, exc)

        # Join context and truncate if too long
        joined_context = "\n\n".join(context_chunks)[:15000]

        if self.generation_disabled or self.generation_model is None:
            msg = "Generation disabled: only retrieval results returned. Enable generation to produce answers."
            return msg, (retrieved[0] if retrieved else [])

        response = await self.generation_model.generate(
            query,
            context=[{"type": "text", "text": joined_context}],
        )
        return response, (retrieved[0] if retrieved else [])
