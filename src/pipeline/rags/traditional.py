from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytesseract
import torch
from PIL import Image
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from tqdm import tqdm

from pipeline.rags.base_rag import BaseRAG
from pipeline.rags.helpers import EmbeddingIndexer, MetadataManager
from utils.device import cleanup_memory, log_memory_usage

logger = logging.getLogger(__name__)


class TraditionalRAG(BaseRAG):
    """
    Traditional text-only RAG implementation.

    Extraction: each PDF/text file is split into per-page corpuses.
    Each corpus stored as txt file named {document_stem}_{page_number}.txt
    Metadata entry has description = page text (truncated if very long).
    Indexing: embed page texts in batches using existing EmbeddingIndexer mechanics (adapted for text).
    Retrieval: cosine / model-native scoring similar to MultiModal.
    Answer: retrieve contexts then pass as concatenated text chunks to generation model.
    """

    def __init__(
        self,
        name: str,
        data_dir: Path,
        configs: dict | None = None,
    ) -> None:
        configs = configs or {}
        super().__init__(name, data_dir, configs)

        # Load defaults
        defaults_path = (
            Path(__file__).parent.parent.parent / "configs" / "defaults.json"
        )
        with defaults_path.open() as f:
            defaults = json.load(f)

        self.extraction_batch_size = configs.get(
            "extraction_batch_size",
            defaults["processing"]["extraction"]["batch_size"],
        )
        self.processing_summary_template = configs.get(
            "processing_summary_template",
            defaults["logging"]["processing_summary_template"],
        )

        # Text-specific paths
        self.text_corpuses_dir = self.corpuses_dir  # reuse directory

        # Metadata & embedding helpers
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
        self.embedding_indexer = EmbeddingIndexer(
            self.embedding_model,
            self.device_config,
            self.metadata_path,
            self.embeddings_ids_path,
            self.embeddings_path,
            batch_size=self.batch_size,
            expected_batch_embedding_dims=configs.get(
                "expected_batch_embedding_dims",
                defaults["processing"]["indexing"]["expected_batch_embedding_dims"],
            ),
        )

    async def extract(
        self,
        documents: list[Path],
        *,
        preprocessed: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """
        Extract text corpuses.

        If preprocessed=True, all provided documents must already be textual (.txt/.md) corpuses
        and will be ingested directly (no OCR or PDF parsing).
        If preprocessed=False, raw PDFs are split per page, images OCR'd, and text/markdown
        files ingested (single page heuristic) into per-page corpuses.
        """
        if batch_size is None:
            batch_size = self.extraction_batch_size
        if not documents:
            print("No documents provided for extraction.")
            return

        corpus_entries = (
            self._collect_preprocessed_texts(documents)
            if preprocessed
            else self._collect_and_process_raw_documents(documents)
        )

        if not corpus_entries:
            print("No new corpuses to process.")
            return

        await self._write_corpuses_and_metadata(corpus_entries, batch_size)

    # ---------------- Collection Helpers ----------------
    def _collect_preprocessed_texts(
        self,
        documents: list[Path],
    ) -> list[tuple[str, Path, str]]:
        allowed = {".txt", ".md"}
        entries: list[tuple[str, Path, str]] = []
        for path in documents:
            if not path.exists() or path.suffix.lower() not in allowed:
                continue
            corpus_id = path.stem
            if self.metadata_manager.is_document_processed(corpus_id):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                content = ""
            entries.append(
                (corpus_id, self.text_corpuses_dir / f"{corpus_id}.txt", content),
            )
        return entries

    def _collect_and_process_raw_documents(
        self,
        documents: list[Path],
    ) -> list[tuple[str, Path, str]]:
        entries: list[tuple[str, Path, str]] = []
        for path in documents:
            if not path.exists():
                continue
            suffix = path.suffix.lower()
            stem = path.stem
            if suffix == ".pdf":
                entries.extend(self._pdf_entries(path, stem))
            elif suffix in {".png", ".jpg", ".jpeg"}:
                image_entry = self._image_entry(path, stem)
                if image_entry:
                    entries.append(image_entry)
            elif suffix in {".txt", ".md"}:
                text_entry = self._text_entry(path, stem)
                if text_entry:
                    entries.append(text_entry)
        return entries

    # ---------------- Individual Type Handlers ----------------
    def _pdf_entries(self, path: Path, stem: str) -> list[tuple[str, Path, str]]:
        out: list[tuple[str, Path, str]] = []
        try:
            reader = PdfReader(str(path))
        except (PdfReadError, FileNotFoundError, OSError, ValueError) as exc:
            logger.warning("Failed to read PDF '%s': %s", path, exc)
            return out
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
            out.append((corpus_id, self.text_corpuses_dir / f"{corpus_id}.txt", text))
        return out

    def _image_entry(self, path: Path, stem: str) -> tuple[str, Path, str] | None:
        corpus_id = stem
        if self.metadata_manager.is_document_processed(corpus_id):
            return None
        try:
            text = pytesseract.image_to_string(Image.open(path))
        except (OSError, ValueError):
            text = ""
        return (corpus_id, self.text_corpuses_dir / f"{corpus_id}.txt", text)

    def _text_entry(self, path: Path, stem: str) -> tuple[str, Path, str] | None:
        corpus_id = f"{stem}_1" if "_" not in stem else stem
        if self.metadata_manager.is_document_processed(corpus_id):
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            text = ""
        return (corpus_id, self.text_corpuses_dir / f"{corpus_id}.txt", text)

    # ---------------- Persist & Metadata ----------------
    async def _write_corpuses_and_metadata(
        self,
        corpus_entries: list[tuple[str, Path, str]],
        batch_size: int,
    ) -> None:
        processed = 0
        skipped = 0
        total = len(corpus_entries)
        total_batches = (total + batch_size - 1) // batch_size
        for i in range(0, total, batch_size):
            batch = corpus_entries[i : i + batch_size]
            batch_num = i // batch_size + 1
            log_memory_usage(f"Processing text batch {batch_num}/{total_batches}")
            metadata_batch: list[tuple[str, str, str, bool]] = []
            for corpus_id, text_path, content in batch:
                try:
                    text_path.write_text(content, encoding="utf-8")
                except OSError:
                    skipped += 1
                    continue
                description = content[:5000]
                metadata_batch.append((corpus_id, text_path.name, description, False))
                processed += 1
            if metadata_batch:
                self.metadata_manager.update_metadata_batch(metadata_batch)
            print(
                f"✓ Batch {batch_num}/{total_batches}: {len(metadata_batch)}/{len(batch)} saved",
            )
            cleanup_memory()
            await asyncio.sleep(0.01)
        self.metadata_manager.flush()
        print(
            self.processing_summary_template.format(
                processed=processed,
                skipped=skipped + (total - processed - skipped),
            ),
        )

    async def index(self) -> None:  # type: ignore[override]
        # Load unembedded corpus texts
        unembedded = self.metadata_manager.get_unembedded_documents()
        if not unembedded:
            print("No documents to index.")
            return

        corpus_ids: list[str] = []
        texts: list[str] = []
        for doc_id, meta in unembedded:
            corpus_ids.append(doc_id)
            text_path = self.text_corpuses_dir / meta["name"]
            try:
                texts.append(text_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError) as exc:
                logger.debug("Failed reading %s: %s", text_path, exc)
                texts.append("")

        existing_embeddings: list[torch.Tensor] = []
        existing_ids: list[str] = []
        if self.embeddings_path.exists() and self.embeddings_ids_path.exists():
            existing_embeddings = torch.load(
                self.embeddings_path,
                map_location="cpu",
                weights_only=True,
            )
            with self.embeddings_ids_path.open() as f:
                existing_ids = [json.loads(line) for line in f]

        new_embeddings: list[torch.Tensor] = []
        new_ids: list[str] = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i : i + self.batch_size]
            batch_ids = corpus_ids[i : i + self.batch_size]
            try:
                batch_embs = await self.embedding_model.embed_texts(batch_texts)
            except (RuntimeError, ValueError) as exc:  # pragma: no cover
                logger.warning(
                    "Embedding batch %d failed: %s",
                    i // self.batch_size + 1,
                    exc,
                )
                continue
            for cid, emb in zip(batch_ids, batch_embs, strict=False):
                if not isinstance(emb, torch.Tensor) or emb.numel() == 0:
                    logger.debug("Skipping zero-length embedding for %s", cid)
                    continue
                new_embeddings.append(emb)
                new_ids.append(cid)

        if new_embeddings:
            all_embeddings = existing_embeddings + new_embeddings
            torch.save(all_embeddings, self.embeddings_path)
            all_ids = existing_ids + new_ids
            with self.embeddings_ids_path.open("w") as f:
                for cid in all_ids:
                    f.write(json.dumps(cid) + "\n")
            self.metadata_manager.mark_as_embedded(new_ids)
            print(
                f"Indexed {len(new_embeddings)} texts (total stored: {len(all_ids)}).",
            )
        else:
            print("No new texts indexed.")

    async def retrieve(
        self,
        queries: str | list[str],
        top_k: int | None = None,
    ) -> list[list[tuple[dict, float]]]:  # type: ignore[override]
        if top_k is None:
            top_k = self.top_k
        query_list = [queries] if isinstance(queries, str) else queries
        if not query_list:
            return []
        embeddings, embedding_ids, metadata = self.embedding_indexer.load_index()
        # Filter zero-length embeddings (possible legacy artifacts)
        filtered = [
            (emb, eid)
            for emb, eid in zip(embeddings, embedding_ids, strict=False)
            if isinstance(emb, torch.Tensor) and emb.numel() > 0
        ]
        if len(filtered) < len(embeddings):
            logger.warning(
                "Filtered %d zero-length embeddings from index (using %d).",
                len(embeddings) - len(filtered),
                len(filtered),
            )
        if not filtered:
            logger.error("No valid embeddings available for retrieval.")
            return [[] for _ in query_list]
        doc_vectors, embedding_ids = zip(*filtered, strict=False)
        q_embs = await self.embedding_model.embed_texts(query_list)
        doc_vectors = [
            e.to(
                dtype=self.device_config.dtype,
                device=self.device_config.device_str,
                non_blocking=True,
            )
            for e in doc_vectors
        ]
        q_vectors = [
            q.to(
                dtype=self.device_config.dtype,
                device=self.device_config.device_str,
                non_blocking=True,
            )
            for q in q_embs
        ]
        results: list[list[tuple[dict, float]]] = []
        import numpy as np
        from torch.nn import functional

        has_processor = hasattr(self.embedding_model, "processor") and hasattr(
            self.embedding_model.processor,
            "score",
        )
        if not has_processor:
            doc_matrix = torch.stack(doc_vectors)
            doc_norm = functional.normalize(doc_matrix, dim=1)
        for q_vec in q_vectors:
            if has_processor:
                scores = self.embedding_model.processor.score([q_vec], doc_vectors)
                if isinstance(scores, torch.Tensor):
                    scores = scores.squeeze(0).detach().cpu().numpy()
            else:
                q_norm = functional.normalize(q_vec.unsqueeze(0), dim=1)
                scores = torch.mm(q_norm, doc_norm.T).squeeze(0).detach().cpu().numpy()
            if scores.ndim > 1:
                scores = scores.flatten()
            idxs = np.argsort(scores)[-top_k:][::-1]
            query_results = [
                (metadata[embedding_ids[j]], float(scores[j])) for j in idxs
            ]
            results.append(query_results)
        return results

    async def answer(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[str, list[tuple[dict, float]]]:
        if top_k is None:
            top_k = self.top_k
        retrieved = await self.retrieve(query, top_k=top_k)
        context_chunks: list[str] = []
        for meta, _score in retrieved[0]:
            text_path = self.text_corpuses_dir / meta["name"]
            try:
                context_chunks.append(text_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError) as exc:
                logger.debug("Failed adding context %s: %s", text_path, exc)
                continue
        # Concatenate limited by model context length heuristically (simple truncation)
        joined_context = "\n\n".join(context_chunks)[:15000]
        response = await self.generation_model.generate(
            query,
            context=[{"type": "text", "text": joined_context}],
        )
        return response, retrieved[0]
