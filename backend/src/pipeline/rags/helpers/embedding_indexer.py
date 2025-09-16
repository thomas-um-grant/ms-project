import json
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingIndexer:
    """
    Handle embedding operations and index management.

    Supports model-suffixed embedding files (selection handled upstream in
    BaseRAG). Provides image and text indexing helpers.
    """

    def __init__(
        self,
        embedding_model: Any,
        device_config: Any,
        metadata_path: Path,
        embeddings_ids_path: Path,
        embeddings_path: Path,
        batch_size: int = 8,
        expected_batch_embedding_dims: int = 3,
    ) -> None:
        """
        Initialize the EmbeddingIndexer.

        Args:
            embedding_model: The model used for generating embeddings
            device_config: Device configuration for tensor operations
            metadata_path: Path to the metadata JSON file
            embeddings_ids_path: Path to the embeddings IDs JSONL file
            embeddings_path: Path to the embeddings tensor file
            batch_size: Batch size for embedding operations
            expected_batch_embedding_dims: Expected dimensions for batch embeddings

        """
        self.embedding_model = embedding_model
        self.device_config = device_config
        self.batch_size = batch_size
        self.expected_batch_embedding_dims = expected_batch_embedding_dims

        # Set up file paths
        self.metadata_path = metadata_path
        self.embeddings_path = embeddings_path
        self.embeddings_ids_path = embeddings_ids_path

    def load_index(self) -> tuple[list[torch.Tensor], list[str], dict[str, Any]]:
        """
        Load the indexed embeddings and metadata.

        Returns:
            Tuple containing embeddings, embedding IDs, and metadata

        Raises:
            FileNotFoundError: If index files are missing

        """
        if (
            not self.embeddings_path.exists()
            or not self.embeddings_ids_path.exists()
            or not self.metadata_path.exists()
        ):
            msg = "Index files not found. Run indexing first."
            raise FileNotFoundError(msg)

        # Load embeddings using PyTorch with weights_only for security
        embeddings = torch.load(
            self.embeddings_path,
            map_location="cpu",  # Load to CPU first for memory efficiency
            weights_only=True,
        )
        # Upcast to float32 if older indices were saved in fp16/bf16
        upcasted = False
        fixed_embeddings: list[torch.Tensor] = []
        for emb in embeddings:
            if isinstance(emb, torch.Tensor) and emb.dtype != torch.float32:
                fixed_embeddings.append(emb.to(torch.float32))
                upcasted = True
            else:
                fixed_embeddings.append(emb)
        if upcasted:
            try:
                torch.save(fixed_embeddings, self.embeddings_path)
                embeddings = fixed_embeddings
                logger.info("Upcasted stored embeddings to float32 for consistency.")
            except Exception:  # pragma: no cover
                logger.exception(
                    "Failed to persist upcasted embeddings; proceeding in-memory.",
                )

        with self.embeddings_ids_path.open() as f:
            embeddings_ids = [json.loads(line) for line in f]

        with self.metadata_path.open() as f:
            metadata = json.load(f)

        return embeddings, embeddings_ids, metadata

    async def index_images(
        self,
        image_paths: list[Path],
        corpus_ids: list[str],
    ) -> list[str]:
        """
        Index images by generating embeddings.

        Args:
            image_paths: List of paths to images to index
            corpus_ids: List of corresponding corpus IDs

        Returns:
            List of successfully indexed corpus IDs

        """
        embeddings: list[torch.Tensor] = []
        indexed_ids: list[str] = []

        # Load existing embeddings + ids if they exist
        existing_embeddings: list[torch.Tensor] = []
        existing_ids: list[str] = []
        existing_set: set[str] = set()
        if self.embeddings_path.exists() and self.embeddings_ids_path.exists():
            logger.debug(
                "Loading existing embeddings for incremental indexing: %s",
                self.embeddings_path,
            )
            try:
                existing_embeddings = torch.load(
                    self.embeddings_path,
                    map_location="cpu",
                    weights_only=True,
                )
                # Upcast to float32 if older indices were saved in fp16/bf16
                if any(
                    isinstance(t, torch.Tensor) and t.dtype != torch.float32
                    for t in existing_embeddings
                ):
                    existing_embeddings = [
                        (t.to(torch.float32) if isinstance(t, torch.Tensor) else t)
                        for t in existing_embeddings
                    ]
                    try:
                        torch.save(existing_embeddings, self.embeddings_path)
                        logger.info(
                            "Upcasted existing stored embeddings to float32 for consistency.",
                        )
                    except Exception:  # pragma: no cover
                        logger.exception(
                            "Failed to persist upcasted existing embeddings; proceeding in-memory.",
                        )
                with self.embeddings_ids_path.open("r") as f:
                    existing_ids = [json.loads(line) for line in f]
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed loading existing index (%s). Rebuilding from scratch.",
                    exc,
                )
                existing_embeddings = []
                existing_ids = []

            if len(existing_embeddings) != len(existing_ids):
                logger.warning(
                    "Embedding / ID count mismatch (embeddings=%d ids=%d). Discarding existing index.",
                    len(existing_embeddings),
                    len(existing_ids),
                )
                existing_embeddings = []
                existing_ids = []
            existing_set = set(existing_ids)

        # Filter out any corpus_ids that are *already* present to prevent duplicate vectors
        if existing_set:
            filtered_image_paths: list[Path] = []
            filtered_corpus_ids: list[str] = []
            skipped = 0
            for p, cid in zip(image_paths, corpus_ids, strict=False):
                if cid in existing_set:
                    skipped += 1
                    continue
                filtered_image_paths.append(p)
                filtered_corpus_ids.append(cid)
            if skipped:
                logger.info(
                    "Skipped %d images already indexed (existing id reuse prevented). New=%d",
                    skipped,
                    len(filtered_corpus_ids),
                )
            image_paths = filtered_image_paths
            corpus_ids = filtered_corpus_ids

        if not image_paths:
            # Even if no new images, we may still need to dedupe an older index
            if existing_ids and len(existing_ids) != len(existing_set):
                logger.warning(
                    "Index contains %d ids but only %d are unique. Performing maintenance dedup.",
                    len(existing_ids),
                    len(existing_set),
                )
                # Build first occurrence mapping
                seen: set[str] = set()
                dedup_embeddings: list[torch.Tensor] = []
                dedup_ids: list[str] = []
                for emb, cid in zip(
                    existing_embeddings,
                    existing_ids,
                    strict=False,
                ):  # pragma: no cover (maintenance path)
                    if cid in seen:
                        continue
                    seen.add(cid)
                    dedup_embeddings.append(emb)
                    dedup_ids.append(cid)
                torch.save(dedup_embeddings, self.embeddings_path)
                with self.embeddings_ids_path.open("w") as f:
                    for cid in dedup_ids:
                        f.write(json.dumps(cid) + "\n")
                logger.info(
                    "Maintenance dedup complete: %d -> %d unique embeddings.",
                    len(existing_ids),
                    len(dedup_ids),
                )
            else:
                logger.info(
                    "No new images to index (all %d requested IDs already present; index already unique).",
                    len(existing_ids),
                )
            return []

        # Process images in batches
        for i in tqdm(range(0, len(image_paths), self.batch_size)):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_ids = corpus_ids[i : i + self.batch_size]

            # Load images for this batch
            batch_images = [Image.open(path) for path in batch_paths]

            try:
                # Generate embeddings using async embedding method
                batch_embeddings = await self.embedding_model.embed_images(
                    batch_images,
                    dtype=self.device_config.dtype,
                )

                # Since we now get individual tensors, just append them directly
                for j, embedding in enumerate(batch_embeddings):
                    cid = batch_ids[j]
                    # Guard against accidental duplicate within the same new batch set
                    if cid in existing_set or cid in indexed_ids:
                        logger.debug(
                            "Duplicate id '%s' encountered in batch; skipping vector.",
                            cid,
                        )
                        continue
                    embeddings.append(embedding)
                    indexed_ids.append(cid)

            except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                print(f"Failed to process batch starting at index {i}: {e}")
                # Skip this batch and continue
                continue

        # Combine with existing embeddings / ids
        # Combine (existing first to preserve stability)
        all_embeddings = existing_embeddings + embeddings
        all_ids = existing_ids + indexed_ids

        # Final safety dedupe (should be no-ops normally)
        if all_ids:
            seen_final: set[str] = set()
            dedup_embeddings_final: list[torch.Tensor] = []
            dedup_ids_final: list[str] = []
            for emb, cid in zip(all_embeddings, all_ids, strict=False):
                if cid in seen_final:
                    continue
                seen_final.add(cid)
                dedup_embeddings_final.append(emb)
                dedup_ids_final.append(cid)
            if len(dedup_ids_final) != len(all_ids):
                logger.warning(
                    "Removed %d duplicate embedding ids during save (final).",
                    len(all_ids) - len(dedup_ids_final),
                )
            all_embeddings = dedup_embeddings_final
            all_ids = dedup_ids_final

        # Save embeddings + ids atomically when new embeddings added
        if indexed_ids:
            torch.save(all_embeddings, self.embeddings_path)
            with self.embeddings_ids_path.open("w") as f:
                for corp_id in all_ids:
                    f.write(json.dumps(corp_id) + "\n")
            logger.info(
                "Indexed %d new images (total %d unique embeddings).",
                len(indexed_ids),
                len(all_ids),
            )
        else:
            logger.info(
                "No new embeddings generated; index unchanged (%d unique).",
                len(all_ids),
            )

        return indexed_ids

    async def index_texts(
        self,
        texts: list[str],
        corpus_ids: list[str],
    ) -> list[str]:
        """
        Index textual documents by generating embeddings.

        Args:
            texts: Text content list
            corpus_ids: Matching corpus IDs

        Returns:
            List of successfully indexed corpus IDs

        """
        if not texts:
            return []

        existing_embeddings: list[torch.Tensor] = []
        existing_ids: list[str] = []
        if self.embeddings_path.exists() and self.embeddings_ids_path.exists():
            try:
                existing_embeddings = torch.load(
                    self.embeddings_path,
                    map_location="cpu",
                    weights_only=True,
                )
                with self.embeddings_ids_path.open("r") as f:
                    existing_ids = [json.loads(line) for line in f]
            except (
                OSError,
                RuntimeError,
                ValueError,
                json.JSONDecodeError,
            ):  # pragma: no cover
                existing_embeddings = []  # fallback rebuild
                existing_ids = []

        new_embeddings: list[torch.Tensor] = []
        new_ids: list[str] = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i : i + self.batch_size]
            batch_ids = corpus_ids[i : i + self.batch_size]
            try:
                batch_embs = await self.embedding_model.embed_texts(batch_texts)
            except (RuntimeError, ValueError) as e:  # pragma: no cover
                print(f"Failed to process text batch starting at index {i}: {e}")
                continue
            for j, emb in enumerate(batch_embs):
                if isinstance(emb, torch.Tensor) and emb.numel() > 0:
                    new_embeddings.append(emb)
                    new_ids.append(batch_ids[j])

        if not new_embeddings:
            return []

        all_embeddings = existing_embeddings + new_embeddings
        all_ids = existing_ids + new_ids

        torch.save(all_embeddings, self.embeddings_path)
        with self.embeddings_ids_path.open("w") as f:
            for cid in all_ids:
                f.write(json.dumps(cid) + "\n")

        return new_ids

    def get_existing_embedding_ids(self) -> list[str]:
        """
        Get list of existing embedding IDs.

        Returns:
            List of embedding IDs that are already indexed

        """
        if not self.embeddings_ids_path.exists():
            return []

        with self.embeddings_ids_path.open("r") as f:
            return [json.loads(line) for line in f]
