import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm


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
        embeddings = []
        indexed_ids = []

        # Load existing embeddings + ids if they exist
        existing_embeddings: list[torch.Tensor] = []
        existing_ids: list[str] = []
        if self.embeddings_path.exists() and self.embeddings_ids_path.exists():
            print("Loading existing embeddings...")
            existing_embeddings = torch.load(
                self.embeddings_path,
                map_location="cpu",
                weights_only=True,
            )
            with self.embeddings_ids_path.open("r") as f:
                existing_ids = [json.loads(line) for line in f]

            if len(existing_embeddings) != len(existing_ids):
                print(
                    f"[WARNING] Embedding / ID count mismatch (embeddings={len(existing_embeddings)} ids={len(existing_ids)}). Discarding existing index.",
                )
                existing_embeddings = []
                existing_ids = []

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
                    embeddings.append(embedding)
                    indexed_ids.append(batch_ids[j])

            except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                print(f"Failed to process batch starting at index {i}: {e}")
                # Skip this batch and continue
                continue

        # Combine with existing embeddings / ids
        all_embeddings = existing_embeddings + embeddings
        all_ids = existing_ids + indexed_ids

        # Save embeddings + ids atomically when new embeddings added
        if embeddings:
            torch.save(all_embeddings, self.embeddings_path)
            with self.embeddings_ids_path.open("w") as f:
                for corp_id in all_ids:
                    f.write(json.dumps(corp_id) + "\n")

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
