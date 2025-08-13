import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm


class EmbeddingIndexer:
    """Handles embedding operations and index management."""

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

        # Load existing embeddings if they exist
        existing_embeddings = []
        if self.embeddings_path.exists():
            print("Loading existing embeddings...")
            existing_embeddings = torch.load(
                self.embeddings_path,
                map_location="cpu",
                weights_only=True,
            )

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

        # Combine with existing embeddings
        all_embeddings = existing_embeddings + embeddings

        # Save embeddings to disk
        if embeddings:  # Only save if we have new embeddings
            torch.save(all_embeddings, self.embeddings_path)

            # Save embedding IDs
            with self.embeddings_ids_path.open("w") as f:
                for corp_id in indexed_ids:
                    f.write(json.dumps(corp_id) + "\n")

        return indexed_ids

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
