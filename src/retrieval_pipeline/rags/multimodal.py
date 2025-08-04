import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add src path for imports
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from retrieval_pipeline.rags.base_rag import BaseRAG
from retrieval_pipeline.utils import pdf_to_images


class MultiModalRAG(BaseRAG):
    """Device-agnostic MultiModal RAG system."""

    def __init__(
        self,
        name: str,
        data_dir: Path,
        embedding_model: str = "colqwen2",
        generation_model: str = "colqwen2",
        configs: dict | None = None,
    ):
        self._validate_params(configs)

        super().__init__(name, data_dir, embedding_model, generation_model, configs)

    def _validate_params(
        self,
        configs,
    ):
        if configs is None:
            return

        # TODO: Add validations to ensure configs defined are valid for multimodal RAG

        if configs.get("top_k") is not None and (
            not isinstance(configs.get("top_k"), int) or configs.get("top_k") <= 0
        ):
            e = "Top K must be a positive integer."
            raise ValueError(e)

        if configs.get("pruning_threshold") is not None and not isinstance(
            configs.get("pruning_threshold"),
            int | float,
        ):
            e = "Pruning threshold must be a number."
            raise TypeError(e)

    def _safe_tensor_convert(
        self,
        tensor: torch.Tensor,
        target_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Safely convert tensor with NaN handling and dtype conversion."""
        # Convert to target dtype if needed
        if tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)

        # Move to CPU for storage
        tensor_cpu = tensor.cpu()

        # Handle NaN values
        if torch.isnan(tensor_cpu).any():
            print("Warning: NaN values detected, replacing with zeros")
            tensor_cpu = torch.nan_to_num(tensor_cpu, nan=0.0)

        return tensor_cpu

    def _load_index(self) -> tuple[list[torch.Tensor], list, dict]:
        """Load the indexed embeddings and metadata."""
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

    async def _extract_image_description(self, image: Image.Image) -> str:
        # Use generation model to extract image description
        description = await self.generation_model.generate(
            "Describe this image in great details, it will be used as the description metadata for retrieval. Return the description only in plain text.",
            context=[{"type": "image", "image": image}],
        )
        return description

    async def extract(self, documents: list[Path]) -> None:
        """Extract relevant corpuses for retrieval."""
        # Extract individual pages from PDFs into images, and store them as pngs in corpuses directory
        # Create metadata for each image with 'embedded' flag set to False
        corpuses_dir = self.data_dir / "corpuses"
        corpuses_dir.mkdir(parents=True, exist_ok=True)

        metadata = {}
        if self.metadata_path.exists():
            with self.metadata_path.open("r") as f:
                metadata = json.load(f)

        # Get existing corpus IDs to avoid duplicates
        max_corpus_id = max(
            (int(data["corpus-id"]) for data in metadata.values()),
            default=-1,
        )

        for corpus_id, doc_path in tqdm(enumerate(documents, start=max_corpus_id + 1)):
            # Convert each page of the PDF to an image
            images = await pdf_to_images(doc_path)

            for doc_id, img in tqdm(enumerate(images, start=1)):  # Pages start from 1
                img.save(corpuses_dir / f"{corpus_id}_{doc_id}.png")

                # Extract description of the image
                description = await self._extract_image_description(img)

                metadata[f"{corpus_id}_{doc_id}"] = {
                    "corpus-id": corpus_id,
                    "doc-id": doc_id,
                    "name": f"{corpus_id}_{doc_id}.png",
                    "description": description,
                    "embedded": False,
                }

            with self.metadata_path.open("w") as f:
                json.dump(metadata, f, indent=2)

    async def index(self) -> None:
        """
        Index image pages with device-agnostic processing using async embedding.

        Args:
            corpuses: List of dictionaries, each containing corpus information.

        """
        metadata = {}
        embs_ids = []
        embs = []

        if self.metadata_path.exists():
            with self.metadata_path.open("r") as f:
                metadata = json.load(f)

        if self.embeddings_ids_path.exists():
            with self.embeddings_ids_path.open("r") as f:
                embs_ids = [json.loads(line) for line in f]

        corpuses_to_embed = []
        images_paths = []
        for data_id, data in metadata.items():
            if not data["embedded"]:
                corpuses_to_embed.append(data_id)
                images_paths.append(self.data_dir / "corpuses" / data["name"])

        for i in tqdm(range(0, len(images_paths), self.batch_size)):
            corps = corpuses_to_embed[i : i + self.batch_size]
            imgs = [
                Image.open(img_path)
                for img_path in images_paths[i : i + self.batch_size]
            ]

            # Use async embedding method
            embeddings = await self.embedding_model.embed_images(
                imgs,
                batch_size=len(imgs),
            )

            # Safe tensor conversion and dtype handling
            embeddings_processed = self._safe_tensor_convert(
                embeddings,
                self.device_config.dtype,
            )

            # Add individual embeddings to the list
            # Handle both single tensor and batch tensor cases
            batch_embedding_dims = 3  # Expected dimensions for batch embeddings
            if (
                embeddings_processed.dim() == batch_embedding_dims
            ):  # Batch of embeddings
                for j in range(embeddings_processed.size(0)):
                    embs.append(embeddings_processed[j])
                    metadata[corps[j]]["embedded"] = True
                    embs_ids.append(corps[j])
            else:  # Single embedding
                embs.append(embeddings_processed)
                metadata[corps[0]]["embedded"] = True
                embs_ids.append(corps[0])

        # Concatenate all embeddings (existing + new)
        if self.embeddings_path.exists():
            print("Some embeddings already exist, loading existing index.")
            existing_embs = torch.load(
                self.embeddings_path,
                map_location="cpu",
                weights_only=True,
            )
            full_embs = existing_embs + embs
        else:
            full_embs = embs

        # Save embeddings using PyTorch (handles variable shapes naturally)
        torch.save(full_embs, self.embeddings_path)

        with self.metadata_path.open("w") as meta_file:
            json.dump(metadata, meta_file, indent=2)

        with self.embeddings_ids_path.open("w") as f:
            for corp_id in embs_ids:
                f.write(json.dumps(corp_id) + "\n")

        print(f"Indexed {len(images_paths)} images.")

    async def retrieve(
        self,
        queries: str | list[str],
        top_k: int = 5,
    ) -> list[tuple[dict, float]]:
        """
        Retrieve top-k most similar documents for single or multiple queries using async embedding.

        Args:
            queries: Text query or list of text queries to search for
            top_k: Number of top results to return per query

        Returns:
            List of (metadata, score) tuples. For multiple queries, results are flattened.

        """
        # Convert single query to list for uniform processing
        query_texts = [queries] if isinstance(queries, str) else queries

        if not query_texts:
            return []

        embeddings, embedding_ids, metadata = self._load_index()

        # Use async text embedding method for all queries
        q_emb_tensor = await self.embedding_model.embed_texts(query_texts)

        # Handle NaN values in query embeddings
        if torch.isnan(q_emb_tensor).any():
            print("Warning: NaN values found in query embeddings, cleaning...")
            q_emb_tensor = torch.nan_to_num(q_emb_tensor, nan=0.0)

        # Convert embeddings to the correct device and dtype efficiently
        doc_vectors = []
        for emb_tensor in embeddings:
            # Move to correct device and ensure correct dtype
            doc_vector = emb_tensor.to(
                dtype=self.device_config.dtype,
                device=self.device_config.device_str,
                non_blocking=True,  # Async transfer for better performance
            )
            doc_vectors.append(doc_vector)

        # Process each query and collect all results
        all_results = []
        for i in tqdm(range(len(query_texts))):
            query_vector = q_emb_tensor[i]  # Get the i-th query vector

            # Use processor.score method for proper late interaction
            scores = self.embedding_model.processor.score([query_vector], doc_vectors)

            # Convert scores to numpy array if needed
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            # Flatten scores if they have extra dimensions
            if scores.ndim > 1:
                scores = scores.flatten()

            # Get top-k results for this query
            ids = np.argsort(scores)[-top_k:][::-1]
            query_results = [
                (metadata[embedding_ids[j]], float(scores[j])) for j in ids
            ]
            all_results.extend(query_results)

        return all_results

    async def answer(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[str, list[tuple[dict, float]]]:
        """Generate answers based on the retrieved images or documents."""
        # Retrieve the most relevant images or documents
        results = await self.retrieve(query, top_k=top_k)

        # Generate answers from the retrieved results
        context: list[Any] = []
        for metadata, _ in results:
            context.append(
                {
                    "type": "image",
                    "image": str(self.data_dir / "corpuses" / metadata["name"]),
                },
            )

        response = await self.generation_model.generate(
            query,
            context=context,
        )

        return response, results
