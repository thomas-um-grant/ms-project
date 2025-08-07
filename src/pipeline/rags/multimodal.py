import asyncio
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from PIL import Image
from tqdm import tqdm

from pipeline.rags.base_rag import BaseRAG
from utils.general import pdf_to_images


class MultiModalRAG(BaseRAG):
    """Device-agnostic MultiModal RAG system."""

    def __init__(
        self,
        name: str,
        data_dir: Path,
        configs: dict | None = None,
    ):
        self._validate_params(configs)

        super().__init__(name, data_dir, configs)

    def _resize_image(self, image: Image.Image, max_size: int = 1200) -> Image.Image:
        """Resize image to fit within max_size while maintaining aspect ratio."""
        width, height = image.size

        # If image is already smaller than max_size, return as is
        if max(width, height) <= max_size:
            return image

        # Calculate new dimensions while maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)

        print(f"Resizing image from {width}x{height} to {new_width}x{new_height}")

        # Resize with high-quality resampling
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image

    def _cleanup_memory(self) -> None:
        """Force garbage collection and clear GPU/MPS cache."""
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _log_memory_usage(self, stage: str = "") -> None:
        """Log current memory usage for debugging."""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            if torch.cuda.is_available():
                gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
                print(f"[{stage}] RAM: {memory_mb:.1f}MB, GPU: {gpu_mb:.1f}MB")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print(f"[{stage}] RAM: {memory_mb:.1f}MB, MPS: active")
            else:
                print(f"[{stage}] RAM: {memory_mb:.1f}MB")
        except ImportError:
            print(f"[{stage}] Memory monitoring unavailable (psutil not installed)")

    async def _extract_image_descriptions_batch(
        self,
        images: list[Image.Image],
        corpus_ids: list[str],
        batch_size: int = 1,
        max_retries: int = 3,
    ) -> list[str]:
        """
        Extract descriptions for multiple images with memory management and retry logic.

        Args:
            images: List of PIL images to process
            corpus_ids: List of corresponding corpus IDs for logging
            batch_size: Number of images to process in each batch (default: 1 for stability)
            max_retries: Number of retry attempts per image (default: 3)

        Returns:
            List of descriptions (or "Image description unavailable" for failures)

        """
        descriptions = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_ids = corpus_ids[i : i + batch_size]
            batch_descriptions = []

            self._log_memory_usage(f"Batch {i // batch_size + 1} start")

            # Process each image in the batch with retry logic
            for j, img in enumerate(batch_images):
                description = await self._generate_description_with_retries(
                    img,
                    batch_ids[j],
                    max_retries,
                )
                batch_descriptions.append(description)

                # Visual feedback for success/failure
                status = "✓" if description != "Image description unavailable" else "✗"
                print(f"{status} Generated description for {batch_ids[j]}")

                # Delay between individual images to prevent server overload
                await asyncio.sleep(1.0)

            descriptions.extend(batch_descriptions)

            # Clean up memory and add delay between batches
            self._cleanup_memory()
            self._log_memory_usage(f"Batch {i // batch_size + 1} end")
            await asyncio.sleep(2.0)

        return descriptions

    async def _generate_description_with_retries(
        self,
        image: Image.Image,
        image_id: str,
        max_retries: int,
    ) -> str:
        """
        Generate description for a single image with retry logic and memory cleanup.

        Args:
            image: PIL image to describe
            image_id: ID for logging purposes
            max_retries: Number of retry attempts

        Returns:
            Description string or "Image description unavailable" if all attempts fail

        """
        for attempt in range(max_retries):
            try:
                self._log_memory_usage(f"{image_id} attempt {attempt + 1}")

                description = await self.generation_model.generate(
                    "Describe this image in great details, it will be used as the description metadata for retrieval. Return the description only in plain text.",
                    context=[{"type": "image", "image": image}],
                )

                # Success - cleanup and return
                self._cleanup_memory()
                self._log_memory_usage(f"{image_id} success cleanup")
                return description

            except Exception as e:
                print(f"{image_id} attempt {attempt + 1} failed: {e}")
                self._cleanup_memory()  # Cleanup even on failure

                if attempt < max_retries - 1:
                    # Progressive backoff: 2s, 4s, 6s
                    delay = (attempt + 1) * 2
                    print(f"Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)

        # All attempts failed
        print(f"All {max_retries} attempts failed for {image_id}")
        return "Image description unavailable"

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

    async def extract(
        self,
        documents: list[Path],
        preprocessed: bool = False,
        batch_size: int = 3,
    ) -> None:
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

        if preprocessed:
            batch_images = []
            batch_ids = []

            for i, doc_path in enumerate(documents):
                corpus_id, doc_id = doc_path.stem.split("_")

                # Check if this document is already processed
                if f"{corpus_id}_{doc_id}" in metadata:
                    print(f"Document {corpus_id}_{doc_id} already processed, skipping.")
                    continue

                img = Image.open(doc_path)
                img_resized = self._resize_image(img)
                img_resized.save(corpuses_dir / f"{corpus_id}_{doc_id}.png")

                # Add to batch for description processing
                batch_images.append(img_resized)
                batch_ids.append(f"{corpus_id}_{doc_id}")

                # Process batch when full or at end
                if len(batch_images) >= batch_size or i == len(documents) - 1:
                    self._log_memory_usage(
                        f"Preprocessed batch {len(batch_images)} images",
                    )

                    # Generate descriptions one at a time for maximum stability
                    descriptions = await self._extract_image_descriptions_batch(
                        batch_images,
                        batch_ids,
                        batch_size=1,  # Process one image at a time
                    )

                    # Update metadata
                    for j, description in enumerate(descriptions):
                        corpus_doc_id = batch_ids[j]
                        corpus_id_str, doc_id_str = corpus_doc_id.split("_")

                        metadata[corpus_doc_id] = {
                            "corpus-id": corpus_id_str,
                            "doc-id": doc_id_str,
                            "name": f"{corpus_doc_id}.png",
                            "description": description,
                            "embedded": False,
                        }

                    # Save metadata
                    with self.metadata_path.open("w") as f:
                        json.dump(metadata, f, indent=2)

                    # Clear batch and cleanup memory
                    batch_images = []
                    batch_ids = []
                    self._cleanup_memory()
                    await asyncio.sleep(1.0)  # Increased delay from 0.1 to 1.0 seconds

            return

        # Process documents in smaller batches to manage memory and prevent crashes
        document_batch_size = 3  # Reduced from 5 to 3
        image_description_batch_size = 1  # Process one at a time for maximum stability

        for batch_start in range(0, len(documents), document_batch_size):
            batch_docs = documents[batch_start : batch_start + document_batch_size]
            batch_corpus_ids = []
            batch_images = []

            self._log_memory_usage(
                f"Document batch {batch_start // document_batch_size + 1} start",
            )

            for corpus_id, doc_path in enumerate(
                batch_docs,
                start=max_corpus_id + 1 + batch_start,
            ):
                # Check type of document
                if doc_path.suffix.lower() == ".pdf":
                    images = await pdf_to_images(doc_path)
                elif doc_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    images = [Image.open(doc_path)]
                else:
                    print(f"Unsupported file type: {doc_path.suffix}. Skipping.")
                    continue

                for doc_id, img in enumerate(images, start=1):  # Pages start from 1
                    # Resize image to cap at 1200px max dimension
                    img_resized = self._resize_image(img)
                    img_resized.save(corpuses_dir / f"{corpus_id}_{doc_id}.png")

                    # Store for batch description processing
                    batch_images.append(img_resized)
                    batch_corpus_ids.append(f"{corpus_id}_{doc_id}")

            # Generate descriptions one at a time for maximum stability
            if batch_images:
                descriptions = await self._extract_image_descriptions_batch(
                    batch_images,
                    batch_corpus_ids,
                    batch_size=image_description_batch_size,
                )

                # Update metadata with descriptions
                for i, description in enumerate(descriptions):
                    corpus_doc_id = batch_corpus_ids[i]
                    corpus_id, doc_id = corpus_doc_id.split("_")

                    metadata[corpus_doc_id] = {
                        "corpus-id": int(corpus_id),
                        "doc-id": int(doc_id),
                        "name": f"{corpus_doc_id}.png",
                        "description": description,
                        "embedded": False,
                    }

                # Save metadata after each batch
                with self.metadata_path.open("w") as f:
                    json.dump(metadata, f, indent=2)

            # Clean up memory after each batch
            self._cleanup_memory()
            self._log_memory_usage(
                f"Document batch {batch_start // document_batch_size + 1} end",
            )

            # Longer delay to let memory settle and prevent server overload
            await asyncio.sleep(1.5)  # Increased from 0.2 to 1.5 seconds

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
    ) -> list[list[tuple[dict, float]]]:
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

        # Convert query embeddings to the correct device and dtype
        q_emb_tensor = q_emb_tensor.to(
            dtype=self.device_config.dtype,
            device=self.device_config.device_str,
            non_blocking=True,
        )

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
            all_results.append(query_results)

        return all_results

    async def answer(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[str, list[tuple[dict, float]]]:
        """Generate answer based on the retrieved images or documents."""
        # Retrieve the most relevant images or documents
        results = await self.retrieve(query, top_k=top_k)

        # Generate answers from the retrieved results
        context: list[Any] = []
        for metadata, _ in results[0]:
            context.append(
                {
                    "type": "image",
                    "image": str(
                        self.data_dir
                        / self.knowledge_base
                        / "corpuses"
                        / metadata["name"],
                    ),
                },
            )

        response = await self.generation_model.generate(
            query,
            context=context,
        )

        return response, results[0]
