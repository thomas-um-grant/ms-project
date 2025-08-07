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
from utils.general import pdf_to_images, resize_image
from utils.tensor import safe_tensor_convert


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
        batch_size: int = 4,
        max_retries: int = 3,
    ) -> list[str]:
        """
        Extract descriptions for multiple images.

        Args:
            images: List of PIL images to process
            corpus_ids: List of corresponding corpus IDs for logging
            batch_size: Number of images to process in each batch (default: 4)
            max_retries: Number of retry attempts per image (default: 3)

        Returns:
            List of descriptions (or "Image description unavailable" for failures)

        """
        descriptions = []
        total_batches = (len(images) + batch_size - 1) // batch_size

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_ids = corpus_ids[i : i + batch_size]

            batch_num = i // batch_size + 1
            self._log_memory_usage(f"Batch {batch_num}/{total_batches} start")

            # Process each image in the batch with retry logic
            batch_descriptions = []
            for j, img in enumerate(batch_images):
                description = await self._generate_description_with_retries(
                    img,
                    batch_ids[j],
                    max_retries,
                )
                batch_descriptions.append(description)

                # Minimal delay between individual images
                if j < len(batch_images) - 1:
                    await asyncio.sleep(0.1)

            descriptions.extend(batch_descriptions)

            # Show batch completion status
            successful = sum(
                1
                for desc in batch_descriptions
                if desc != "Image description unavailable"
            )
            print(
                f"✓ Batch {batch_num}/{total_batches}: {successful}/{len(batch_descriptions)} successful",
            )

            if batch_num % 3 == 0:  # Cleanup every 3 batches
                self._cleanup_memory()
                self._log_memory_usage(f"Cleanup after batch {batch_num}")
                await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.05)

        return descriptions

    async def _generate_description_with_retries(
        self,
        image: Image.Image,
        image_id: str,
        max_retries: int,
    ) -> str:
        """
        Generate description for a single image with retry logic.

        Args:
            image: PIL image to describe
            image_id: ID for logging purposes
            max_retries: Number of retry attempts

        Returns:
            Description string or "Image description unavailable" if all attempts fail

        """
        for attempt in range(max_retries):
            try:
                description = await self.generation_model.generate(
                    "Describe this image in great details, it will be used as the description metadata for retrieval. Return the description only in plain text.",
                    context=[{"type": "image", "image": image}],
                )

                return description

            except Exception as e:
                print(f"{image_id} attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    # backoff: 1s, 2s, 3s
                    delay = attempt + 1
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

        # Step 1: Prepare all image paths and names based on preprocessed flag
        image_data = []  # List of (image_path, doc_name) tuples

        if preprocessed:
            # For preprocessed images, just collect existing image files
            for doc_path in documents:
                splits = doc_path.stem.split("_")
                if len(splits) < 2:
                    corpus_id = str(doc_path.stem)
                    doc_id = None
                else:
                    corpus_id, doc_id = splits

                doc_name = f"{corpus_id}_{doc_id}" if doc_id else corpus_id

                # Check if already extracted
                if doc_name in metadata:
                    print(f"Document {doc_name} already extracted, skipping.")
                    continue

                # Resize and save image
                img = Image.open(doc_path)
                img_resized = resize_image(img)
                img_resized.save(corpuses_dir / f"{doc_name}.png")

                image_data.append((corpuses_dir / f"{doc_name}.png", doc_name))
        else:
            # For non-preprocessed, extract from raw file first
            max_corpus_id = max(
                (int(data["corpus-id"]) for data in metadata.values()),
                default=-1,
            )

            corpus_counter = max_corpus_id + 1
            for doc_path in documents:
                # Check document type and extract images
                if doc_path.suffix.lower() == ".pdf":
                    images = await pdf_to_images(doc_path)
                elif doc_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    images = [Image.open(doc_path)]
                else:
                    print(f"Unsupported file type: {doc_path.suffix}. Skipping.")
                    continue

                for doc_id, img in enumerate(images, start=1):
                    doc_name = f"{corpus_counter}_{doc_id}"

                    # Check if already processed
                    if doc_name in metadata:
                        print(f"Document {doc_name} already processed, skipping.")
                        continue

                    # Resize and save image
                    img_resized = resize_image(img)
                    img_resized.save(corpuses_dir / f"{doc_name}.png")

                    image_data.append((corpuses_dir / f"{doc_name}.png", doc_name))

                corpus_counter += 1

        # Step 2: Process all images in batches
        batch_images = []
        batch_ids = []
        processed_count = 0
        skipped_count = len(documents) - len(list(image_data))
        total_batches = (
            (len(image_data) + batch_size - 1) // batch_size if image_data else 0
        )
        current_batch = 0

        for i, (image_path, doc_name) in enumerate(image_data):
            # Load the saved image
            img = Image.open(image_path)
            batch_images.append(img)
            batch_ids.append(doc_name)
            processed_count += 1

            # Process batch when full or at end
            if (len(batch_images) >= batch_size or i == len(image_data) - 1) and batch_images:  # Only process if we have images
                current_batch += 1
                self._log_memory_usage(
                    f"Processing batch {current_batch}/{total_batches}",
                )

                    # Generate descriptions with batching
                    descriptions = await self._extract_image_descriptions_batch(
                        batch_images,
                        batch_ids,
                        batch_size=4,
                    )

                    # Update metadata
                    for j, description in enumerate(descriptions):
                        corpus_doc_id = batch_ids[j]

                        id_split = corpus_doc_id.split("_")
                        if len(id_split) < 2:
                            corpus_id_str = str(corpus_doc_id)
                            doc_id_str = ""
                        else:
                            corpus_id_str, doc_id_str = id_split

                        metadata[corpus_doc_id] = {
                            "corpus-id": corpus_id_str,
                            "doc-id": doc_id_str,
                            "name": f"{corpus_doc_id}.png",
                            "description": description,
                            "embedded": False,
                        }

                    # Save metadata after each batch
                    with self.metadata_path.open("w") as f:
                        json.dump(metadata, f, indent=2)

                    print(
                        f"✅ Batch {current_batch}/{total_batches} saved. Progress: {processed_count} processed, {skipped_count} skipped",
                    )

                    # Clear batch and cleanup memory
                    batch_images = []
                    batch_ids = []
                    self._cleanup_memory()
                    await asyncio.sleep(0.1)

        # Final summary
        print(
            f"🏁 Processing complete! {processed_count} documents processed, {skipped_count} skipped",
        )

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
            embeddings_processed = safe_tensor_convert(
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
