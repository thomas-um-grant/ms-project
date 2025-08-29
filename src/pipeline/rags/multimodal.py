import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pipeline.rags.base_rag import BaseRAG
from pipeline.rags.helpers import EmbeddingIndexer, ImageProcessor, MetadataManager
from utils.device import cleanup_memory, log_memory_usage
from utils.general import pdf_to_images, resize_image

logger = logging.getLogger(__name__)


class MultiModalRAG(BaseRAG):
    """Device-agnostic MultiModal RAG system."""

    def __init__(
        self,
        name: str,
        data_dir: Path,
        configs: dict | None = None,
    ):
        self._validate_params(configs)
        configs = configs or {}

        super().__init__(name, data_dir, configs)

        defaults = self._load_defaults()
        self.extraction_batch_size = configs.get(
            "extraction_batch_size",
            defaults["processing"]["extraction"]["batch_size"],
        )
        self.extraction_config = configs.get(
            "extraction_config",
            defaults["processing"]["extraction"],
        )
        self.processing_summary_template = configs.get(
            "processing_summary_template",
            defaults["logging"]["processing_summary_template"],
        )

        self.image_processor = ImageProcessor(
            self.generation_model,
            image_description_prompt=configs.get(
                "image_description_prompt",
                defaults["prompts"]["image_description"],
            ),
            fallback_description=configs.get(
                "fallback_description",
                defaults["prompts"]["fallback_description"],
            ),
            batch_completion_template=configs.get(
                "batch_completion_template",
                defaults["logging"]["batch_completion_template"],
            ),
        )
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

    def _validate_params(self, configs):
        """Validate configuration parameters."""
        if configs is None:
            return

        if configs.get("top_k") is not None and (
            not isinstance(configs.get("top_k"), int) or configs.get("top_k") <= 0
        ):
            raise ValueError("Top K must be a positive integer.")

        if configs.get("pruning_threshold") is not None and not isinstance(
            configs.get("pruning_threshold"),
            (int, float),
        ):
            raise TypeError("Pruning threshold must be a number.")

    async def extract(
        self,
        documents: list[Path],
        *,
        preprocessed: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """Extract relevant corpuses for retrieval."""
        if batch_size is None:
            batch_size = self.extraction_batch_size
        # Use configured batch size if not provided
        if batch_size is None:
            batch_size = self.extraction_config["batch_size"]
        # Extract individual pages from PDFs into images, and store them as pngs in corpuses directory
        # Create metadata for each image with embedded status flag
        # Step 1: Prepare all image paths and names based on preprocessed flag
        image_data = []  # List of (image_path, doc_name) tuples

        if preprocessed:
            # For preprocessed images, just collect existing image files
            for doc_path in documents:
                splits = doc_path.stem.split("_")
                if len(splits) < 2:
                    corpus_id = str(doc_path.stem)
                    doc_id = None
                elif len(splits) == 2:
                    corpus_id, doc_id = splits
                else:
                    corpus_id, doc_id = "_".join(splits[:-1]), splits[-1]

                doc_name = f"{corpus_id}_{doc_id}" if doc_id else corpus_id

                # Check if already extracted
                if self.metadata_manager.is_document_processed(doc_name):
                    print(f"Document {doc_name} already extracted, skipping.")
                    continue

                # Resize and save image
                img = Image.open(doc_path)
                img_resized = resize_image(img)
                img_resized.save(self.corpuses_dir / f"{doc_name}.png")

                image_data.append((self.corpuses_dir / f"{doc_name}.png", doc_name))
        else:
            # For non-preprocessed, extract from raw file first
            max_corpus_id = self.metadata_manager.get_max_corpus_id()

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

                for document_id, img in enumerate(images, start=1):
                    doc_name = f"{corpus_counter}_{document_id}"

                    # Check if already processed
                    if self.metadata_manager.is_document_processed(doc_name):
                        print(f"Document {doc_name} already processed, skipping.")
                        continue

                    # Resize and save image
                    img_resized = resize_image(img)
                    img_resized.save(self.corpuses_dir / f"{doc_name}.png")

                    image_data.append((self.corpuses_dir / f"{doc_name}.png", doc_name))

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
            if (
                len(batch_images) >= batch_size or i == len(image_data) - 1
            ) and batch_images:  # Only process if we have images
                current_batch += 1
                log_memory_usage(
                    f"Processing batch {current_batch}/{total_batches}",
                )

                # Generate descriptions with batching (batch_size config handled by image_processor)
                if (
                    self.generation_disabled
                    or self.image_processor.generation_model is None
                ):
                    # Provide fallback descriptions without model inference
                    descriptions = [
                        self.image_processor.fallback_description for _ in batch_images
                    ]
                else:
                    descriptions = (
                        await self.image_processor.extract_descriptions_batch(
                            batch_images,
                            batch_ids,
                        )
                    )

                # Update metadata using metadata manager - batch update for efficiency
                metadata_entries = [
                    (batch_ids[j], f"{batch_ids[j]}.png", descriptions[j], False)
                    for j in range(len(descriptions))
                ]
                self.metadata_manager.update_metadata_batch(metadata_entries)

                print(
                    f"✅ Batch {current_batch}/{total_batches} saved. Progress: {processed_count} processed, {skipped_count} skipped",
                )

                # Clear batch and cleanup memory
                batch_images = []
                batch_ids = []
                cleanup_memory()
                await asyncio.sleep(self.extraction_config["sleep_between_images"])

        # Final summary using template
        print(
            self.processing_summary_template.format(
                processed=processed_count,
                skipped=skipped_count,
            ),
        )

        # Ensure all metadata changes are saved to disk
        self.metadata_manager.flush()

    async def index(self) -> None:
        """Index image pages with device-agnostic processing using async embedding."""
        # Get unembedded documents from metadata manager
        unembedded_docs = self.metadata_manager.get_unembedded_documents(
            self.embedding_model_tag,
        )

        if not unembedded_docs:
            print("No documents to index.")
            return

        # Prepare image paths and corpus IDs
        image_paths = []
        corpus_ids = []

        for doc_id, doc_data in unembedded_docs:
            corpus_ids.append(doc_id)
            image_paths.append(
                self.corpuses_dir / doc_data["name"],
            )

        # Index images using the embedding indexer
        indexed_ids = await self.embedding_indexer.index_images(image_paths, corpus_ids)

        # Mark indexed documents as embedded
        if indexed_ids:
            self.metadata_manager.mark_as_embedded(
                indexed_ids,
                self.embedding_model_tag,
            )
            print(f"Indexed {len(indexed_ids)} images.")
        else:
            print("No images were successfully indexed.")

    async def retrieve(
        self,
        queries: str | list[str],
        top_k: int | None = None,
    ) -> list[list[tuple[dict, float]]]:
        """
        Retrieve top-k most similar documents for single or multiple queries using async embedding.

        Args:
            queries: Text query or list of text queries to search for
            top_k: Number of top results to return per query (uses config default if None)

        Returns:
            List of (metadata, score) tuples. For multiple queries, results are flattened.

        """
        # Use configured top_k if not provided
        if top_k is None:
            top_k = self.top_k if self.top_k else 100

        # Cap at 100 anyways
        top_k = min(top_k, 100)

        # Convert single query to list for uniform processing
        query_texts = [queries] if isinstance(queries, str) else queries

        if not query_texts:
            return []

        embeddings, embedding_ids, metadata = self.embedding_indexer.load_index()

        if len(embeddings) != len(embedding_ids):
            msg = f"Corrupt index: {len(embeddings)} embeddings vs {len(embedding_ids)} ids"
            raise RuntimeError(msg)
        if not embeddings:
            return [[] for _ in query_texts]

        # Get query embeddings (already cleaned by embedding model)
        q_emb_tensors = await self.embedding_model.embed_texts(query_texts)

        # Convert embeddings to correct device/dtype
        query_vectors = [
            q_tensor.to(
                dtype=self.device_config.dtype,
                device=self.device_config.device_str,
                non_blocking=True,
            )
            for q_tensor in q_emb_tensors
        ]

        doc_vectors = [
            emb_tensor.to(
                dtype=self.device_config.dtype,
                device=self.device_config.device_str,
                non_blocking=True,
            )
            for emb_tensor in embeddings
        ]

        # Process each query and collect results
        all_results = []
        for i in tqdm(range(len(query_texts))):
            scores = self.embedding_model.processor.score(
                [query_vectors[i]],
                doc_vectors,
            )

            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            if scores.ndim > 1:
                scores = scores.flatten()

            # Get top-k results for this query
            ids = np.argsort(scores)[-top_k:][::-1]
            query_results = [
                (metadata[embedding_ids[j]], float(scores[j])) for j in ids
            ]
            all_results.append(query_results)

        # If rerank is enabled, apply it to the results
        if self.auto_rerank and self.reranker_method:
            try:
                return await self.rerank(
                    queries=query_texts,
                    retrieved_corpuses=all_results,
                    method=self.reranker_method,
                )
            except Exception:  # pragma: no cover - fail open
                logger.exception(
                    "Automatic rerank failed; returning raw retrieval results",
                )

        return all_results

    async def answer(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[str, list[tuple[dict, float]]]:
        """Generate answer based on the retrieved images or documents."""
        # Use configured top_k if not provided
        if top_k is None:
            top_k = self.top_k

        # Retrieve the most relevant images or documents
        results = await self.retrieve(query, top_k=top_k)

        # Generate answers from the retrieved results
        context: list[Any] = []
        for metadata, _ in results[0]:
            context.append(
                {
                    "type": "image",
                    "image": str(
                        self.corpuses_dir / metadata["name"],
                    ),
                },
            )

        if self.generation_disabled or self.generation_model is None:
            msg = "Generation disabled: only retrieval results returned. Enable generation to produce answers."
            return msg, results[0]

        response = await self.generation_model.generate(query, context=context)
        return response, results[0]
