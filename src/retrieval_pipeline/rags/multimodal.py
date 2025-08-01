import asyncio
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

# Add src path for imports
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from database.dbs_manager import VectorDB
from database.schemas.vector_db_schemas import CollectionSchemas
from retrieval_pipeline.models.embedding_models import setup_embedding_model
from retrieval_pipeline.models.generation_models import setup_generation_model
from retrieval_pipeline.rags.base_rag import BaseRAG
from retrieval_pipeline.utils import resize_image

logging.basicConfig(
    level=logging.INFO,
    filename="multimodal_rag.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MultiModalRAG(BaseRAG):
    def __init__(
        self,
        name: str,
        vector_db: VectorDB,
        embedding_model: str = "colqwen2",
        generation_model: str = "colqwen2",
        configs: dict | None = None,
    ):
        super().__init__(vector_db, name)

        self._validate_params(configs)

        self.embedding_model = setup_embedding_model(
            embedding_model,
            device=self.device,
        )
        self.generation_model = setup_generation_model(
            generation_model,
            device=self.device,
        )

        if configs:
            self.chunking_strategy = configs.get(
                "chunking_strategy",
                "page_chunking",
            )  # ["page_chunking", "token_chunking", "semantic_chunking", ""]
            self.query_enhancement = configs.get(
                "query_enhancement",
                "none",
            )  # ["none", "hyde", "step_back_prompting", "multi_query"]
            self.retrieval_strategy = configs.get(
                "retrieval_strategy",
                "semantic",
            )  # ["semantic", "bm25", "hybrid"]
            self.similarity_metric = configs.get(
                "similarity_metric",
                "max_sim",
            )  # ["cosine", "euclidean", "max_sim"]
            self.routing_strategy = configs.get(
                "routing_strategy",
                "none",
            )  # ["none", "semantic"]
            self.top_k = configs.get("top_k", 5)  # Number of top results to retrieve
            self.pruning_threshold = configs.get(
                "pruning_threshold",
                0.0,
            )  # Whether to prune the retrieved passages based on relevance

    def _get_collection_schema(self) -> dict:
        return CollectionSchemas.multimodal_image_schema()

    async def index(self, images_path: list[Path], batch_size: int = 4):
        """Index images for retrieval with batching to avoid memory issues."""
        # Filter out non-existent image paths
        existing_images_path = [path for path in images_path if path.exists()]

        logger.info(
            f"Found {len(existing_images_path)} existing images out of {len(images_path)} total"
        )

        if not existing_images_path:
            logger.warning("No existing images found to index!")
            return

        # Convert paths to PIL Images
        images = []
        for image_path in existing_images_path:
            try:
                images.append(Image.open(image_path))
                logger.info(f"Successfully opened: {image_path}")
            except Exception:
                logger.exception(f"Failed to open {image_path}")
                continue

        if not images:
            logger.error("No images could be opened!")
            return

        # Resize and embed images in batches to avoid memory issues
        self.images = [resize_image(image) for image in images]

        all_embeddings = []
        logger.info(
            f"Processing {len(self.images)} images in batches of {batch_size}...",
        )

        for i in range(0, len(self.images), batch_size):
            batch_images = self.images[i : i + batch_size]

            if type(batch_images) is not list:
                batch_images = [batch_images]

            logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(self.images) + batch_size - 1) // batch_size}",
            )

            batch_embeddings = await self.embedding_model.embed_images(batch_images)
            # batch_embeddings is a tensor of shape (batch_size, tokens, dim)
            # Convert each image's multi-vector embedding to a list
            for batch_idx in range(batch_embeddings.shape[0]):
                image_embedding = batch_embeddings[batch_idx]  # Shape: (tokens, dim)
                all_embeddings.append(image_embedding.tolist())

            logger.info(
                f"Batch {i // batch_size + 1} processed: {len(batch_embeddings)} embeddings",
            )

            # Clear GPU memory if possible
            if hasattr(self.embedding_model, "model") and hasattr(
                self.embedding_model.model,
                "cpu",
            ):
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Store in db - use only existing images
        data_points = []
        for image_path, embedding in zip(
            existing_images_path[
                : len(all_embeddings)
            ],  # Match the actual embeddings count
            all_embeddings,
            strict=True,
        ):
            data_points.append(
                {
                    "properties": {
                        "dataset_name": self.collection_name,
                        "corpus_id": image_path.stem.split("_")[0],
                        "doc_id": image_path.stem.split("_")[1],
                        "image_path": str(image_path),
                    },
                    "vector": {"multi_vector": embedding},
                },
            )

        # Insert into vector database
        await self.vector_db.insert_vectors(
            self.collection_name,
            data_points,
            batch_size,
        )

    async def retrieve(self, texts: list[str], top_k: int = 5):
        # TODO: Query enhancement

        # Embed queries
        query_embeddings = await self.embedding_model.embed_texts(texts)

        # TODO: Retrieve using strategy and similarity metric
        all_results = []
        for query_embedding in query_embeddings:
            results = await self.vector_db.search_vectors(
                self.collection_name,
                query_embedding,
                top_k,
            )
            all_results.extend(results)

        # TODO: Prune
        pruned_results = all_results

        # Return top K results
        return pruned_results

    async def answer(self, queries: list[str]):
        # TODO: Implement answer generation
        pass

    def _validate_params(
        self,
        configs,
    ):
        if configs is None:
            return

        if configs.get("embedding_model") and configs.get("embedding_model") not in [
            "colqwen2",
        ]:
            e = f"Embedding model '{configs.get('embedding_model')}' is not available."
            raise ValueError(e)

        if configs.get("generation_model") and configs.get("generation_model") not in [
            "colqwen2",
        ]:
            e = f"Generation model '{configs.get('generation_model')}' is not available."
            raise ValueError(e)

        if configs.get("chunking_strategy") and configs.get(
            "chunking_strategy",
        ) not in ["page_chunking"]:
            e = f"Chunking strategy '{configs.get('chunking_strategy')}' is not supported."
            raise ValueError(e)

        if configs.get("query_enhancement") and configs.get(
            "query_enhancement",
        ) not in ["none"]:
            e = f"Query enhancement '{configs.get('query_enhancement')}' is not supported."
            raise ValueError(e)

        if configs.get("retrieval_strategy") and configs.get(
            "retrieval_strategy",
        ) not in ["semantic"]:
            e = f"Retrieval strategy '{configs.get('retrieval_strategy')}' is not supported."
            raise ValueError(e)

        if configs.get("similarity_metric") and configs.get(
            "similarity_metric",
        ) not in ["cosine", "max_sim"]:
            e = f"Similarity metric '{configs.get('similarity_metric')}' is not supported."
            raise ValueError(e)

        if configs.get("routing_strategy") and configs.get("routing_strategy") not in [
            "none",
        ]:
            e = f"Routing strategy '{configs.get('routing_strategy')}' is not supported."
            raise ValueError(e)

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


async def main():
    """Test the MultiModalRAG implementation."""
    # 1. Setup vector database
    logger.info("Setting up Weaviate connection...")
    try:
        vector_db = VectorDB()  # Uses local Weaviate by default
        logger.info("✅ Weaviate connection successful")
    except Exception as e:
        logger.info(f"❌ Weaviate connection failed: {e}")
        logger.info("Make sure Weaviate is running locally on port 8080")
        logger.info(
            "Run: docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.25.1",
        )
        return

    # 2. Create RAG instance
    logger.info("\nCreating MultiModalRAG instance...")
    try:
        rag = MultiModalRAG(
            name="test_multimodal_rag",
            vector_db=vector_db,
            embedding_model="colqwen2",
            generation_model="colqwen2",
            configs={
                "chunking_strategy": "page_chunking",
                "query_enhancement": "none",
                "retrieval_strategy": "semantic",
                "similarity_metric": "max_sim",
                "routing_strategy": "none",
                "top_k": 3,
                "pruning_threshold": 0.0,
            },
        )
        await rag.initialize()
        logger.info("✅ MultiModalRAG instance created")
    except Exception as e:
        logger.info(f"❌ Failed to create RAG instance: {e}")
        return

    # 3. Define test image paths (will be skipped if they don't exist)
    logger.info("\nPreparing test images...")
    test_images_dir = Path(__file__).parent.parent.parent / "data/test_images"

    # Define test image paths - these will be skipped if they don't exist
    test_image_paths = []
    for i in range(1, 4):
        test_image_paths.append(test_images_dir / f"0_{i}.jpg")

    logger.info(f"Looking for {len(test_image_paths)} test images in {test_images_dir}")
    existing_images = [path for path in test_image_paths if path.exists()]
    logger.info(f"Found {len(existing_images)} existing images")

    # 4. Index the images with small batch size (if any exist)
    logger.info(f"\nIndexing {len(test_image_paths)} images...")
    try:
        await rag.index(
            test_image_paths,
            batch_size=1,
        )  # Use batch_size=1 for memory safety

        # Check if any images were actually indexed
        if existing_images:
            logger.info("✅ Images indexed successfully")
        else:
            logger.info("⚠️ No images were found to index, but collection is ready")
    except Exception as e:
        logger.info(f"❌ Failed to index images: {e}")
        return

    # 5. Test retrieval with different queries
    logger.info("\nTesting retrieval...")
    test_queries = [
        "Who is the strategic cost transformation leader at deloitte?",
        "What is the world's economy representation in percentage in Deloitte surveys?",
    ]

    for query in test_queries:
        # logger.info(f"\nQuery: '{query}'")
        try:
            results = await rag.retrieve([query], top_k=3)
            logger.info(f"Found {len(results)} results:")

            for i, result in enumerate(results):
                # logger.info(f"     Properties: {result['properties']}")
                logger.info(f"     Score: {result['score']:.4f}")

        except Exception:
            # logger.info(f"❌ Retrieval failed for query '{query}': {e}")
            pass

    # Close the Weaviate connection properly
    vector_db.client.close()

    logger.info("\n🎉 Testing completed!")


if __name__ == "__main__":
    asyncio.run(main())
