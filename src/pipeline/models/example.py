# colqwen_weaviate_v4.py

import logging
import os
import pathlib
import uuid
from pathlib import Path
from typing import Any

import torch
import weaviate
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image
from weaviate.classes.config import Configure, DataType, Property

# Configuration
MODEL_NAME = "vidore/colqwen2-v1.0"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
COLLECTION_NAME = "DocPage"
BATCH_CHUNK = 50
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
MAX_IMAGE_SIZE = (448, 448)  # Resize images to reduce vector size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model init
device = torch.device("cpu")  # Using CPU to avoid MPS issues with this model
logger.info(f"Using device: {device}")
processor = ColQwen2Processor.from_pretrained(MODEL_NAME)
model = ColQwen2.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
model.eval()


def embed_image(img: Image.Image) -> list[float]:
    # Resize image to reduce vector size
    img = img.resize(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

    batch_images = processor.process_images([img]).to(device)
    with torch.no_grad():
        out = model(**batch_images)
    # out is shape [batch_size, seq_len, embedding_dim]
    multi_vector = out[0].cpu().float()  # shape [seq_len, embedding_dim]

    # Use mean pooling to create a fixed-size vector
    pooled_vector = multi_vector.mean(dim=0).tolist()  # shape [embedding_dim]

    logger.info(f"Generated embedding with {len(pooled_vector)} dimensions")
    return pooled_vector


def embed_text(query: str) -> list[float]:
    batch_queries = processor.process_queries([query]).to(device)
    with torch.no_grad():
        out = model(**batch_queries)
    # out is shape [batch_size, seq_len, embedding_dim]
    multi_vector = out[0].cpu().float()  # shape [seq_len, embedding_dim]

    # Use mean pooling to create a fixed-size vector
    pooled_vector = multi_vector.mean(dim=0).tolist()  # shape [embedding_dim]

    return pooled_vector


def ensure_collection(client: weaviate.WeaviateClient) -> Any:
    # Delete collection if it exists to recreate with correct config
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
        logger.info(f"Deleted existing collection: {COLLECTION_NAME}")

    # Create collection with no vectorizer (we'll provide our own embeddings)
    client.collections.create(
        name=COLLECTION_NAME,
        properties=[
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="page_num", data_type=DataType.INT),
            Property(name="text_snippet", data_type=DataType.TEXT),
        ],
        vectorizer_config=Configure.Vectorizer.none(),
    )
    logger.info(f"Created collection: {COLLECTION_NAME}")
    return client.collections.get(COLLECTION_NAME)


def index_folder(client: weaviate.WeaviateClient, folder: str, doc_id: str) -> None:
    coll = ensure_collection(client)
    files = sorted(
        [
            f
            for f in pathlib.Path(folder).iterdir()
            if str(f).lower().endswith(IMAGE_EXTENSIONS)
        ],
    )

    with coll.batch.dynamic() as batch:
        for fname in files:
            page_num = (
                int(str(fname).split(".")[0].split("_")[-1]) if "_" in str(fname) else 0
            )
            img = Image.open(folder / fname).convert("RGB")
            embedding = embed_image(img)
            batch.add_object(
                properties={"doc_id": doc_id, "page_num": page_num, "text_snippet": ""},
                vector=embedding,
                uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}-{fname}")),
            )

    logger.info(f"Indexed {len(files)} images into {COLLECTION_NAME}")


def search_by_text(
    client: weaviate.WeaviateClient,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    coll = client.collections.get(COLLECTION_NAME)
    embedding = embed_text(query)
    resp = coll.query.near_vector(
        near_vector=embedding,
        limit=top_k,
        include_vector=False,
    )
    results = []
    for obj in resp.objects:
        props = obj.properties or {}
        # Get distance from metadata if available
        distance = None
        if hasattr(obj, "metadata") and obj.metadata:
            distance = obj.metadata.distance
        results.append(
            {
                "doc_id": props.get("doc_id"),
                "page_num": props.get("page_num"),
                "text_snippet": props.get("text_snippet"),
                "distance": distance,
            },
        )
    return results


def main() -> None:
    client = weaviate.connect_to_local()
    try:
        folder = Path(__file__).parent.parent.parent / "data/test_images"
        doc_id = "local_doc"
        index_folder(client, str(folder), doc_id)
        results = search_by_text(client, "AI regulations in the document", top_k=3)
        for r in results:
            distance_str = (
                f"{r['distance']:.3f}" if r["distance"] is not None else "N/A"
            )
            print(f"Doc={r['doc_id']} Page={r['page_num']} Dist={distance_str}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
