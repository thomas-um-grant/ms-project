import base64
import json
import logging
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from tqdm import tqdm
from vespa.application import VespaAsync

from src.repositories.embeddings_repository import EmbeddingsRepository
from src.repositories.vespa_repository import VespaRepository
from src.vespa_utils.vespa_schemas import (
    VespaDataPoint,
    VespaDocumentBase,
    evaluation_dataset_schema,
)

logger = logging.getLogger(__name__)


class ColPaliEvaluationBEIRDatasets(Enum):
    ARXIVQA_DATASET = "vidore/arxivqa_test_subsampled_beir"
    DOCVQA_DATASET = "vidore/docvqa_test_subsampled_beir"
    INFOVQA_DATASET = "vidore/infovqa_test_subsampled_beir"
    TABFQUAD_DATASET = "vidore/tabfquad_test_subsampled_beir"
    TATQDA_DATASET = "vidore/tatdqa_test_beir"
    GOVERNMENTAL_DATASET = "vidore/syntheticDocQA_government_reports_test_beir"


def get_base64_image(image: Image.Image) -> str:
    """
    Convert PIL image to base64 string.

    Args:
    image: PIL Image object

    Returns:
    str: Base64 encoded string of the image

    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return str(base64.b64encode(buffered.getvalue()), "utf-8")


def resize_image(image: Image.Image, max_dim: int = 2048) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.

    Args:
    image: PIL Image object
    max_dim: Maximum dimension (width or height)

    Returns:
    Image.Image: Resized image

    """
    img_width, img_height = image.size
    aspect_ratio = img_width / img_height

    if img_width > max_dim:
        new_width = max_dim
        new_height = int(new_width / aspect_ratio)
    else:
        new_width = img_width
        new_height = img_height

    if new_height > max_dim:
        new_height = max_dim
        new_width = int(new_height * aspect_ratio)

    return image.resize((new_width, new_height), Image.LANCZOS)


def binarize_tensor(tensor: torch.Tensor) -> str:
    """
    Binarize a floating-point 1-d tensor by thresholding at zero
    and packing the bits into bytes. Returns the hex str representation of the bytes.
    """
    if not tensor.is_floating_point():
        raise ValueError("Input tensor must be of floating-point type.")
    return (
        np.packbits(np.where(tensor > 0, 1, 0), axis=0).astype(np.int8).tobytes().hex()
    )


def load_vidore_dataset(
    dataset_name,
    head: int = None,
    name: str = "corpus",
) -> Dataset:
    """
    Loads a specified dataset from the Vidore BEIR collection.

    Args:
            dataset_name (str): The name of the dataset to load. Must be a member of ColPaliEvaluationBEIRDatasets.
            head (int, optional): If provided, limits the dataset to the first `head` examples.

    Returns:
            Dataset: The loaded Hugging Face dataset split.

    Raises:
            ValueError: If the provided dataset name is not supported.

    """
    if dataset_name in ColPaliEvaluationBEIRDatasets:
        return load_dataset(
            dataset_name,
            name=name,
            split=f"test{f'[:{head}]' if head else ''}",
        )
    raise ValueError(f"The dataset ({dataset_name}) is not supported.")


def load_custom_dataset(
    dataset_name: str,
    head: int = None,
    name: str = "corpus",
) -> Dataset:
    """
    Loads a custom dataset from the dataset folder.

    Args:
            dataset_name (str): The name of the dataset to load.
            head (int, optional): If provided, limits the dataset to the first `head` examples.

    Returns:
            Dataset: The loaded dataset dict split.

    """
    dataset_path = (
        Path(__file__).parent.parent / f"dataset/data/{dataset_name}/dataset.json"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"The dataset ({dataset_name}) does not exist in the expected path: {dataset_path}",
        )

    # Load the dataset from the specified path
    with dataset_path.open("r") as f:
        dataset = json.load(f)

        corpus_columns = ["id", "image", "doc-id", "corpus-id"]
        queries_columns = ["query-id", "query", "query-type"]
        qrels_columns = ["corpus-id", "query-id", "doc-id", "answer", "score"]
        # Convert list of lists into column-wise dicts
        corpus_data = {
            col: [row[col] for row in dataset["corpus"]] for col in corpus_columns
        }
        queries_data = {
            col: [row[col] for row in dataset["queries"]] for col in queries_columns
        }
        qrels_data = {
            col: [row[col] for row in dataset["qrels"]] for col in qrels_columns
        }
        dataset = DatasetDict(
            {
                "corpus": Dataset.from_dict(corpus_data),
                "queries": Dataset.from_dict(queries_data),
                "qrels": Dataset.from_dict(qrels_data),
            },
        )

    return dataset[name][:head] if head else dataset[name]


async def feed_dataset_to_vespa(
    vespa_app: Any,
    embeddings_repo: EmbeddingsRepository,
    dataset_name: str,
    ds: Dataset,
    batch_size: int = 8,
    concurrency: int = 4,
):
    """
    Feed a dataset to Vespa in batches using the EmbeddingsRepository and VespaRepository APIs.

    Args:
        vespa_app: Vespa application instance (should have a .repo attribute or be a VespaService)
        embeddings_repo: EmbeddingsRepository instance for embedding images
        dataset_name: Name of the dataset
        ds: Dataset object (dict-like, with 'image', 'corpus-id', 'doc-id', ...)
        batch_size: Batch size for feeding/embedding
        concurrency: Concurrency level for embedding
    Returns:
        bool: True if all batches succeeded, False otherwise

    """
    if embeddings_repo is None or vespa_app is None:
        raise ValueError("embeddings_repo and vespa_app must be provided.")

    async with VespaAsync(vespa_app) as session:
        vespa_repo = VespaRepository(session=session, vespa_app=vespa_app)

        # Use the EvaluationDatasetDataPointFields model
        images = ds["image"]
        corpus_ids = ds["corpus-id"]
        doc_ids = (
            ds["doc-id"]
            if "doc-id" in ds.column_names
            else ["_" for _ in range(len(ds["image"]))]
        )

        batch_success = True
        for batch_start in tqdm(range(0, len(images), batch_size)):
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            batch_corpus_ids = corpus_ids[batch_start:batch_end]
            batch_doc_ids = doc_ids[batch_start:batch_end]

            # Load images if they are file paths
            loaded_images = []
            for img in batch_images:
                if isinstance(img, str):
                    # Assume it's a file path
                    loaded_images.append(
                        Image.open(
                            Path(__file__).parent.parent
                            / f"dataset/data/{dataset_name}/corpuses/{img}",
                        ),
                    )
                else:
                    loaded_images.append(img)

            # Embed images
            try:
                embeddings = await embeddings_repo.embed_images(
                    loaded_images,
                    concurrency=concurrency,
                )
            except Exception as e:
                logger.error(
                    f"Failed to embed images for batch {batch_start}-{batch_end}: {e}",
                )
                batch_success = False
                continue

            # Format as VespaDataPoint
            data_points = []
            for _, (corpus_id, doc_id, image, embedding) in enumerate(
                zip(
                    batch_corpus_ids,
                    batch_doc_ids,
                    loaded_images,
                    embeddings,
                    strict=True,
                ),
            ):
                embedding_dict = {
                    str(j): emb.tolist() for j, emb in enumerate(embedding)
                }
                embedding_binary_dict = {
                    str(j): binarize_tensor(emb) for j, emb in enumerate(embedding)
                }
                file_id = f"{dataset_name}{f'_{doc_id}' if doc_id and doc_id.strip() != '_' else ''}_{corpus_id}"
                resized_image = resize_image(image)
                base_64_image = get_base64_image(resized_image)

                vespa_base_model: type[VespaDocumentBase] = evaluation_dataset_schema
                vespa_field_data_point_fields = vespa_base_model.get_hit_model()
                vespa_fields = vespa_field_data_point_fields.model_validate(
                    {
                        "id": file_id,
                        "dataset_name": dataset_name,
                        "corpus_id": corpus_id,
                        "doc_id": doc_id,
                        "image": base_64_image,
                        "embedding": embedding_dict,
                        "binary_embedding": embedding_binary_dict,
                    },
                )

                data_point = VespaDataPoint(id=file_id, fields=vespa_fields)
                data_points.append(data_point)

            # Feed to Vespa
            try:
                # TODO: Need to fix feed_iterable first before sending batches. It is throwing.
                # result = await vespa_app.feed_iterable(
                #     "evaluation_dataset", data_points
                # )
                # if not result:
                #     logger.error(
                #         f"Vespa feed_iterable failed for batch {batch_start}-{batch_end}"
                #     )
                #     batch_success = False
                # else:
                #     logger.info(
                #         f"Vespa feed_iterable succeeded for batch {batch_start}-{batch_end}"
                #     )
                #     batch_success = True

                for vespa_data_point in data_points:
                    response, _ = await vespa_repo.feed_data_point(
                        data_point=vespa_data_point,
                        schema_name="evaluation_dataset",
                    )

                    batch_success = True

                    if not response:
                        logger.error(
                            f"Vespa feed_iterable failed for batch {batch_start}-{batch_end}",
                        )
                        batch_success = False

            except Exception as e:
                logger.error(
                    f"Exception during Vespa feed_iterable for batch {batch_start}-{batch_end}: {e}",
                )
                batch_success = False

    return batch_success
