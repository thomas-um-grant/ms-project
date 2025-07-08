import hashlib
import logging
import math
from enum import Enum
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from vidore_benchmark.utils.iter_utils import batched

from evaluation.retriever.sherpa_retriever import SherpaVisionRetriever
from evaluation.utils.vespa_utils import (
    get_base64_image,
    resize_image,
)

logger = logging.getLogger(__name__)


class ColPaliEvaluationBEIRDatasets(Enum):
    ARXIVQA_DATASET = "vidore/arxivqa_test_subsampled_beir"
    DOCVQA_DATASET = "vidore/docvqa_test_subsampled_beir"
    INFOVQA_DATASET = "vidore/infovqa_test_subsampled_beir"
    TABFQUAD_DATASET = "vidore/tabfquad_test_subsampled_beir"
    TATQDA_DATASET = "vidore/tatdqa_test_beir"
    ECONOMICS_DATASET = (
        "vidore/synthetics_economics_macro_economy_2024_filtered_v1.0_multilingual"
    )
    RESTAURANTS_DATASET = "vidore/synthetic_rse_restaurant_filtered_v1.0_multilingual"
    BIOMEDICAL_DATASET = (
        "vidore/synthetic_mit_biomedical_tissue_interactions_unfiltered_multilingual"
    )


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


def load_vidore_dataset(dataset_name, head: int = None):
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
            dataset_name, name="corpus", split=f"test{f'[:{head}]' if head else ''}"
        )
    else:
        raise ValueError(f"The dataset ({dataset_name}) is not supported.")


async def prepare_dataset(
    dataset_name: str,
    dataset: Dataset,
    retriever: SherpaVisionRetriever,
    batch_size: int = 4,
    dataloader_prebatch_size: int | None = None,
) -> list[dict]:
    """
    Prepares a dataset for Vespa ingestion by resizing images, computing embeddings, and constructing document records.

    Args:
            dataset_name (str): Name of the dataset being processed.
            dataset (Dataset): Hugging Face Dataset containing image and metadata fields.
            retriever (SherpaVisionRetriever): The retriever used to generate image embeddings.

    Returns:
            List[dict]: A list of document records formatted for Vespa indexing.
    """
    data_feed = []

    print("Generating Vespa data feed")

    # Get all images from dataset
    images = dataset["image"]
    corpus_ids = dataset["corpus-id"]
    doc_ids = (
        dataset["doc-id"] if "doc-id" in dataset else ["_" for _ in range(len(dataset))]
    )

    # Generate embeddings for all images
    embeddings: list[torch.Tensor] = []

    if dataloader_prebatch_size is None:
        dataloader_prebatch_size = 10 * batch_size
    if dataloader_prebatch_size < batch_size:
        logger.warning(
            f"`dataloader_prebatch_size` ({dataloader_prebatch_size}) is smaller than `batch_passage` "
            f"({batch_size}). Setting the pre-batch size to the passager batch size."
        )
        dataloader_prebatch_size = batch_size

    # Embed in batch so not all images are loaded to memory at once
    for ds_batch in tqdm(
        batched(dataset, n=dataloader_prebatch_size),
        desc="Dataloader pre-batching for passages",
        total=math.ceil(len(dataset) / (dataloader_prebatch_size)),
        leave=False,
    ):
        passages: list[Any] = [batch["image"] for batch in ds_batch]

        batch_embedding_passages = await retriever.forward_passages(
            passages=passages,
            batch_size=batch_size,
        )

        if isinstance(batch_embedding_passages, torch.Tensor):
            batch_embedding_passages = list(
                torch.unbind(batch_embedding_passages.to("cpu"))
            )
            embeddings.extend(batch_embedding_passages)
        else:
            for embedding_passage in batch_embedding_passages:
                # TODO: Had to extract embedding_passage from a list of len(1)
                # Need to check if this is a bug
                embeddings.append(torch.as_tensor(embedding_passage[0]).to("cpu"))

    for corpus_id, doc_id, image, embedding in tqdm(
        zip(
            corpus_ids,
            doc_ids,
            images,
            embeddings,
            strict=True,
        ),
    ):
        embedding_dict = dict()
        embedding_binary_dict = dict()
        for j, emb in enumerate(embedding):
            logger.info(f"Embedding {j}: ({len(emb)})")
            embedding_dict[j] = emb.tolist()
            embedding_binary_dict[j] = binarize_tensor(emb)

        # Create document ID
        id_hash = hashlib.md5(f"{corpus_id}_{doc_id}".encode()).hexdigest()

        # Resize image and convert to base64
        resized_image = resize_image(image)
        base_64_image = get_base64_image(resized_image)

        # Create document fields
        fields = {
            "id": f"{dataset_name}{f'_{doc_id}' if doc_id else ''}_{corpus_id}",
            "dataset_name": dataset_name,
            "corpus_id": corpus_id,
            "doc_id": doc_id,
            "image": base_64_image,
            "embedding": embedding_dict,
            "binary_embedding": embedding_binary_dict,
        }

        data_feed.append(
            {
                "id": id_hash,
                "fields": fields,
            }
        )

    return data_feed
