import base64
from enum import Enum
from io import BytesIO

import numpy as np
import torch
from PIL import Image


class ColPaliEvaluationBEIRDatasets(Enum):
    ARXIVQA_DATASET = "vidore/arxivqa_test_subsampled_beir"
    DOCVQA_DATASET = "vidore/docvqa_test_subsampled_beir"
    INFOVQA_DATASET = "vidore/infovqa_test_subsampled_beir"
    TABFQUAD_DATASET = "vidore/tabfquad_test_subsampled_beir"
    TATQDA_DATASET = "vidore/tatdqa_test_beir"
    GOVERNMENTAL_DATASET = "vidore/syntheticDocQA_government_reports_test_beir"


def get_torch_device(device: str = "auto") -> str:
    """
    Get the appropriate torch device based on the input string.

    Args:
        device (str): Device type, can be 'auto', 'cuda', 'mps', or 'cpu'.

    Returns:
        str: The selected device type.

    """
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


def score_max_sim(
    qs: torch.Tensor | list[torch.Tensor],
    ps: torch.Tensor | list[torch.Tensor],
    batch_size: int = 128,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Compute the late-interaction/MaxSim score (ColBERT-like).

    Because the embedding tensors are multi-vector and can thus have different shapes, they
    should be fed as:
    (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
    (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
        obtained by padding the list of tensors.

    Args:
        qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
        ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
        batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
        device (`Union[str, torch.device]`, *optional*): Device to use for computation. If not
            provided, uses `get_torch_device("auto")`.

    Returns:
        `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
        tensor is saved on the "cpu" device.

    """
    device = device or get_torch_device("auto")

    if len(qs) == 0:
        error_msg = "No queries provided"
        raise ValueError(error_msg)
    if len(ps) == 0:
        error_msg = "No passages provided"
        raise ValueError(error_msg)

    scores_list: list[torch.Tensor] = []

    for i in range(0, len(qs), batch_size):
        scores_batch = []
        qs_batch = torch.nn.utils.rnn.pad_sequence(
            qs[i : i + batch_size],
            batch_first=True,
            padding_value=0,
        ).to(
            device,
        )
        for j in range(0, len(ps), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(
                ps[j : j + batch_size],
                batch_first=True,
                padding_value=0,
            ).to(device)
            scores_batch.append(
                torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                .max(dim=3)[0]
                .sum(dim=2),
            )
        scores_batch_tensor = torch.cat(scores_batch, dim=1).cpu()
        scores_list.append(scores_batch_tensor)

    scores = torch.cat(scores_list, dim=0)

    if scores.shape[0] != len(qs):
        error_msg = f"Expected {len(qs)} scores, got {scores.shape[0]}"
        raise ValueError(error_msg)

    scores = scores.to(torch.float32)
    return scores


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


def binarize_tensor(tensor: torch.Tensor) -> str:
    """Binarize a floating-point 1-d tensor by thresholding at zero and packing the bits into bytes. Returns the hex str representation of the bytes."""
    if not tensor.is_floating_point():
        msg = "Input tensor must be of floating-point type."
        raise ValueError(msg)
    return (
        np.packbits(np.where(tensor > 0, 1, 0), axis=0).astype(np.int8).tobytes().hex()
    )


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

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
