import torch


def score_multi_vector(
    qs: torch.Tensor | list[torch.Tensor],
    ps: torch.Tensor | list[torch.Tensor],
    batch_size: int = 128,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
    query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
    image of a document page.

    Because the embedding tensors are multi-vector and can thus have different shapes, they
    should be fed as:
    (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
    (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

    Args:
            qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            device (`Union[str, torch.device]`, *optional*): Device to use for computation.

    Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
    """

    if len(qs) == 0:
        raise ValueError("No queries provided")
    if len(ps) == 0:
        raise ValueError("No passages provided")

    scores_list: list[torch.Tensor] = []

    for i in range(0, len(qs), batch_size):
        scores_batch = []
        qs_batch = torch.nn.utils.rnn.pad_sequence(
            qs[i : i + batch_size], batch_first=True, padding_value=0
        ).to(device)
        for j in range(0, len(ps), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(
                ps[j : j + batch_size], batch_first=True, padding_value=0
            ).to(device)
            scores_batch.append(
                torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                .max(dim=3)[0]
                .sum(dim=2)
            )
        scores_batch = torch.cat(scores_batch, dim=1).cpu()
        scores_list.append(scores_batch)

    scores = torch.cat(scores_list, dim=0)
    assert scores.shape[0] == len(qs), (
        f"Expected {len(qs)} scores, got {scores.shape[0]}"
    )

    scores = scores.to(torch.float32)
    return scores
