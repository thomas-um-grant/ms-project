from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from evaluation.mteb_evaluator import CustomRetrievalEvaluator
from pipeline.rags.base_rag import BaseRAG

logger = logging.getLogger(__name__)


class Evaluator:
    """
    BEIR dataset type. A BEIR dataset must contain 3 subsets.

    corpus: The dataset containing the corpus of documents. Should contain the following columns:
            - corpus-id: The column containing the id of all documents as strings.
            - doc-id: (Optional)The column containing the id of all single pages of documents as strings.
            - image: The column containing the image names as strings to be loaded as PIL images.
    queries: The dataset containing the queries. Should contain the following columns:
            - query-id: The column containing the id of all queries as strings.
            - query: The column containing all the query texts.
            - query-type: (Optional) The column containing the type of the query as strings.
    qrels: The dataset containing the query relevance scores. Should contain the following columns:
            - query-id: The column containing the query ids as strings.
            - corpus-id: The column containing the document ids as strings.
            - doc-id: (Optional) The column containing the document ids as strings.
            - score: The column containing the relevance scores as integers.

    Note: In the TREC format used here, `score` is an integer indicating the relevance of the document to the query.
    For each query i, the relevance scores are integers in the range [0, N_i], where the higher the score, the more
    relevant the document is to the given query.

    Adapted from
    Source: https://github.com/illuin-tech/vidore-benchmark/blob/main/src/vidore_benchmark/evaluation/vidore_evaluators/vidore_evaluator_beir.py
    """

    def __init__(
        self,
        rag: BaseRAG,
        corpus_id_column: str | None = None,
        doc_id_column: str | None = None,
        query_id_column: str | None = None,
        query_column: str | None = None,
        passage_column: str | None = None,
        score_column: str | None = None,
    ):
        # Dataset column names
        self.rag = rag
        self.corpus_id_column = corpus_id_column if corpus_id_column else "corpus-id"
        self.doc_id_column = doc_id_column if doc_id_column else "doc-id"
        self.query_id_column = query_id_column if query_id_column else "query-id"
        self.query_column = query_column if query_column else "query"
        self.passage_column = passage_column if passage_column else "image"
        self.score_column = score_column if score_column else "score"

        # Custom evaluator for MTEB metrics
        self.custom_evaluator = CustomRetrievalEvaluator(
            k_values=[1, 3, 5, 10, 20, 50, 100],
        )

    async def evaluate_dataset(
        self,
        ds: dict,
        k: int = 100,
        batch_size: int = 20,
        complexity: str = "v1",
    ) -> dict[str, float | None]:
        """
        Evaluate a dataset.

        Flow:
        1. Embed queries using the retriever.
        2. Retrieve documents with scores.
        3. Compute metrics comparing scores with ground truth relevance.
        """
        # Load datasets
        ds_queries = ds["queries"]
        ds_qrels = ds["qrels"]

        # Cast IDs to string to ensure compatibility with MTEB
        query_ids: list[str] = [str(elt) for elt in ds_queries[self.query_id_column]]

        # If we score a v2 dataset, we need to adjust the qrels to take into account the doc-id
        complexity_error = f"Unknown complexity level: {complexity}"
        qrels: dict[str, dict[str, int]] = defaultdict(dict)
        for qrel in ds_qrels:
            query_id = str(qrel[self.query_id_column])
            corpus_id = str(qrel[self.corpus_id_column])

            if complexity == "v1":
                qrels[query_id][corpus_id] = qrel[self.score_column]

            elif complexity == "v2":
                doc_id = str(qrel[self.doc_id_column])
                qrels[query_id][f"{corpus_id}_{doc_id}"] = qrel[self.score_column]

            else:
                raise ValueError(complexity_error)

        # Retrieve documents from Vespa using search_many
        logger.info(
            f"Retrieving documents for {len(query_ids)} queries with k={k}...",
        )

        # Batch queries to avoid overloading
        queries = list(ds_queries[self.query_column])
        all_results = {}  # Store results grouped by query index

        # Create checkpoint filename with dataset name and timestamp
        checkpoint_file = (
            Path(__file__).parent / f"checkpoint_rag_{len(queries)}_queries.json"
        )

        # Ensure the checkpoint directory exists
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        # Try to load existing checkpoint
        start_index = 0
        if checkpoint_file.exists():
            try:
                with checkpoint_file.open() as f:
                    checkpoint_data = json.load(f)
                    all_results = checkpoint_data.get("results", {})
                    start_index = checkpoint_data.get("last_processed_index", 0)
                    logger.info(f"Resuming from checkpoint at index {start_index}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Could not load checkpoint: {e}. Starting from beginning.",
                )
                all_results = {}
                start_index = 0

        for i in range(start_index, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]
            if not batch_queries:
                continue

            results = await self.rag.retrieve(
                queries=batch_queries,
                num_results=k,
            )

            # Process results and keep them grouped by query
            for query_idx, response in enumerate(results):
                global_query_idx = i + query_idx
                all_results[str(global_query_idx)] = response

            # Save checkpoint after each batch
            checkpoint_data = {
                "results": all_results,
                "last_processed_index": i + batch_size,
                "total_queries": len(queries),
                "batch_size": batch_size,
            }

            with checkpoint_file.open("w") as f:
                json.dump(
                    checkpoint_data,
                    f,
                    default=str,
                )  # default=str handles non-serializable objects

            logger.info(
                f"Processed batch {(i // batch_size) + 1}/{(len(queries) + batch_size - 1) // batch_size}, saved checkpoint",
            )

        # Clean up checkpoint file on successful completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("Evaluation completed successfully, checkpoint file removed")

        # Get the retrieved doc ids and their scores
        logger.info(
            f"Retrieved {len(all_results)} results. Computing metrics...",
        )

        # Need something like that for computing MTEB metrics:
        # results =
        # {
        # "query_0": {"doc_i": 19.125, "doc_1": 18.75, ...},
        # "query_1": {"doc_j": 17.25, "doc_1": 16.75, ...},
        # }
        scoring_results: dict[str, dict[str, float]] = {}

        for query_id in query_ids:
            hits = all_results[int(query_id)]
            retrieved_corpus = []
            for hit in hits:
                metadata, score = hit[0], hit[1]

                if complexity == "v1":
                    scoring_id = metadata["corpus-id"]
                elif complexity == "v2":
                    scoring_id = f"{metadata['corpus-id']}_{metadata['doc-id']}"
                else:
                    raise ValueError(complexity_error)

                retrieved_corpus.append(
                    {
                        "id": scoring_id,
                        "score": float(score),
                    },
                )

            scoring_results[query_id] = {
                corpus["id"]: corpus["score"] for corpus in retrieved_corpus
            }

        # Compute the MTEB metrics
        metrics = self.custom_evaluator.compute_retrieval_scores(
            qrels=qrels,
            results=scoring_results,
        )

        return metrics
