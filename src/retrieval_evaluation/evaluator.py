from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from retrieval_evaluation.evaluation_retriever import (
    EvaluationRetriever,
)
from retrieval_evaluation.mteb_evaluator import CustomRetrievalEvaluator

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for the ViDoRe benchmark for datasets with a BEIR format.
    Uses the EvaluationRetriever for embeddings and direct scoring, and Vespa for document storage and retrieval.

    BEIR dataset type. A BEIR dataset must contain 3 subsets:
    corpus: The dataset containing the corpus of documents. Should contain the following columns:
            - corpus-id: The column containing the document IDs as integers.
            - image: The column containing the image data (PIL format).
    queries: The dataset containing the queries. Should contain the following columns:
            - query-id: The column containing the query IDs as integers.
            - query: The column containing the query text.
    qrels: The dataset containing the query relevance scores (TREC format). Should contain the following columns:
            - query-id: The column containing the query IDs as integers.
            - corpus-id: The column containing the document IDs as integers.
            - score: The column containing the relevance scores as integers.

    Note: In the TREC format used here, `score` is an integer indicating the relevance of the document to the query.
    For each query i, the relevance scores are integers in the range [0, N_i], where the higher the score, the more
    relevant the document is to the given query.

    Source: https://github.com/illuin-tech/vidore-benchmark/blob/main/src/vidore_benchmark/evaluation/vidore_evaluators/vidore_evaluator_beir.py
    """

    def __init__(
        self,
        vision_retriever: EvaluationRetriever,
        corpus_id_column: str | None = None,
        query_id_column: str | None = None,
        query_column: str | None = None,
        passage_column: str | None = None,
        score_column: str | None = None,
    ):
        # Dataset column names
        self.vision_retriever = vision_retriever
        self.corpus_id_column = corpus_id_column if corpus_id_column else "corpus-id"
        self.query_id_column = query_id_column if query_id_column else "query-id"
        self.query_column = query_column if query_column else "query"
        self.passage_column = passage_column if passage_column else "image"
        self.score_column = score_column if score_column else "score"

    async def evaluate_dataset(
        self,
        ds: dict,
        k: int = 100,  # Number of documents to retrieve
        ds_name: str = None,  # Dataset name for DB filtering
        batch_size: int = 20,
        **kwargs,
    ) -> dict[str, dict[str, float | None]]:
        """
        Evaluate a dataset using Vespa retrieval and scoring.

        Flow:
        1. Embed queries using the retriever.
        2. Retrieve documents from Vespa with scores.
        3. Compute metrics comparing Vespa scores with ground truth relevance.
        """
        # Load datasets
        ds_queries = ds["queries"]
        ds_qrels = ds["qrels"]

        # Cast IDs to string to ensure compatibility with MTEB
        query_ids: list[str] = [str(elt) for elt in ds_queries[self.query_id_column]]

        qrels: dict[str, dict[str, int]] = defaultdict(dict)
        for qrel in ds_qrels:
            query_id = str(qrel[self.query_id_column])
            corpus_id = str(qrel[self.corpus_id_column])
            qrels[query_id][corpus_id] = qrel[self.score_column]

        # Retrieve documents from Vespa using search_many
        logger.info(
            f"Retrieving documents for {len(query_ids)} queries from Vespa for dataset '{ds_name}' with k={k}...",
        )

        # Batch queries to avoid overloading Vespa
        queries = list(ds_queries[self.query_column])
        all_vespa_results = {}  # Store results grouped by query index

        # Create checkpoint filename with dataset name and timestamp
        checkpoint_file = (
            Path(__file__).parent / f"checkpoint_{ds_name}_{len(queries)}_queries.json"
        )

        # Ensure the checkpoint directory exists
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        # Try to load existing checkpoint
        start_index = 0
        if checkpoint_file.exists():
            try:
                with checkpoint_file.open() as f:
                    checkpoint_data = json.load(f)
                    all_vespa_results = checkpoint_data.get("results", {})
                    start_index = checkpoint_data.get("last_processed_index", 0)
                    logger.info(f"Resuming from checkpoint at index {start_index}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Could not load checkpoint: {e}. Starting from beginning.",
                )
                all_vespa_results = {}
                start_index = 0

        for i in range(start_index, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]
            if not batch_queries:
                continue

            vespa_results = await self.vision_retriever.retrieve(
                queries=batch_queries,
                num_results=k,
                variant="ann_float",
                document_filters=[("dataset_name", ds_name)],
            )

            # Process results and keep them grouped by query
            for query_idx, response in enumerate(vespa_results):
                global_query_idx = i + query_idx
                hits = response.hits

                filtered_hits = []
                for hit in hits:
                    hit.relevance = float(hit.relevance) if hit.relevance else 0.0

                    filtered_hit = {
                        "id": hit.id,
                        "fields": {
                            "id": hit.fields.id,
                            "dataset_name": hit.fields.dataset_name,
                            "corpus_id": hit.fields.corpus_id,
                            "doc_id": hit.fields.doc_id,
                        },
                        "relevance": hit.relevance,
                    }
                    filtered_hits.append(filtered_hit)

                all_vespa_results[str(global_query_idx)] = filtered_hits

            # Save checkpoint after each batch
            checkpoint_data = {
                "results": all_vespa_results,
                "last_processed_index": i + batch_size,
                "total_queries": len(queries),
                "dataset_name": ds_name,
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

        # Get the retrieved document IDs and their Vespa scores
        logger.info(
            f"Retrieved {len(all_vespa_results)} results from Vespa for dataset '{ds_name}'. Computing metrics...",
        )

        # Map query IDs to their hits using the stored query index
        query_to_hits = {}
        for i, query_id in enumerate(ds_queries[self.query_id_column]):
            query_hits = all_vespa_results.get(str(i), [])
            query_to_hits[str(query_id)] = query_hits

        # Compute scores for retrieved documents using different methods

        # Need something like that for computing MTEB metrics:
        # results =
        # {
        # "query_0": {"doc_i": 19.125, "doc_1": 18.75, ...},
        # "query_1": {"doc_j": 17.25, "doc_1": 16.75, ...},
        # }
        results: dict[str, dict[str, float]] = {}

        for query_id in query_ids:
            hits = query_to_hits[query_id]
            retrieved_corpus = [
                {
                    "id": str(hit["fields"]["corpus_id"]),
                    "score": float(hit["relevance"]),
                }
                for hit in hits
            ]

            results[query_id] = {
                corpus["id"]: corpus["score"] for corpus in retrieved_corpus
            }

            for i in range(len(query_ids)):
                if str(i) not in results[query_id]:
                    results[query_id][str(i)] = 0

        # Compute the MTEB metrics
        metrics = self.compute_retrieval_scores(
            qrels=qrels,
            results=results,
            ignore_identical_ids=False,
        )

        return metrics

    @staticmethod
    def compute_retrieval_scores(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        ignore_identical_ids: bool = False,
        **kwargs,
    ) -> dict[str, float | None]:
        """
        Compute the MTEB retrieval metrics (NDCG, MAP, Recall, Precision, NDCG, MRR, NDCG, and NDCG).

        Args:
                qrels: A dictionary containing the degree of relevance between queries and documents,
                        following the BEIR convention (0: irrelevant, 1: relevant).
                results: A dictionary containing the retrieval results, i.e. the retrieval
                        scores for each document for each query.
                        Example input:
                        ```python
                        {
                                "query_0": {"doc_i": 19.125, "doc_1": 18.75, ...},
                                "query_1": {"doc_j": 17.25, "doc_1": 16.75, ...},
                                ...
                        }
                        ```
                ignore_identical_ids: Whether to ignore identical IDs in the results, e.g. set to `True` if the
                        queries and documents have overlapping IDs.
                **kwargs: Additional keyword arguments.

        """
        mteb_evaluator = CustomRetrievalEvaluator()

        ndcg, _map, recall, precision, naucs = mteb_evaluator.evaluate(
            qrels=qrels,
            results=results,
            k_values=mteb_evaluator.k_values,
            ignore_identical_ids=ignore_identical_ids,
        )

        mrr = mteb_evaluator.evaluate_custom(
            qrels,
            results,
            mteb_evaluator.k_values,
            "mrr",
        )

        scores: dict[str, float | None] = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr[0].items()},
            **{f"naucs_at_{k.split('@')[1]}": v for (k, v) in naucs.items()},
        }

        return scores
