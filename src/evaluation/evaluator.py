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

    # --- helpers added ---
    def _build_qrels(self, ds_qrels, complexity: str) -> dict[str, dict[str, int]]:
        """
        Build qrels dict respecting complexity.
        v1: collapse all pages of a corpus; relevance becomes max(score).
        v2: use corpus-id_doc-id composite keys with original score.
        """
        qrels: dict[str, dict[str, int]] = defaultdict(dict)
        if complexity == "v1":
            # temp structure {query_id: {corpus_id: max_score}}
            temp: dict[str, dict[str, int]] = defaultdict(dict)
            for qrel in ds_qrels:
                try:
                    qid = str(qrel[self.query_id_column])
                    cid = str(qrel[self.corpus_id_column])
                    rel = int(qrel[self.score_column])
                except Exception:  # pragma: no cover
                    continue
                prev = temp[qid].get(cid, 0)
                if rel > prev:
                    temp[qid][cid] = rel
            qrels.update(temp)
        elif complexity == "v2":
            for qrel in ds_qrels:
                try:
                    qid = str(qrel[self.query_id_column])
                    cid = str(qrel[self.corpus_id_column])
                    did = str(qrel[self.doc_id_column])
                    rel = int(qrel[self.score_column])
                except Exception:  # pragma: no cover
                    continue
                qrels[qid][f"{cid}_{did}"] = rel
        else:
            raise ValueError(f"Unknown complexity level: {complexity}")
        return qrels

    def _process_retrieval_results(
        self,
        all_results: dict[str, list[tuple[dict, float]]],
        query_ids: list[str],
        complexity: str,
    ) -> dict[str, dict[str, float]]:
        """
        Convert raw retrieval outputs into scoring dict expected by pytrec_eval.
        v1: keep best score per corpus-id.
        v2: use composite corpus-id_doc-id keys with their scores.
        """
        scoring_results: dict[str, dict[str, float]] = {}
        complexity_error = f"Unknown complexity level: {complexity}"
        for qid in query_ids:
            hits = all_results.get(qid, [])
            per_query: dict[str, float] = {}
            for hit in hits:
                try:
                    metadata, score = hit[0], float(hit[1])
                except Exception:  # pragma: no cover
                    continue
                if complexity == "v1":
                    key = metadata["corpus-id"]
                elif complexity == "v2":
                    key = f"{metadata['corpus-id']}_{metadata['doc-id']}"
                else:
                    raise ValueError(complexity_error)
                # keep max score for stability
                if score > per_query.get(key, float("-inf")):
                    per_query[key] = score
            scoring_results[qid] = per_query
        return scoring_results

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

        # Build qrels respecting complexity
        qrels = self._build_qrels(ds_qrels, complexity)

        logger.info(
            f"Retrieving documents for {len(query_ids)} queries with k={k} (complexity={complexity})...",
        )

        # Batch queries to avoid overloading
        queries = list(ds_queries[self.query_column])
        all_results: dict[str, list[tuple[dict, float]]] = {}
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
            except (json.JSONDecodeError, KeyError) as e:  # pragma: no cover
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
                top_k=k,
            )
            for query_idx, response in enumerate(results):
                global_query_idx = i + query_idx
                actual_query_id = query_ids[global_query_idx]
                all_results[actual_query_id] = response
            checkpoint_data = {
                "results": all_results,
                "last_processed_index": i + batch_size,
                "total_queries": len(queries),
                "batch_size": batch_size,
            }
            with checkpoint_file.open("w") as f:
                json.dump(checkpoint_data, f, default=str)
            logger.info(
                f"Processed batch {(i // batch_size) + 1}/{(len(queries) + batch_size - 1) // batch_size}, saved checkpoint",
            )

        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("Evaluation completed successfully, checkpoint file removed")

        logger.info(
            f"Retrieved {len(all_results)} query result sets. Computing metrics...",
        )

        scoring_results = self._process_retrieval_results(
            all_results,
            query_ids,
            complexity,
        )

        metrics = self.custom_evaluator.compute_retrieval_scores(
            qrels=qrels,
            results=scoring_results,
        )
        return metrics
