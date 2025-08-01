from __future__ import annotations

import logging
from collections import defaultdict

import httpx
from repositories.vespa_repository import VespaRepository
from schemas.vespa_schema import QueryRequest
from services.vespa_service import VespaService
from vespa.application import VespaAsync
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever

from retrieval_evaluation.evaluator.base_vidore_evaluator import BaseViDoReEvaluator
from retrieval_evaluation.utils.vespa_utils import connect_existing_vespa

logger = logging.getLogger(__name__)


class CustomEvaluator(BaseViDoReEvaluator):
    """
    Evaluator for the ViDoRe benchmark for datasets with a BEIR format.
    Uses the CustomVisionRetriever for embeddings and direct scoring, and Vespa for document storage and retrieval.

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
        vision_retriever: BaseVisionRetriever,
        corpus_id_column: str | None = None,
        query_id_column: str | None = None,
        query_column: str | None = None,
        passage_column: str | None = None,
        score_column: str | None = None,
    ):
        super().__init__(vision_retriever=vision_retriever)

        # Dataset column names
        self.corpus_id_column = corpus_id_column if corpus_id_column else "corpus-id"
        self.query_id_column = query_id_column if query_id_column else "query-id"
        self.query_column = query_column if query_column else "query"
        if passage_column:
            self.passage_column = passage_column
        else:
            self.passage_column = (
                "image"
                if self.vision_retriever.use_visual_embedding
                else "text_description"
            )
        self.score_column = score_column if score_column else "score"

    async def evaluate_dataset(
        self,
        ds: dict,
        batch_query: int,
        batch_passage: int,
        batch_score: int | None = None,
        dataloader_prebatch_query: int | None = None,
        dataloader_prebatch_passage: int | None = None,
        k: int = 10,  # Number of documents to retrieve
        ds_name: str = None,  # Dataset name for DB filtering
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

        # Get the embeddings for the queries
        query_embeddings = await self._get_query_embeddings(
            ds=ds_queries,
            query_column=self.query_column,
            batch_query=batch_query,
            dataloader_prebatch_size=dataloader_prebatch_query,
        )

        # Prepare Query Requests for Vespa
        query_requests = [
            QueryRequest(
                text=query_text,
                embeddings=query_embedding,
                k=k,
                document_filters=[("dataset_name", ds_name)],
            )
            for query_text, query_embedding in zip(
                ds_queries[self.query_column],
                query_embeddings,
                strict=False,
            )
        ]

        # Connect to Vespa
        vespa_app = connect_existing_vespa(
            "dbevals",
            tenant_name="dev",
            instance_name="default",
        )

        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        async with VespaAsync(
            vespa_app,
            connections=1,
            limits=limits,
            timeout=30,
        ) as vespa_session:
            vespa_repo = VespaRepository(vespa_session)
            vespa_service = VespaService(vespa_repo)
            logger.info("Initialised Vespa service")

            # Retrieve documents from Vespa using search_many
            vespa_results = await vespa_service.search_many(
                query_requests,
                variant="evaluation",
                schema="EvaluationDatasetFields",
            )

            # Get the retrieved document IDs and their Vespa scores
            query_to_hits = {}
            for query_id, result in zip(
                ds_queries[self.query_id_column],
                vespa_results,
                strict=False,
            ):
                query_to_hits[str(query_id)] = result.hits

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
                    "id": str(hit.fields.corpus_id),
                    "score": float(hit.relevance),
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
