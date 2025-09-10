# Adapted from Source: https://github.com/illuin-tech/vidore-benchmark/blob/main/src/vidore_benchmark/evaluation/eval_utils.py#L19

import logging

import pytrec_eval
from mteb.evaluation.evaluators.RetrievalEvaluator import RetrievalEvaluator
from mteb.evaluation.evaluators.utils import (
    hole,
    mrr,
    recall_cap,
    top_k_accuracy,
)

logger = logging.getLogger(__name__)


class CustomRetrievalEvaluator:
    """Wrapper class for the MTEB retrieval evaluator."""

    def __init__(self, k_values: list[int] | None = None):
        if k_values is None:
            k_values = [1, 3, 5, 10, 20, 50, 100]
        self.k_values = k_values

    def compute_retrieval_scores(
        self,
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
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

        """
        ndcg, _map, recall, precision, naucs = self._evaluate(
            qrels=qrels,
            results=results,
            k_values=self.k_values,
        )

        mrr = self._evaluate_custom(
            qrels,
            results,
            self.k_values,
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

    @staticmethod
    def _evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ]:
        all_ndcgs: dict[str, list[float]] = {}
        all_aps: dict[str, list[float]] = {}
        all_recalls: dict[str, list[float]] = {}
        all_precisions: dict[str, list[float]] = {}

        for k in k_values:
            all_ndcgs[f"NDCG@{k}"] = []
            all_aps[f"MAP@{k}"] = []
            all_recalls[f"Recall@{k}"] = []
            all_precisions[f"P@{k}"] = []

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels,
            {map_string, ndcg_string, recall_string, precision_string},
        )
        scores = evaluator.evaluate(results)

        for query_id in scores:
            for k in k_values:
                all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
                all_aps[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
                all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
                all_precisions[f"P@{k}"].append(scores[query_id]["P_" + str(k)])

        # Create separate dictionaries for averaged results
        ndcg: dict[str, float] = {}
        _map: dict[str, float] = {}
        recall: dict[str, float] = {}
        precision: dict[str, float] = {}

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(sum(all_ndcgs[f"NDCG@{k}"]) / len(scores), 5)
            _map[f"MAP@{k}"] = round(sum(all_aps[f"MAP@{k}"]) / len(scores), 5)
            recall[f"Recall@{k}"] = round(
                sum(all_recalls[f"Recall@{k}"]) / len(scores),
                5,
            )
            precision[f"P@{k}"] = round(sum(all_precisions[f"P@{k}"]) / len(scores), 5)

        # Create a safe copy of the dictionaries for evaluate_abstention
        abstention_data = {
            **{k: v.copy() for k, v in all_ndcgs.items()},
            **{k: v.copy() for k, v in all_aps.items()},
            **{k: v.copy() for k, v in all_recalls.items()},
            **{k: v.copy() for k, v in all_precisions.items()},
        }

        # Guard: filter out queries with empty result sets for abstention or skip entirely
        try:
            if any(len(r) == 0 for r in results.values()):
                # If all are empty, skip abstention entirely
                if all(len(r) == 0 for r in results.values()):
                    logger.warning(
                        "All queries returned empty retrieval sets; skipping nAUC abstention metrics.",
                    )
                    naucs: dict[str, float] = {}
                else:
                    # Filter results & correspondingly filter metric per-query lists
                    non_empty_qids = [qid for qid, r in results.items() if len(r) > 0]
                    if not non_empty_qids:
                        naucs = {}
                    else:
                        qid_order = list(scores.keys())  # type: ignore[name-defined]
                        keep_idx = [
                            i
                            for i, qid in enumerate(qid_order)
                            if qid in non_empty_qids
                        ]
                        filtered_abstention = {
                            k: [vals[i] for i in keep_idx]
                            for k, vals in abstention_data.items()
                        }
                        filtered_results = {qid: results[qid] for qid in non_empty_qids}
                        naucs = RetrievalEvaluator.evaluate_abstention(
                            filtered_results,
                            filtered_abstention,
                        )
            else:
                naucs = RetrievalEvaluator.evaluate_abstention(
                    results,
                    abstention_data,
                )
        except IndexError:
            logger.warning(
                "Abstention evaluation failed due to empty similarity scores; omitting nAUC metrics.",
            )
            naucs = {}

        return ndcg, _map, recall, precision, naucs

    @staticmethod
    def _evaluate_custom(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        metric: str,
        output_type: str = "all",
    ) -> tuple[dict[str, float], dict[str, float]]:
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            metric_scores = mrr(qrels, results, k_values, output_type)

        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            metric_scores = recall_cap(qrels, results, k_values, output_type)

        elif metric.lower() in ["hole", "hole@k"]:
            metric_scores = hole(qrels, results, k_values, output_type)

        elif metric.lower() in [
            "acc",
            "top_k_acc",
            "accuracy",
            "accuracy@k",
            "top_k_accuracy",
        ]:
            metric_scores = top_k_accuracy(qrels, results, k_values, output_type)

        # Guard against empty results causing IndexError
        try:
            if any(len(r) == 0 for r in results.values()):
                non_empty_qids = [qid for qid, r in results.items() if len(r) > 0]
                if not non_empty_qids:
                    naucs: dict[str, float] = {}
                else:
                    filtered_results = {qid: results[qid] for qid in non_empty_qids}
                    reference_key = next(iter(metric_scores))  # type: ignore[name-defined]
                    length = len(metric_scores[reference_key])  # type: ignore[name-defined]
                    if length != len(results):
                        naucs = {}
                    else:
                        qid_order = list(results.keys())
                        keep_idx = [
                            i
                            for i, qid in enumerate(qid_order)
                            if qid in non_empty_qids
                        ]
                        filtered_metric_scores = {
                            k: [vals[i] for i in keep_idx]
                            for k, vals in metric_scores.items()  # type: ignore[name-defined]
                        }
                        naucs = RetrievalEvaluator.evaluate_abstention(
                            filtered_results,
                            filtered_metric_scores,
                        )
            else:
                naucs = RetrievalEvaluator.evaluate_abstention(
                    results,
                    metric_scores,  # type: ignore[name-defined]
                )
        except (IndexError, KeyError, ValueError):
            logger.warning(
                "Abstention evaluation failed for custom metric; omitting nAUC metrics.",
            )
            naucs = {}
        metric_scores_avg = {k: sum(v) / len(v) for k, v in metric_scores.items()}  # type: ignore[name-defined]
        return metric_scores_avg, naucs
