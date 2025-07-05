import json
import logging
import os
import sys
from enum import Enum
from pathlib import Path

from datasets import load_dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from repositories.colpali_repository import ColPaliEngineModels

logger = logging.getLogger(__name__)


class ColPaliEvaluationDatasets(Enum):
    ARXIVQA_DATASET = "vidore/arxivqa_test_subsampled"
    DOCVQA_DATASET = "vidore/docvqa_test_subsampled"
    INFOVQA_DATASET = "vidore/infovqa_test_subsampled"
    TABFQUAD_DATASET = "vidore/tabfquad_test_subsampled"
    TATQDA_DATASET = "vidore/tatdqa_test"


def run_evals(
    models: list[ColPaliEngineModels],
    datasets: list[ColPaliEvaluationDatasets],
    metrics_path: str | None = None,
    evaluate_vidore_retrievers: bool = True,
):
    """
    Run evaluations on a list of models and datasets.

    Args:
    - models (list): A list of model names or paths.
    - datasets (list): A list of dataset names to evaluate on.
    - metrics_path (str): The path to save the evaluation metrics.
    - evaluate_vidore_retrievers (bool): Whether to evaluate ViDoRe retrievers.

    """
    if metrics_path is None:
        metrics_path = "src/evals/results/vidore_metrics.json"

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    for model in models:
        for dataset in datasets:
            try:
                logger.info(f"Evaluating {model} on {dataset}...")
                if model not in metrics:
                    metrics[model] = {}
                if dataset not in metrics[model]:
                    metrics[model][dataset] = {}

                metrics[model][dataset] = _evaluate(
                    model_name_or_path=model,
                    dataset_to_load=dataset,
                    device="mps",
                    split="test",
                    num_workers=0,  # TODO: Change to 4, currently set to 0 because having a RuntimeError: _share_filename_: only available on CPU
                    batch_query=4,
                    batch_passage=4,
                )
            except Exception as e:
                logger.info(f"Error evaluating {model} on {dataset}: {e}")
                logger.info("Skipping this model-dataset pair.")
                continue

    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    logger.info(f"Metrics saved to {metrics_path}")


def _evaluate(
    model_name_or_path: str,
    use_vidore_retriever: bool,
    dataset_to_load: str,
    device: str,
    split: str,
    num_workers: int,
    batch_query: int,
    batch_passage: int,
):
    """
    Evaluate a pretrained model on a dataset.

    Args:
    - model_name_or_path (str): The name or path of the pretrained model.
    - dataset_to_load (str): The name of the dataset to load.
    - device (str): The device to use for evaluation.
    - split (str): The split of the dataset to evaluate on.
    - num_workers (int): The number of workers to use for data loading.
    - batch_query (int): The batch size for query processing.
    - batch_passage (int): The batch size for passage processing.

    Returns:
    - pandas.DataFrame: A dataframe containing the evaluation metrics.

    """
    # Setup retriever with pretrained model
    sherpaRetriever = RAGVisionRetriever(
        model_name_or_path=model_name_or_path,
        device=device,
        num_workers=num_workers,
        use_vidore_retriever=use_vidore_retriever,
    )

    # Evaluate on a QA datasets
    sherpaEvaluator = RAGEvaluatorQA(sherpaRetriever)
    ds = load_dataset(dataset_to_load, split=split)

    return sherpaEvaluator.evaluate_dataset(
        ds=ds,
        batch_query=batch_query,
        batch_passage=batch_passage,
        batch_score=batch_passage,
    )
