import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import nest_asyncio
from datasets import Dataset, DatasetDict, load_dataset

# Get the absolute path to the project root directory
ROOT_DIR = Path(__file__).parent.parent

# Define other commonly used paths
SRC_DIR = ROOT_DIR / "src"
EVAL_DIR = ROOT_DIR / "evaluation"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))


from evaluation.evaluator.custom_evaluator import CustomEvaluator
from evaluation.retriever.custom_retriever import CustomVisionRetriever
from evaluation.utils.dataset_utils import ColPaliEvaluationBEIRDatasets
from repositories.colpali_repository import ColPaliEngineModels

logger = logging.getLogger(__name__)


async def _run_evaluation(
    models: list[ColPaliEngineModels],
    datasets: list[ColPaliEvaluationBEIRDatasets | str],
    metrics_path: str | None = None,
    evaluate_vidore_retrievers: bool = True,
    eval_mode: str | None = None,
):
    """
    Run evaluations on a list of models and datasets.

    Args:
    - models (list): A list of model names or paths.
    - datasets (list): A list of dataset names to evaluate on.
    - metrics_path (str): The path to save the evaluation metrics.
    - evaluate_vidore_retrievers (bool): Whether to evaluate ViDoRe retrievers.

    """
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

                metrics[model][dataset] = await _evaluate(
                    model_name=model,
                    dataset_to_load=dataset,
                    device="cuda",
                    num_workers=1,
                    batch_query=1,
                    batch_passage=1,
                    eval_mode=eval_mode,
                )
            except Exception as e:
                logger.info(f"Error evaluating {model} on {dataset}: {e}")
                logger.info("Skipping this model-dataset pair.")
                continue

    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    logger.info(f"Metrics saved to {metrics_path}")


async def _evaluate(
    model_name: str,
    dataset_to_load: str,
    device: str,
    num_workers: int,
    batch_query: int,
    batch_passage: int,
    eval_mode: str | None = None,
):
    """
    Evaluate a pretrained model on a dataset.

    Args:
    - model_name (str): The name or path of the pretrained model.
    - dataset_to_load (str): The name of the dataset to load.
    - device (str): The device to use for evaluation.
    - num_workers (int): The number of workers to use for data loading.
    - batch_query (int): The batch size for query processing.
    - batch_passage (int): The batch size for passage processing.

    Returns:
    - pandas.DataFrame: A dataframe containing the evaluation metrics.

    """
    # Setup retriever with pretrained model
    customRetriever = CustomVisionRetriever(
        model_name=model_name,
        dtype="auto",
        device=device,
        num_workers=num_workers,
    )
    ds = {}

    if dataset_to_load in ColPaliEvaluationBEIRDatasets:
        ds = {
            "corpus": load_dataset(dataset_to_load, name="corpus", split="test"),
            "queries": load_dataset(dataset_to_load, name="queries", split="test"),
            "qrels": load_dataset(dataset_to_load, name="qrels", split="test"),
        }
    # Look for custom datasets
    elif (
        Path(__file__).parent / f"dataset/data/{dataset_to_load}/dataset.json"
    ).exists():
        dataset_json_path = (
            Path(__file__).parent / f"dataset/data/{dataset_to_load}/dataset.json"
        )
        with dataset_json_path.open("r") as f:
            dataset = json.load(f)

        corpus_columns = ["id", "image", "doc-id", "corpus-id"]
        queries_columns = ["query-id", "query", "query-type"]
        qrels_columns = ["corpus-id", "query-id", "answer", "score"]

        # Convert list of lists into column-wise dicts
        corpus_data = {
            col: [row[i] for row in dataset["corpus"]]
            for i, col in enumerate(corpus_columns)
        }
        queries_data = {
            col: [row[i] for row in dataset["queries"]]
            for i, col in enumerate(queries_columns)
        }
        qrels_data = {
            col: [row[i] for row in dataset["qrels"]]
            for i, col in enumerate(qrels_columns)
        }

        dataset = DatasetDict(
            {
                "corpus": Dataset.from_dict(corpus_data),
                "queries": Dataset.from_dict(queries_data),
                "qrels": Dataset.from_dict(qrels_data),
            },
        )

        # Convert JSON to Dataset object
        corpus_columns = dataset["corpus"]["columns"]
        queries_columns = dataset["queries"]["columns"]
        qrels_columns = dataset["qrels"]["columns"]

        ds = {
            "corpus": dataset["corpus"],
            "queries": dataset["queries"],
            "qrels": dataset["qrels"],
        }

    else:
        raise ValueError(f"The dataset ({dataset_to_load}) is not supported.")

    customEvaluator = CustomEvaluator(customRetriever)

    return await customEvaluator.evaluate_dataset(
        ds=ds,
        batch_query=batch_query,
        batch_passage=batch_passage,
        batch_score=batch_passage,
        ds_name=dataset_to_load,
    )


def list_of_strings(arg):
    return arg.split(",")


async def main():
    """
    Main function to ulpload an evaluation dataset to Vespa.

    Usage:
    python3 apps/digital_brain_be/evaluation/evaluate.py --model-names "vidore/colqwen2-v1.0" --dataset-names "vidore/tabfquad_test_subsampled_beir" --metrics-output-path "evaluation/results/metrics.json"
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description="Evaluate models on datasets")

    parser.add_argument(
        "--model-names",
        default="vidore/colqwen2-v1.0",
        help="Model names to embed the datasets separated by commas",
        type=list_of_strings,
    )
    parser.add_argument(
        "--dataset-names",
        default="vidore/tabfquad_test_subsampled_beir",
        help="BEIR formatted datasets separated by commas",
        type=list_of_strings,
    )
    parser.add_argument("--dataset-dir", help="Directory containing dataset to process")
    parser.add_argument(
        "--metrics-output-path",
        default="apps/digital_brain_be/evaluation/results/metrics.json",
        help="Directory to save evaluation metrics",
    )
    parser.add_argument(
        "--eval-mode",
        default="v1",
        help="Evaluation mode, v1 accesses at corpus level, v2 accesses at page level. Only relevant for custom datasets.",
    )
    args = parser.parse_args()

    logging.basicConfig(filename="app.log", level=logging.INFO)

    logger.info("Evaluation started...")

    # Model and dataset names
    models: [ColPaliEngineModels] = args.model_names
    datasets: [ColPaliEvaluationBEIRDatasets] = args.dataset_names

    await _run_evaluation(
        models,
        datasets,
        args.metrics_output_path,
        eval_mode=args.eval_mode,
    )

    logger.info("Evaluation completed.")


if __name__ == "__main__":
    nest_asyncio.apply()  # TODO: Find a better fix - To prevent feed_async_iterable to throw
    asyncio.run(main())
