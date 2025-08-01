import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# import nest_asyncio
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv

load_dotenv()

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


from retrieval_evaluation.evaluation_retriever import (
    EvaluationRetriever,
)
from retrieval_evaluation.evaluator import Evaluator
from retrieval_evaluation.utils.dataset_utils import ColPaliEvaluationBEIRDatasets

logger = logging.getLogger(__name__)


async def _run_evaluation(
    datasets: list[ColPaliEvaluationBEIRDatasets | str],
    evaluation_name: str = "evaluation",
    metrics_path: str | None = None,
):
    """
    Run evaluations on a list of datasets.

    Args:
    - datasets (list): A list of dataset names to evaluate on.
    - evaluation_name (str): The name of the evaluation to run.
    - metrics_path (str): The path to save the evaluation metrics.

    """
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    for dataset in datasets:
        try:
            logger.info(f"Running {evaluation_name} on {dataset}...")
            if evaluation_name not in metrics:
                metrics[evaluation_name] = {}
            if dataset not in metrics[evaluation_name]:
                metrics[evaluation_name][dataset] = {}

            metrics[evaluation_name][dataset] = await _evaluate(
                dataset_to_load=dataset,
                device="cuda",
                num_workers=1,
            )

            # Save after each dataset computed
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)

        except Exception as e:
            logger.info(
                f"Error running {evaluation_name} on {dataset}: {e}",
                exc_info=True,
            )
            logger.info("Skipping.")
            continue

    logger.info(f"Metrics saved to {metrics_path}")


async def _evaluate(
    dataset_to_load: str,
    device: str,
    num_workers: int,
):
    """
    Evaluate a pretrained model on a dataset.

    Args:
    - dataset_to_load (str): The name of the dataset to load.
    - device (str): The device to use for retrieval_evaluation.
    - num_workers (int): The number of workers to use for data loading.

    Returns:
    - pandas.DataFrame: A dataframe containing the evaluation metrics.

    """
    ds = None
    if dataset_to_load in ColPaliEvaluationBEIRDatasets:
        try:
            ds = {
                "corpus": load_dataset(dataset_to_load, name="corpus", split="test"),
                "queries": load_dataset(dataset_to_load, name="queries", split="test"),
                "qrels": load_dataset(dataset_to_load, name="qrels", split="test"),
            }
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_to_load}: {e}")
            raise ValueError(f"Dataset {dataset_to_load} could not be loaded.")

    # Look for custom datasets
    elif (
        Path(__file__).parent / f"dataset/data/{dataset_to_load}/dataset.json"
    ).exists():
        dataset_json_path = (
            Path(__file__).parent / f"dataset/data/{dataset_to_load}/dataset.json"
        )
        try:
            with dataset_json_path.open("r") as f:
                dataset = json.load(f)

            corpus_columns = ["id", "image", "doc-id", "corpus-id"]
            queries_columns = ["query-id", "query", "query-type"]
            qrels_columns = ["corpus-id", "query-id", "answer", "score"]

            # Convert list of lists into column-wise dicts
            corpus_data = {
                col: [row[col] for row in dataset["corpus"]] for col in corpus_columns
            }
            queries_data = {
                col: [row[col] for row in dataset["queries"]] for col in queries_columns
            }
            qrels_data = {
                col: [row[col] for row in dataset["qrels"]] for col in qrels_columns
            }

            ds = DatasetDict(
                {
                    "corpus": Dataset.from_dict(corpus_data),
                    "queries": Dataset.from_dict(queries_data),
                    "qrels": Dataset.from_dict(qrels_data),
                },
            )
        except Exception as e:
            logger.error(f"Error loading custom dataset {dataset_to_load}: {e}")
            raise ValueError(f"Custom dataset {dataset_to_load} could not be loaded.")

    else:
        raise ValueError(f"The dataset ({dataset_to_load}) is not supported.")

    if ds is None:
        raise ValueError(f"Dataset {dataset_to_load} could not be loaded.")

    # Setup retriever and evaluator
    logger.info("Setting up EvaluationRetriever and Evaluator...")
    retriever = EvaluationRetriever(device=device, num_workers=num_workers)
    evaluator = Evaluator(retriever)
    logger.info("EvaluationRetriever and Evaluator set up successfully.")

    return await evaluator.evaluate_dataset(
        ds=ds,
        k=100,
        ds_name=dataset_to_load,
    )


def list_of_strings(arg):
    return arg.split(",")


async def main():
    """
    Main function to evaluate a dataset with the retriever.

    Usage:
    uv run evaluation/evaluate.py --dataset-names "consulting_dataset" --evaluation-name "default"
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description="Evaluate models on datasets")

    parser.add_argument(
        "--dataset-names",
        default="vidore/tabfquad_test_subsampled_beir",
        help="BEIR formatted datasets separated by commas",
        type=list_of_strings,
    )
    parser.add_argument(
        "--evaluation-name",
        default="evaluation",
        help="Name of the evaluation",
        type=str,
    )
    parser.add_argument(
        "--metrics-output-path",
        default="evaluation/results/metrics.json",
        help="Directory to save evaluation metrics",
    )
    args = parser.parse_args()

    logging.basicConfig(filename="app.log", level=logging.INFO)

    logger.info("Evaluation started...")

    # Extract list of dataset names from the argument
    ds_names = []
    if isinstance(args.dataset_names, str):
        ds_names = args.dataset_names.split(",")
    elif isinstance(args.dataset_names, list):
        ds_names = [name.strip() for name in args.dataset_names]

    await _run_evaluation(ds_names, args.evaluation_name, args.metrics_output_path)

    logger.info("Evaluation completed.")


if __name__ == "__main__":
    asyncio.run(main())
