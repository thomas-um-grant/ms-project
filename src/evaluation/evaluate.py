import argparse
import asyncio
import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

from evaluation.evaluator import Evaluator

load_dotenv()
logger = logging.getLogger(__name__)


async def _run_evaluation(
    rag_configs_file: str,
    evaluation_name: str,
):
    """
    Run evaluations on a list of datasets.

    Args:
    - rag_configs_file (str): The rag configs to load for the evaluation.
    - evaluation_name (str): The name of the evaluation to run.

    """
    # Load RAG configurations
    rag_configs_path = (
        Path(__file__).parent.parent / "configs" / f"{rag_configs_file}.json"
    )
    if not rag_configs_path.exists():
        logger.error(f"RAG configs file not found: {rag_configs_path}")
        return

    with rag_configs_path.open("r") as f:
        rag_configs = json.load(f)

    metrics_path = Path(__file__).parent / "results/metrics.json"
    metrics_path.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if metrics_path.exists():
        with metrics_path.open("rb") as f:
            metrics = json.load(f)

    try:
        logger.info(f"Running {evaluation_name} on {rag_configs['name']}...")
        if rag_configs["name"] not in metrics:
            metrics[rag_configs["name"]] = {}
        if evaluation_name not in metrics[rag_configs["name"]]:
            metrics[rag_configs["name"]][evaluation_name] = {}

        metrics[rag_configs["name"]][evaluation_name] = await _evaluate(rag_configs)

        # Save after each dataset computed
        with metrics_path.open("w") as f:
            json.dump(metrics, f)

    except Exception as e:
        logger.info(
            f"Error running {evaluation_name} on {rag_configs['name']}: {e}",
            exc_info=True,
        )
        return

    logger.info(f"Metrics saved to {metrics_path}")


async def _evaluate(rag_configs: dict):
    """
    Evaluate a pretrained model on a dataset.

    Args:
    - rag_configs (dict): The RAG configurations.

    Returns:
    - pandas.DataFrame: A dataframe containing the evaluation metrics.

    """
    ds = None
    try:
        dataset_json_path = (
            Path(__file__).parent.parent
            / f"data/evaluation/{rag_configs['name']}/dataset.json"
        )
        if not dataset_json_path.exists():
            raise FileNotFoundError(
                f"Dataset JSON file not found: {dataset_json_path}",
            )

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

    # Setup retriever and evaluator
    logger.info("Setting up RAGFactory and Evaluator...")
    retriever = RAGFactory(rag_configs)
    evaluator = Evaluator(retriever)
    logger.info("RAGFactory and Evaluator set up successfully.")

    return await evaluator.evaluate_dataset(
        ds=ds,
        k=100,
        ds_name=dataset_to_load,
    )


async def main():
    """
    Main function to evaluate a dataset with the retriever.

    Usage:
    uv run evaluation/evaluate.py --rag-configs "multimodal_colqwen" --evaluation-name "default"
    """
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")

    parser.add_argument(
        "--rag-configs",
        default="multimodal_colqwen_page",
        help="RAG configurations",
    )
    parser.add_argument(
        "--evaluation-name",
        default="evaluation",
        help="Name of the evaluation",
        type=str,
    )
    args = parser.parse_args()

    logging.basicConfig(filename="app.log", level=logging.INFO)

    logger.info("Evaluation started...")

    await _run_evaluation(args.rag_configs, args.evaluation_name)

    logger.info("Evaluation completed.")


if __name__ == "__main__":
    # Add this at the top of the file to allow running the script directly
    # # Add src path for imports
    # src_path = Path(__file__).parent.parent
    # if str(src_path) not in sys.path:
    #     sys.path.append(str(src_path))

    asyncio.run(main())
