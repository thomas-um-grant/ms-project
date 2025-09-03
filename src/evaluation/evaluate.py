import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

# Add src path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

load_dotenv()

from evaluation.evaluator import Evaluator
from pipeline.rags.factory_rag import RAGFactory

# Configure logging
log_file_path = Path(__file__).parent / "evaluation.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True)

# Clear existing handlers and configure root logger
logging.root.handlers = []
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            log_file_path,
            mode="w",
            encoding="utf-8",
        ),  # Overwrite file each run
    ],
    force=True,
)

# Set all loggers to DEBUG to capture everything
logging.getLogger().setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.info("=== Evaluation script started ===")
logger.info(f"Log file: {log_file_path.absolute()}")


def _rows_are_dicts(rows: list) -> bool:
    return isinstance(rows, list) and len(rows) > 0 and isinstance(rows[0], dict)


def _build_column_dict(rows: list, columns: list[str]) -> dict[str, list]:
    """Support both list-of-dicts (Vidore JSON) and list-of-lists (generic BEIR export)."""
    if _rows_are_dicts(rows):
        return {col: [row.get(col) for row in rows] for col in columns}
    # Assume list-of-lists; build index map 1:1
    idx_map = {i: col for i, col in enumerate(columns)}
    out: dict[str, list] = {c: [] for c in columns}
    for row in rows:
        for i, val in enumerate(row):
            col = idx_map.get(i)
            if col:
                out[col].append(val)
    return out


def _infer_complexity(qrels_ds: Dataset, has_doc_ids_flag: bool) -> str:
    if not has_doc_ids_flag:
        return "v1"
    # If every (corpus-id, doc-id) pair collapses to unique doc-id '0' treat as v1
    try:
        doc_ids = (
            set(qrels_ds["doc-id"]) if "doc-id" in qrels_ds.column_names else set()
        )
        if len(doc_ids) <= 1:
            return "v1"
    except KeyError:
        return "v1"
    return "v2"


async def _run_evaluation(
    rag_configs_file: str,
    evaluation_name: str,
    complexity_mode: str = "auto",
    *,
    disable_generation: bool = False,
):
    """
    Run evaluations on a list of datasets.

    Args:
    - rag_configs_file (str): The rag configs to load for the evaluation.
    - evaluation_name (str): The name of the evaluation to run.
    - complexity_mode (str): 'auto', 'v1', or 'v2'. If auto, infer from dataset.

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

    metrics_file = (
        Path(__file__).parent / "results" / f"metrics_{rag_configs['name']}.json"
    )
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(
            f"Running {evaluation_name} on {rag_configs['name']} (complexity_mode={complexity_mode})...",
        )

        # Inject disable_generation into configs dict (non-destructive to persisted file)
        if disable_generation:
            rag_configs.setdefault("configs", {})["disable_generation"] = True
        result = await _evaluate(
            rag_configs,
            complexity_mode=complexity_mode,
        )

        # Reload metrics after evaluation completes
        if metrics_file.is_file():
            try:
                with metrics_file.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except json.JSONDecodeError:
                logger.warning("metrics.json was corrupt; starting fresh.")
                metrics = {}
        else:
            metrics = {}

        base_key = evaluation_name
        metrics.setdefault(rag_configs["name"], {})
        if base_key in metrics[rag_configs["name"]]:
            evaluation_name = f"{base_key}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
            logger.info(
                "Evaluation name already existed. Using unique key '%s'",
                evaluation_name,
            )

        metrics[rag_configs["name"]][evaluation_name] = result
        with metrics_file.open("w", encoding="utf-8") as f:
            json.dump(metrics, f)

    except (OSError, ValueError, RuntimeError):  # narrow common runtime issues
        logger.exception(
            "Error running %s on %s",
            evaluation_name,
            rag_configs.get("name"),
        )
        raise
    logger.info(f"Metrics saved to {metrics_file}")


async def _evaluate(rag_configs: dict, complexity_mode: str = "auto"):
    """
    Evaluate a pretrained model on a dataset.

    Args:
    - rag_configs (dict): The RAG configurations.
    - complexity_mode (str): 'auto', 'v1', or 'v2'.

    Returns:
    - dict: Evaluation metrics.

    """
    rags_data_dir = os.getenv("RAGS_DATA_DIR")
    if rags_data_dir is None:
        rags_data_dir = str(src_path / "data/rags")
        print(f"RAGS_DATA_DIR not set, using default: {rags_data_dir}")

    data_dir = Path(rags_data_dir)

    try:
        dataset_name = rag_configs["configs"]["knowledge_base"]
        dataset_json_path = (
            Path(__file__).parent.parent
            / f"data/evaluation/datasets/{dataset_name}/dataset.json"
        )
        if not dataset_json_path.exists():
            msg = f"Dataset JSON file not found: {dataset_json_path}"
            raise FileNotFoundError(msg)

        with dataset_json_path.open("r") as f:
            dataset = json.load(f)

        # Expected columns (remove ambiguous standalone 'id')
        corpus_columns = ["corpus-id", "doc-id"]
        queries_columns = ["query-id", "query", "query-type"]
        qrels_columns = ["query-id", "corpus-id", "doc-id", "answer", "score"]

        # Build column dicts robustly
        corpus_data = _build_column_dict(dataset["corpus"], corpus_columns)
        queries_data = _build_column_dict(dataset["queries"], queries_columns)
        qrels_data = _build_column_dict(dataset["qrels"], qrels_columns)

        # Validate score field & cast (binarize >0 -> 1)
        original_scores: list[int] = []
        anomalous = 0
        for raw in qrels_data["score"]:
            try:
                val = int(raw)
            except (TypeError, ValueError):
                val = 0
            original_scores.append(val)
            if val not in (0, 1):
                anomalous += 1
        if anomalous:
            logger.warning(
                "Detected %d non-binary relevance scores (out of %d); binarizing (>0 -> 1).",
                anomalous,
                len(original_scores),
            )
        qrels_data["score"] = [1 if v > 0 else 0 for v in original_scores]

        # Integrity checks
        corpus_ids = set(corpus_data.get("corpus-id", []))
        missing = sum(1 for c in qrels_data.get("corpus-id", []) if c not in corpus_ids)
        if missing:
            logger.warning(
                "%d qrels entries reference unknown corpus-id values.",
                missing,
            )

        ds = DatasetDict(
            {
                "corpus": Dataset.from_dict(corpus_data),
                "queries": Dataset.from_dict(queries_data),
                "qrels": Dataset.from_dict(qrels_data),
            },
        )
    except (OSError, json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        logger.exception(
            "Error loading custom dataset %s: %s",
            locals().get("dataset_name"),
            e,
        )
        msg = f"Custom dataset {locals().get('dataset_name')} could not be loaded."
        raise ValueError(msg) from e

    # Determine complexity automatically
    has_doc_ids = "doc-id" in ds["qrels"].column_names and any(
        d is not None for d in ds["qrels"]["doc-id"]
    )
    inferred_complexity = _infer_complexity(ds["qrels"], has_doc_ids)
    effective_complexity = (
        inferred_complexity if complexity_mode == "auto" else complexity_mode
    )
    if effective_complexity not in {"v1", "v2"}:
        logger.warning(
            "Invalid complexity_mode '%s'; falling back to inferred '%s'",
            complexity_mode,
            inferred_complexity,
        )
        effective_complexity = inferred_complexity
    logger.info(
        "Using complexity level: %s (inferred=%s, mode=%s)",
        effective_complexity,
        inferred_complexity,
        complexity_mode,
    )

    # Setup retriever and evaluator
    logger.info("Setting up RAGFactory and Evaluator...")
    # Propagate disable_generation flag if present in rag_configs["configs"] (set via CLI or config)
    disable_generation_flag = bool(
        rag_configs.get("configs", {}).get("disable_generation", False),
    )
    evaluation_rag = RAGFactory.create_rag(
        rag_configs,
        data_dir,
        disable_generation=disable_generation_flag,
    )
    evaluator = Evaluator(evaluation_rag)
    logger.info("RAGFactory and Evaluator set up successfully.")

    return await evaluator.evaluate_dataset(
        ds=ds,
        k=100,
        batch_size=10,
        complexity=effective_complexity,
    )


async def main():
    """
    Main function to evaluate a dataset with the retriever.

    Usage:
    uv run python -m evaluation.evaluate --rag-configs "multimodal_arxiv" --evaluation-name "default_tester"
    """
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")

    parser.add_argument(
        "--rag-configs",
        default="multimodal_page",
        help="RAG configurations",
    )
    parser.add_argument(
        "--evaluation-name",
        default="evaluation",
        help="Name of the evaluation",
        type=str,
    )
    parser.add_argument(
        "--complexity",
        choices=["auto", "v1", "v2"],
        default="auto",
        help="Evaluation complexity: auto infers, or force v1/v2.",
    )
    parser.add_argument(
        "--disable-generation",
        action="store_true",
        help="Disable loading generation model (retrieval-only evaluation).",
    )
    args = parser.parse_args()

    logger.info("Evaluation started...")

    await _run_evaluation(
        args.rag_configs,
        args.evaluation_name,
        args.complexity,
        disable_generation=args.disable_generation,
    )

    logger.info("Evaluation completed.")


if __name__ == "__main__":
    asyncio.run(main())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
