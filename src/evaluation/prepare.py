import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

load_dotenv()

from pipeline.rags.factory_rag import RAGFactory

logger = logging.getLogger(__name__)


def verify_dataset(dataset_path: Path):
    if not dataset_path.exists():
        msg = f"Dataset {dataset_path} not found."
        logger.error(msg)
        raise FileNotFoundError(msg)


def verify_configs(config_path: Path):
    if not config_path.exists():
        msg = f"Config {config_path} not found."
        logger.error(msg)
        raise FileNotFoundError(msg)


async def main():
    """
    Create a new RAG system and index an evaluation dataset.

    Usage:
    uv run python -m evaluation.prepare --dataset-name="vidore/arxivqa_test_subsampled_beir" --rag-configs="multimodal_arxiv"
    """
    parser = argparse.ArgumentParser(description="Index dataset into RAG system")
    parser.add_argument(
        "--dataset-name",
        default="sherpa/consulting_light_dataset",
        help="BEIR formatted dataset",
    )
    parser.add_argument(
        "--rag-configs",
        default="multimodal_colqwen_page",
        help="RAG system configuration file name to use",
    )
    args = parser.parse_args()

    logging.basicConfig(filename="app.log", level=logging.INFO)

    dataset_path = src_path / "data/evaluation/datasets" / args.dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    verify_dataset(dataset_path)

    rag_configs_path = src_path / "configs" / f"{args.rag_configs}.json"
    verify_configs(rag_configs_path)

    evaluation_dir = os.getenv("EVALS_DATA_DIR")
    if evaluation_dir is None:
        evaluation_dir = str(src_path / "data/evaluation/datasets")
        print(f"EVALS_DATA_DIR not set, using default: {evaluation_dir}")
    evaluation_dir = Path(evaluation_dir)

    data_dir = os.getenv("RAGS_DATA_DIR")
    if data_dir is None:
        data_dir = str(src_path / "data/rags")
        print(f"RAGS_DATA_DIR not set, using default: {data_dir}")
    data_dir = Path(data_dir)

    with rag_configs_path.open("r") as f:
        rag_configs = json.load(f)

    evaluation_rag = RAGFactory.create_rag(rag_configs, data_dir)

    kb = rag_configs["configs"]["knowledge_base"]
    corpus_dir = evaluation_dir / kb / "corpuses"

    if rag_configs["type"] == "multimodal":
        documents = (
            list(corpus_dir.glob("*.png"))
            + list(corpus_dir.glob("*.jpg"))
            + list(corpus_dir.glob("*.jpeg"))
        )
        if not documents:
            print(f"No preprocessed images found in {corpus_dir}")
        else:
            print(
                f"Starting extraction (multimodal, preprocessed images) on {len(documents)} images...",
            )
            await evaluation_rag.extract(documents, preprocessed=True, batch_size=8)
            print("Extraction completed.")
    elif rag_configs["type"] == "traditional":
        documents = (
            list(corpus_dir.glob("*.png"))
            + list(corpus_dir.glob("*.jpg"))
            + list(corpus_dir.glob("*.jpeg"))
            + list(corpus_dir.glob("*.pdf"))
            + list(corpus_dir.glob("*.txt"))
            + list(corpus_dir.glob("*.md"))
        )
        if documents:
            print(f"Starting extraction (traditional) on {len(documents)} documents...")
            await evaluation_rag.extract(documents, batch_size=16)
            print("Extraction completed.")
        else:
            print(f"No documents found in {corpus_dir}")
    else:
        msg = f"Unsupported RAG type {rag_configs['type']} for prepare"
        raise ValueError(msg)

    print("Starting indexing...")
    await evaluation_rag.index()
    print("Indexing completed!")
    print(f"Dataset ready for evaluation in '{rag_configs['name']}'.")


if __name__ == "__main__":
    asyncio.run(main())
