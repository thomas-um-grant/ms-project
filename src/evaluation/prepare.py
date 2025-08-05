import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add src path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from retrieval_pipeline.rags.multimodal import MultiModalRAG

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
    uv run evaluation/prepare.py --dataset-name "sherpa/consulting_light_dataset" --rag-configs "multimodal_colqwen"
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

    # Params validation
    dataset_path = Path(__file__).parent.parent / "data/evaluation" / args.dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    verify_dataset(dataset_path)

    rag_configs_path = (
        Path(__file__).parent.parent / "configs" / f"{args.rag_configs}.json"
    )
    verify_configs(rag_configs_path)

    # Create RAG system
    # Configuration
    evaluation_data_dir = os.getenv("EVALS_DATA_DIR")
    if evaluation_data_dir is None:
        # Fallback to default if environment variable is not set
        evaluation_data_dir = (
            "/Users/thomas/repos/imperial/project/ms-project/src/data/evaluation"
        )
        print(f"EVALS_DATA_DIR not set, using default: {evaluation_data_dir}")

    evaluation_data_dir = Path(evaluation_data_dir)

    # Load RAG configs
    rag_configs = {}
    with rag_configs_path.open("r") as f:
        rag_configs = json.load(f)

    if rag_configs["type"] == "traditional":
        # evaluation_rag = TraditionalRAG(**rag_configs)
        pass
    elif rag_configs["type"] == "multimodal":
        evaluation_rag = MultiModalRAG(
            name=rag_configs["name"],
            data_dir=evaluation_data_dir,
            embedding_model=rag_configs["embedding_model"],
            generation_model=rag_configs["generation_model"],
            configs=rag_configs["configs"],
        )
    elif rag_configs["type"] == "graph":
        # evaluation_rag = GraphRAG(**rag_configs)
        pass
    elif rag_configs["type"] == "custom":
        # evaluation_rag = CustomRAG(**rag_configs)
        pass
    else:
        msg = f"Unsupported RAG type: {rag_configs['type']}"
        logger.error(msg)
        raise ValueError(msg)

    # Extract metadata from corpuses
    documents = list((dataset_path / "corpuses").glob("*.jpg"))

    print("Starting extraction...")
    await evaluation_rag.extract(documents, preprocessed=True)
    print("Extraction completed!")

    # Index all corpuses
    print("Starting indexing...")
    await evaluation_rag.index()
    print("Indexing completed!")

    print(f"Dataset ready for evaluation in '{rag_configs['name']}'.")


if __name__ == "__main__":
    asyncio.run(main())
