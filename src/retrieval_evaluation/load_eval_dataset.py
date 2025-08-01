import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

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

from retrieval_evaluation.utils.dataset_utils import (  # noqa: E402
    feed_dataset_to_vespa,
    load_custom_dataset,
    load_vidore_dataset,
)
from src.config import settings
from src.repositories.embeddings_repository import EmbeddingsRepository
from src.repositories.vespa_repository import VespaRepository
from src.vespa_utils.vespa_schemas import evaluation_dataset_schema

logger = logging.getLogger(__name__)


async def main():
    """
    Main function to ulpload an evaluation dataset to Vespa.

    Usage:
    uv run evaluation/load_eval_dataset.py --model-name "vidore/colqwen2-v1.0" --dataset-name "consulting_dataset"
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description="Process dataset and upload to Vespa")
    parser.add_argument(
        "--model-name",
        default="vidore/colqwen2-v1.0",
        help="Model name to embed the dataset",
    )
    parser.add_argument(
        "--dataset-name",
        default="vidore/tabfquad_test_subsampled_beir",
        help="BEIR formatted dataset",
    )
    parser.add_argument(
        "--vespa-mode",
        choices=["deploy", "connect"],
        default="connect",
    )
    args = parser.parse_args()

    logging.basicConfig(filename="app.log", level=logging.INFO)

    # Load dataset
    if not args.dataset_name.startswith("vidore/"):
        ds = load_custom_dataset(
            args.dataset_name,  # Must be available under ./dataset/data/<dataset_name>
            # head=10,  # Load only the first 10 documents for testing
        )
    else:
        ds = load_vidore_dataset(
            args.dataset_name,
            # head=10,  # Load only the first 10 documents for testing
        )  # Only loads the first 10 documents with head=10

    # Vespa
    vespa_url = os.getenv(
        "VESPA_EVALS_APP_URL",
        "https://dbevals.vespa-app.cloud",
    )
    vespa_app_name = os.getenv("VESPA_EVALS_APP_NAME", "dbevals")
    tenant_name = os.getenv("VESPA_TENANT_NAME", "vespa-dev")
    instance_name = os.getenv("VESPA_EVALS_INSTANCE_NAME", "default")

    if args.vespa_mode == "deploy":
        # Deploy new Vespa DB for page data
        eval_schema = VespaRepository.convert_schema_to_vespa_schema(
            evaluation_dataset_schema,
        )
        VespaRepository.add_retrieval_and_rerank_rank_profile(
            eval_schema,
            embedding_field_name="embedding",
        )
        VespaRepository.add_binary_retrieval_rank_profile(
            eval_schema,
            binary_embedding_field_name="binary_embedding",
        )
        app_package = VespaRepository.create_vespa_application(
            vespa_app_name,
            eval_schema,
        )
        VespaRepository.deploy_to_vespa_cloud(app_package, tenant_name)

    vespa_app = VespaRepository.connect_to_vespa_cloud(
        url=vespa_url,
        app_name=vespa_app_name,
        tenant_name=tenant_name,
        instance_name=instance_name,
    )

    if not vespa_app:
        raise ValueError(
            f"Failed to connect to Vespa application {vespa_app_name} "
            f"for tenant {tenant_name} and instance {instance_name}.",
        )

    # Setup embeddings repository
    embeddings_repo = EmbeddingsRepository(
        x_api_key=settings.embeddings_secret_key,
        base_url=settings.embeddings_base_url,
    )

    await feed_dataset_to_vespa(
        vespa_app,
        embeddings_repo,
        args.dataset_name,
        ds,
    )

    print("Dataset upload completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
