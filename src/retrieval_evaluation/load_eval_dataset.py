import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import nest_asyncio
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

from evaluation.retriever.sherpa_retriever import SherpaVisionRetriever  # noqa: E402
from evaluation.utils.dataset_utils import (  # noqa: E402
    load_vidore_dataset,
    prepare_dataset,
)
from evaluation.utils.vespa_utils import (  # noqa: E402
    add_rank_profiles,
    connect_existing_vespa,
    create_vespa_application,
    define_vespa_schema,
    deploy_to_vespa_cloud,
    feed_data_to_vespa,
)

logger = logging.getLogger(__name__)


async def main():
    """
    Main function to ulpload an evaluation dataset to Vespa.

    Usage:
    python3 apps/digital_brain_be/evaluation/load_eval_dataset.py --tenant-name "sherpa-dev" --instance-name "default" --vespa-app-name "sherpadbevals" --model-name "vidore/colqwen2-v1.0" --dataset-name "vidore/tabfquad_test_subsampled_beir"
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description="Process dataset and upload to Vespa")
    parser.add_argument(
        "--vespa-app-name",
        default=os.getenv("VESPA_EVALS_APP_NAME", "sherpadbevals"),
        help="Vespa application name",
    )
    parser.add_argument(
        "--tenant-name",
        default=os.getenv("VESPA_TENANT_NAME", "sherpa-dev"),
        help="Vespa tenant name",
    )
    parser.add_argument(
        "--instance-name",
        default=os.getenv("VESPA_EVALS_INSTANCE_NAME", "default"),
        help="Vespa instance name",
    )
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
    parser.add_argument("--dataset-dir", help="Directory containing dataset to process")
    parser.add_argument(
        "--vespa-mode", choices=["deploy", "connect"], default="connect"
    )
    args = parser.parse_args()

    logging.basicConfig(filename="app.log", level=logging.INFO)

    # Load dataset
    if not args.dataset_name.startswith("vidore/"):
        # TODO: Load dataset from local folder
        pass
    else:
        ds = load_vidore_dataset(
            args.dataset_name
        )  # Only loads the first 5 documents with head=5

    # Setup retriever
    retriever = SherpaVisionRetriever(
        model_name=args.model_name, dtype="auto", device="cuda", num_workers=4
    )

    # Prepare dataset
    formatted_data = await prepare_dataset(
        args.dataset_name, ds, retriever, batch_size=4
    )

    if args.vespa_mode == "deploy":
        # Define Vespa schema
        schema = define_vespa_schema()

        # Add rank profiles
        add_rank_profiles(schema)

        # Create Vespa application
        app_package = create_vespa_application(args.vespa_app_name, schema)

        # Deploy to Vespa Cloud
        app = deploy_to_vespa_cloud(app_package)

    elif args.vespa_mode == "connect":
        app = connect_existing_vespa(
            args.vespa_app_name,
            tenant_name=args.tenant_name,
            instance_name=args.instance_name,
        )

    # Feed data to Vespa
    feed_data_to_vespa(app, formatted_data)

    print("Dataset upload completed successfully!")


if __name__ == "__main__":
    nest_asyncio.apply()  # To prevent feed_async_iterable to throw
    asyncio.run(main())
