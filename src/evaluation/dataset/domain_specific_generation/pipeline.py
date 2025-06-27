import argparse
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from extractor import create_metadata
from formatter import (
    convert_to_excel,
    format_beir_dataset,
    generate_human_evaluation_json,
)
from generator import generate_answers, generate_questions
from langfuse import Langfuse, observe
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    filename=Path(__file__).parent.resolve() / "dataset_generator.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@observe()
def main(model_name: str, raw_folder_path: str = "pdfs") -> None:
    # Setup Langfuse, wrapping OpenAI client
    lang_fuse_client = Langfuse()

    # TODO: Batching all openai calls when available in their SDK
    # TODO: Make sure the json output is valid and well formatted with the correct structure
    # For this, use structured format: https://ai.google.dev/gemini-api/docs/structured-output
    # For now we use a thread conversation with the Azure OpenAI client assistant because Batching is not available yet.
    open_ai_client = AzureOpenAI()

    dataset_folder = Path(__file__).parent.parent.parent.parent / raw_folder_path

    # Extract raw files from specified folder, ensure pdfs are valid and rename them.
    # logger.info(f"Extracting raw files from {dataset_folder}...")
    # extract_raw_files(dataset_folder)

    # Upload files to the assistant (OpenAI) for later processing.
    # logger.info(f"Uploading files from {dataset_folder}...")
    # upload_files(dataset_folder, open_ai_client)

    # Extract metadata from the pdf files with an LLM and save it in a JSON file.
    logger.info("Creating metadata from the documents...")
    create_metadata(lang_fuse_client, open_ai_client, dataset_folder, model_name)

    # Generate the questions with an LLM
    logger.info("Generating questions from the documents...")
    generate_questions(lang_fuse_client, open_ai_client, dataset_folder, model_name)

    # Generate the answers with an LLM
    logger.info("Generating answers from the documents...")
    generate_answers(lang_fuse_client, open_ai_client, dataset_folder, model_name)

    # Format into BEIR dataset
    logger.info("Formatting dataset to BEIR format...")
    format_beir_dataset(dataset_folder)

    # Generate csv file for Human Evaluation
    logger.info("Generating human evaluation JSON and CSV...")
    generate_human_evaluation_json(dataset_folder)
    convert_to_excel(dataset_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset generation pipeline.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-v2",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--raw_folder_path",
        type=str,
        default="pdfs",
        help="Path to the folder containing raw PDF files.",
    )
    args = parser.parse_args()

    # Time the execution
    start_time = time.time()
    main(
        model_name=args.model_name,
        raw_folder_path=args.raw_folder_path,
    )
    end_time = time.time()
    logger.info(f"Execution time: {end_time - start_time} seconds")
