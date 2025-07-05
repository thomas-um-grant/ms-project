import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

from assigner import cluster_documents
from dotenv import load_dotenv
from extractor import (
    create_metadata,
    extract_metadata_pages_from_dataset,
    extract_raw_files,
)
from formatter import (
    convert_to_excel,
    format_beir_dataset,
    generate_human_evaluation_json,
)
from generator import generate_answers, generate_questions
from langfuse import Langfuse, observe
from openai import AzureOpenAI
from retriever import retrieve_top_k_pages_for_questions
from reviewer import review_answers
from uploader import upload_documents, upload_single_pages

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    filename=Path(__file__).parent.resolve() / "dataset_generator.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

AZURE_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT", "https://openai-charter.openai.azure.com/"
)
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-v2")
AZURE_MINI_MODEL = os.getenv("AZURE_OPENAI_MINI_MODEL", "gpt-4o-mini")
AZURE_EMBEDDING_MODEL = os.getenv(
    "AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
)


@observe()
async def main(data_folder_path: str = "pdfs") -> None:
    # Setup Langfuse and AzureOpenAI
    langfuse_client = Langfuse()

    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )

    data_folder = Path(__file__).parent.parent.parent.parent / data_folder_path

    # TODO: Batching all openai calls when available in their SDK
    # TODO: Make sure the json output is valid and well formatted with the correct structure
    # For this, use structured format: https://ai.google.dev/gemini-api/docs/structured-output
    # For now we use a thread conversation with the Azure OpenAI client assistant because Batching is not available yet.

    # Extract raw files from specified folder, ensure pdfs are valid and rename them.
    logger.info(f"Extracting raw files from {data_folder}...")
    extract_raw_files(data_folder)

    # Upload files to the assistant (OpenAI) for later processing.
    logger.info(f"Uploading files from {data_folder}...")
    upload_documents(data_folder, openai_client)

    # Extract metadata from the pdf files with an LLM and save it in a JSON file.
    logger.info("Creating metadata from the documents...")
    create_metadata(langfuse_client, openai_client, data_folder, AZURE_MODEL)

    # Cluster the documents based on their metadata for better question assignment.
    cluster_documents(
        data_folder,
        langfuse_client,
        openai_client,
        model_name=AZURE_MINI_MODEL,
        embedding_model=AZURE_EMBEDDING_MODEL,
    )

    # Generate the questions with an LLM
    logger.info("Generating questions from the documents...")
    generate_questions(langfuse_client, openai_client, data_folder, AZURE_MODEL)

    # Upload single pages to Azure OpenAI file storage to extract metadata and get embeddings.
    logger.info("Uploading pages to OpenAI file storage...")
    await upload_single_pages(data_folder, openai_client)

    # Generate metadata for each page using an LLM.
    logger.info("Extracting metadata for each page...")
    await extract_metadata_pages_from_dataset(
        langfuse_client, openai_client, model=AZURE_MINI_MODEL, data_folder=data_folder
    )

    # Retrieve the top k pages for each question based on their embeddings.
    logger.info("Retrieving top k pages for each question...")
    retrieve_top_k_pages_for_questions(
        data_folder,
        openai_client,
        embedding_model=AZURE_EMBEDDING_MODEL,
    )

    # Generate the answers with an LLM
    logger.info("Generating answers from the documents...")
    generate_answers(
        langfuse_client,
        openai_client,
        data_folder,
        model=AZURE_MODEL,
    )

    # Review the generated answers to ensure they are faithful and correct.
    logger.info("Reviewing generated answers...")
    review_answers(
        data_folder,
        langfuse_client,
        openai_client,
        model=AZURE_MODEL,
    )

    # Format into BEIR dataset
    logger.info("Formatting dataset to BEIR format...")
    format_beir_dataset(data_folder)

    # Generate csv file for Human Evaluation
    logger.info("Generating human evaluation JSON and CSV...")
    generate_human_evaluation_json(data_folder)
    convert_to_excel(data_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset generation pipeline.")
    parser.add_argument(
        "--data_folder_path",
        type=str,
        default="dataset/data",
        help="Path to the folder containing raw PDF files.",
    )
    args = parser.parse_args()

    # Time the execution
    start_time = time.time()
    asyncio.run(
        main(
            data_folder_path=args.data_folder_path,
        )
    )
    end_time = time.time()
    logger.info(f"Execution time: {end_time - start_time} seconds")
