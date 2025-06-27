import json
import logging
from pathlib import Path

from openai import AzureOpenAI

logger = logging.getLogger(__name__)


def upload_files(
    folder: Path,
    open_ai_client: AzureOpenAI,
) -> None:
    """Upload raw files to the assistant."""
    # Define paths
    metadata_file = folder / "extracted_data/metadata.json"

    if not metadata_file.exists():
        logger.error(f"Extracted files metadata not found: {metadata_file}")
        return

    print(f"Uploading files from folder: {folder.resolve() / 'extracted_data/pdfs'}")

    # Load metadata of extracted files
    metadata = {}
    with metadata_file.open() as f:
        metadata = json.load(f)

    for corpus_id in metadata:
        # Load document onto the OpenAI cloud
        pdf_path = folder / f"extracted_data/pdfs/{corpus_id}.pdf"

        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            continue

        print(f"Uploading file {corpus_id} from {pdf_path}")
        uploaded_file = open_ai_client.files.create(
            file=pdf_path.open("rb"),
            purpose="assistants",
        )
        if not uploaded_file:
            logger.error(f"Failed to upload file: {pdf_path}")
            continue

        # Store the file_id and name in the scores dictionary
        metadata[str(corpus_id)]["openai_file_id"] = uploaded_file.id
        print(f"File uploaded with file_id: {uploaded_file.id}.")

    # Save the updated metadata with file IDs
    output_file = folder / "extracted_data/metadata.json"
    with output_file.open("w") as f:
        json.dump(metadata, f, indent=2)
