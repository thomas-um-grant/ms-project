import asyncio
import logging
from pathlib import Path
from typing import Any

from loader import load_metadata, save_metadata
from openai import AzureOpenAI
from PyPDF2 import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


def upload_documents(
    folder: Path,
    openai_client: AzureOpenAI,
) -> None:
    """Upload raw files to the assistant."""
    # Define paths
    metadata = load_metadata(folder)

    for corpus_id in metadata:
        # Validate metadata structure
        if not isinstance(metadata[corpus_id], dict):
            logger.error(f"Invalid metadata structure for corpus_id {corpus_id}")
            continue

        # Load document onto the OpenAI cloud
        pdf_path = folder / f"extracted_data/pdfs/{corpus_id}.pdf"

        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            continue

        try:
            with pdf_path.open("rb") as file_handle:
                uploaded_file = openai_client.files.create(
                    file=file_handle,
                    purpose="assistants",
                )
            if not uploaded_file or not hasattr(uploaded_file, "id"):
                logger.error(f"Failed to upload file: {pdf_path}")
                continue
        except Exception as e:
            logger.error(f"Error uploading file {pdf_path}: {e}")
            continue

        # Store the file_id and name in the scores dictionary
        metadata[str(corpus_id)]["openai_file_id"] = uploaded_file.id
        logger.info(f"File uploaded with file_id: {uploaded_file.id}.")

    # Save the updated metadata with file IDs
    save_metadata(folder, metadata)


async def upload_page(
    page_path: Path,
    openai_client: AzureOpenAI,
) -> Any:
    """Upload a single page to the OpenAI file storage."""
    logger.info(f"Uploading page {page_path.name} to OpenAI file storage.")
    with page_path.open("rb") as f:
        uploaded_file = openai_client.files.create(
            file=f,
            purpose="assistants",
        )

    logger.info(f"Page uploaded with file_id: {uploaded_file.id}.")
    return uploaded_file


async def upload_single_pages(
    folder: Path,
    openai_client: AzureOpenAI,
) -> None:
    """Upload single pages to the assistant."""
    # Define paths
    metadata = load_metadata(folder)

    for count, corpus_id in enumerate(metadata):
        # Skip if the corpus_id already has pages uploaded
        if "pages" in metadata[corpus_id]:
            logger.info(
                f"Skipping corpus_id {corpus_id} as it already has pages uploaded.",
            )
            continue

        # Checkpoint save every 5 files
        if count % 5 == 0:
            save_metadata(folder, metadata)

        # Open PDF to extract individual pages
        corpus_path = folder / metadata[corpus_id]["pdf_path"]
        if corpus_path.suffix == ".pdf":
            reader = PdfReader(corpus_path)
            tasks = []
            temp_paths = []
            for i, page in enumerate(
                reader.pages,
                start=1,
            ):  # Page numbers start from 1
                # Create temp pdf file for the page
                writer = PdfWriter()
                writer.add_page(page)

                temp_page_path = corpus_path.parent / f"{corpus_id}_p{i}.pdf"
                with temp_page_path.open("wb") as temp_pdf_file:
                    writer.write(temp_pdf_file)

                temp_paths.append(temp_page_path)
                tasks.append(upload_page(temp_page_path, openai_client))

            results = await asyncio.gather(*tasks)
            pages = {}
            for i, uploaded_file in enumerate(
                results,
                start=1,
            ):  # Page numbers start from 1
                pages[str(i)] = {
                    "openai_file_id": uploaded_file.id,
                }

            # Remove all temporary page files
            for temp_page_path in temp_paths:
                temp_page_path.unlink(missing_ok=True)

            # Store the file_id and name in the scores dictionary
            metadata[corpus_id]["pages"] = pages

    # Save the updated metadata with file IDs
    save_metadata(folder, metadata)
