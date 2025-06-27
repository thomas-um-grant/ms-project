import json
import logging
import re
import time
from pathlib import Path

import pymupdf
from langfuse import Langfuse
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


def extract_raw_files(folder: Path) -> None:
    """Extract individual pdf files and save them with new ids."""
    metadata = {}
    for corpus_id, pdf_file in enumerate((folder / "raw_data").glob("*.pdf")):
        logger.info(f"Processing PDF file: {pdf_file.name}")

        try:
            document = pymupdf.open(pdf_file)
        except Exception as e:
            logger.exception(f"Failed to open PDF file {pdf_file.name}: {e}")
            continue

        # Store corpus_id (document name) and save the PDF with the new id
        pdf_path = folder / f"extracted_data/pdfs/{corpus_id}.pdf"
        document.save(pdf_path)

        # Store the raw document ids and extracted paths
        metadata[corpus_id] = {
            "pdf_path": str(pdf_path),
        }

        logger.info(f"Renamed {pdf_file.name} to {corpus_id}.pdf")

    # Save the raw documents ids to a JSON file
    output_file = folder / "extracted_data/metadata.json"
    with output_file.open("w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Documents IDs saved to {output_file}")


def create_metadata(
    lang_fuse_client: Langfuse,
    open_ai_client: AzureOpenAI,
    folder: Path,
    model: str,
) -> None:
    # Get metadata prompt from Langfuse client
    prompt_metadata = lang_fuse_client.get_prompt(
        "page_metadata_prompt",
        label="production",
    )

    # Compile the prompt for metadata extraction
    prompt = prompt_metadata.compile(json_schema=prompt_metadata.config["json_schema"])

    assistant = open_ai_client.beta.assistants.create(
        name="PDF Document Analyzer",
        instructions=(
            "You are a helpful assistant that can analyze PDF documents. "
            "Use the file_search tool to find relevant information in uploaded PDFs. "
        ),
        model=model,
        tools=[{"type": "file_search"}],
    )

    # Define paths
    metadata_file = folder / "extracted_data/metadata.json"

    if not metadata_file.exists():
        logger.error(f"Extracted files IDs not found: {metadata_file}")
        return

    # Load metadata of extracted files
    with metadata_file.open() as f:
        metadata = json.load(f)

    logger.info("Loaded metadata")
    for corpus_id, corpus_metadata in metadata.items():
        # Get the PDF file
        file_id = corpus_metadata["openai_file_id"]

        # Create a thread, and save its ID to the metadata
        thread = open_ai_client.beta.threads.create()
        print(f"Created thread: {thread.id}")

        # Create message with the PDF attached
        open_ai_client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt,
            attachments=[
                {"file_id": file_id, "tools": [{"type": "file_search"}]},
            ],
        )
        print("Message created with PDF attachment")

        # Run the assistant
        run = open_ai_client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )
        print(f"Run created: {run.id}")

        # Wait for completion
        json_response_failed = True
        retry_count = 0
        max_retries = 3

        while json_response_failed and retry_count < max_retries:
            while run.status in ["queued", "in_progress", "cancelling"]:
                time.sleep(1)
                run = open_ai_client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id,
                )
                print(f"Run status: {run.status}")

            if run.status == "completed":
                logger.info(
                    f"Run completed for corpus {corpus_id}. Status: {run.status}",
                )
                # Get the assistant's response
                messages = open_ai_client.beta.threads.messages.list(
                    thread_id=thread.id,
                )

                # Get the latest assistant message
                response = None
                for message in messages:
                    if message.role == "assistant":
                        # Check if response exists and has content
                        if message.content and len(message.content) > 0:
                            content_item = message.content[0]
                            if hasattr(content_item, "text") and hasattr(
                                content_item.text,
                                "value",
                            ):
                                response = content_item.text.value

                        # Log the raw response for debugging
                        logger.info(f"Raw response for corpus {corpus_id}: {response}")

                        if not response:
                            logger.error("Empty or invalid response received")
                            break

                        try:
                            # Look for JSON-like patterns
                            json_match = re.search(
                                r"```json\s*([\s\S]*?)\s*```",
                                response,
                            )
                            if json_match:
                                # Extract just the JSON part
                                response = json_match.group(1).strip()

                            # Try to parse the response as JSON
                            response_text = json.loads(response)

                            if isinstance(response_text, (dict)):
                                metadata[str(corpus_id)]["metadata"] = response_text
                                logger.info(
                                    f"Metadata extracted for corpus {corpus_id}.",
                                )
                                json_response_failed = False
                                break

                        except json.JSONDecodeError:
                            logger.exception(
                                f"JSON parsing error for corpus {corpus_id}.",
                            )
                            retry_count += 1

                            # Ask the assistant to fix the JSON format
                            open_ai_client.beta.threads.messages.create(
                                thread_id=thread.id,
                                role="user",
                                content="The previous response was not valid JSON. Please provide a valid JSON object with 'topic' and 'summary' fields only. Do not include any other text, code blocks, or formatting.",
                            )

                            # Create a new run for the follow-up question
                            run = open_ai_client.beta.threads.runs.create(
                                thread_id=thread.id,
                                assistant_id=assistant.id,
                            )
                        except Exception:
                            logger.exception(
                                f"Unexpected error for corpus {corpus_id}. Retrying",
                            )
                        break
            else:
                logger.error(
                    f"Run failed for corpus {corpus_id}. Status: {run.status}",
                )
                retry_count += 1
                break

        if json_response_failed:
            logger.warning(
                f"Could not extract valid JSON metadata for corpus {corpus_id} after {max_retries} attempts",
            )

    # Save the updated metadata with extracted information
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
