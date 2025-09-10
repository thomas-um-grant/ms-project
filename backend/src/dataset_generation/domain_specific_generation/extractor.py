import ast
import asyncio
import json
import logging
import re
import time
from pathlib import Path

import pymupdf
from langfuse import Langfuse
from loader import load_metadata, save_metadata
from openai import AzureOpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_raw_files(folder: Path) -> None:
    """Extract individual pdf files and save them with new ids."""
    metadata = {}
    # Create necessary directories
    (folder / "extracted_data/pdfs").mkdir(parents=True, exist_ok=True)

    for corpus_id, pdf_file in enumerate((folder / "raw_data").glob("*.pdf")):
        logger.info(f"Processing PDF file: {pdf_file.name}")

        try:
            document = pymupdf.open(pdf_file)
        except Exception as e:
            logger.exception(f"Failed to open PDF file {pdf_file.name}: {e}")
            continue

        # Store corpus_id (document name) and save the PDF with the new id
        pdf_path = f"extracted_data/pdfs/{corpus_id}.pdf"
        document.save(folder / pdf_path)
        document.close()

        # Store the raw document ids and extracted paths
        metadata[corpus_id] = {
            "pdf_path": pdf_path,
        }

        logger.info(f"Renamed {pdf_file.name} to {corpus_id}.pdf")

    # Save the raw documents ids to a JSON file
    save_metadata(folder, metadata)

    logger.info(f"Documents IDs saved to {folder / 'extracted_data/metadata.json'}")


def create_metadata(
    langfuse_client: Langfuse,
    openai_client: AzureOpenAI,
    folder: Path,
    model: str,
) -> None:
    # Get metadata prompt from Langfuse client
    prompt_metadata = langfuse_client.get_prompt(
        "page_metadata_prompt",
        label="production",
    )

    # Running list of tags
    tags: set[str] = set()

    assistant = openai_client.beta.assistants.create(
        name="PDF Document Analyzer",
        instructions=(
            "You are a helpful assistant that can analyze PDF documents. "
            "Use the file_search tool to find relevant information in uploaded PDFs. "
        ),
        model=model,
        tools=[{"type": "file_search"}],
    )

    # Define paths
    metadata = load_metadata(folder)

    logger.info("Loaded metadata")
    i = 0
    for corpus_id, corpus_metadata in tqdm(metadata.items()):
        if (
            "metadata" in corpus_metadata
            and (
                "topic" in corpus_metadata["metadata"]
                and "summary" in corpus_metadata["metadata"]
            )
            and (
                corpus_metadata["metadata"]["topic"] != ""
                and corpus_metadata["metadata"]["summary"] != ""
            )
        ):
            logger.info(f"Metadata already exists for corpus {corpus_id}. Skipping...")
            continue

        # Checkpoint in case of network failures or long processing times
        if i % 20 == 0:
            logger.info(f"Processed {i} documents so far...")
            save_metadata(folder, metadata)

        i += 1

        # Get the PDF file
        file_id = corpus_metadata["openai_file_id"]

        # Create a thread, and save its ID to the metadata
        thread = openai_client.beta.threads.create()
        logger.info(f"Created thread: {thread.id}")

        # Compile the prompt for metadata extraction
        prompt = prompt_metadata.compile(
            json_schema=prompt_metadata.config["json_schema"],
            tags=", ".join(tags),
        )

        # Create message with the PDF attached
        openai_client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt,
            attachments=[
                {"file_id": file_id, "tools": [{"type": "file_search"}]},
            ],
        )
        logger.info("Message created with PDF attachment")

        # Run the assistant
        run = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )
        logger.info(f"Run created: {run.id}")

        # Wait for completion
        json_response_failed = True
        retry_count = 0
        max_retries = 3

        while json_response_failed and retry_count < max_retries:
            while run.status in ["queued", "in_progress", "cancelling"]:
                time.sleep(1)
                run = openai_client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id,
                )
                logger.info(f"Run status: {run.status}")

            if run.status == "completed":
                logger.info(
                    f"Run completed for corpus {corpus_id}. Status: {run.status}",
                )
                # Get the assistant's response
                messages = openai_client.beta.threads.messages.list(
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
                                # Check if the response contains the required fields
                                if (
                                    "topic" in response_text
                                    and response_text["topic"] != ""
                                    and "summary" in response_text
                                    and response_text["summary"] != ""
                                    and "tags" in response_text
                                ):
                                    metadata[str(corpus_id)]["metadata"] = response_text
                                    logger.info(
                                        f"Metadata extracted for corpus {corpus_id}.",
                                    )

                                    tags.update(response_text.get("tags", []))
                                    json_response_failed = False
                                    break

                                logger.error(
                                    f"Invalid metadata format for corpus {corpus_id}. "
                                    "Expected fields 'topic' and 'summary' with non-empty values.",
                                )
                                break

                        except json.JSONDecodeError:
                            logger.exception(
                                f"JSON parsing error for corpus {corpus_id}.",
                            )
                            retry_count += 1

                            # Ask the assistant to fix the JSON format
                            openai_client.beta.threads.messages.create(
                                thread_id=thread.id,
                                role="user",
                                content="The previous response was not valid JSON. Please provide a valid JSON object with 'topic', 'summary', and 'tags' fields only. Do not include any other text, code blocks, or formatting.",
                            )

                            # Create a new run for the follow-up question
                            run = openai_client.beta.threads.runs.create(
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
    save_metadata(folder, metadata)

    # Clean up the assistant
    openai_client.beta.assistants.delete(assistant_id=assistant.id)
    logger.info("Assistant cleaned up successfully")


def extract_page_metadata(
    langfuse_client: Langfuse,
    openai_client: AzureOpenAI,
    model: str,
    file_id: str,
    page_id: str,
) -> tuple[str, dict]:
    """
    Sends a single-page PDF to the LLM via the Assistant Thread API and extracts metadata.
    Returns the metadata as a dict.
    """
    # Fetch prompt from langfuse client
    prompt = langfuse_client.get_prompt(
        "single_page_metadata_prompt",
        label="production",
    ).compile(
        json_schema=langfuse_client.get_prompt(
            "single_page_metadata_prompt",
            label="production",
        ).config["json_schema"],
    )

    # Create thread
    thread = openai_client.beta.threads.create()

    # Create assistant
    assistant = openai_client.beta.assistants.create(
        name="Single Page PDF Metadata Extractor",
        instructions=(
            "You are a helpful assistant that can analyze single-page PDF documents. "
            "Use the file_search tool to find relevant information requested by the user in the uploaded PDF. "
        ),
        model=model,
        tools=[{"type": "file_search"}],
    )

    # Send message with PDF attachment
    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
        attachments=[{"file_id": file_id, "tools": [{"type": "file_search"}]}],
    )
    # Run assistant
    run = openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    # Ensure json response is valid
    json_response_failed = True
    attempts = 5
    while json_response_failed and attempts > 0:
        max_wait_time = 60  # 1 minute max wait
        start_time = time.time()
        # Wait for completion
        while run.status in ["queued", "in_progress", "cancelling"]:
            if time.time() - start_time > max_wait_time:
                logger.error(
                    f"Run for file {file_id} timed out after {max_wait_time} seconds.",
                )
                break
            time.sleep(1)
            run = openai_client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            logger.info(f"Run status: {run.status}")

        if run.status == "completed":
            # Get the assistant's response
            messages = openai_client.beta.threads.messages.list(
                thread_id=thread.id,
            )

            # Get the latest assistant message
            response = None
            for message in messages:
                if message.role == "assistant":
                    if message.content and len(message.content) > 0:
                        response = message.content[0].text.value
                    else:
                        logger.error(f"Empty message content for file {file_id}")
                        continue

                    try:
                        json_match = re.search(
                            r"```json\s*([\s\S]*?)\s*```",
                            response,
                        )
                        if json_match:
                            # Extract just the JSON part
                            response_text = json_match.group(1).strip()
                            parsed_response = json.loads(response_text)
                        elif response.startswith("{") and response.endswith("}"):
                            parsed_response = ast.literal_eval(response)
                        else:
                            response_text = response
                            parsed_response = json.loads(response_text)

                        if (
                            isinstance(parsed_response, dict)
                            and "topic" in parsed_response
                            and parsed_response["topic"] != ""
                            and "summary" in parsed_response
                            and parsed_response["summary"] != ""
                        ):
                            # Clean up resources before returning
                            openai_client.beta.assistants.delete(
                                assistant_id=assistant.id,
                            )
                            openai_client.beta.threads.delete(thread_id=thread.id)

                            return (page_id, parsed_response)
                        # If the response is not valid JSON, ask the assistant to fix it
                        openai_client.beta.threads.messages.create(
                            thread_id=thread.id,
                            role="user",
                            content="Please, ensure the response is in a valid JSON format. Return ONLY that json object.",
                        )

                        run = openai_client.beta.threads.runs.create(
                            thread_id=thread.id,
                            assistant_id=assistant.id,
                        )

                        attempts -= 1

                    except Exception:
                        logger.exception(
                            f"Unexpected error for file {file_id}: Retrying",
                        )
                        attempts -= 1

                    break
        else:
            logger.error(
                f"Run failed for file {file_id}. Status: {run.status}",
            )
            break

    # Clean up resources
    openai_client.beta.assistants.delete(assistant_id=assistant.id)
    openai_client.beta.threads.delete(thread_id=thread.id)

    return (page_id, {"topic": "", "summary": ""})


async def extract_pages_info_from_document(
    data_folder: Path,
    corpus_id: str,
    langfuse_client: Langfuse,
    openai_client: AzureOpenAI,
    model: str,
) -> list[dict]:
    """
    Extracts pages from a document.
    Returns a list of pages.
    """
    # Load metadata
    metadata = load_metadata(data_folder)
    pages = metadata[corpus_id]["pages"]

    tasks = []
    for page_id, page in pages.items():  # Page numbers start from 1
        if (
            "topic" in page
            and page["topic"] != ""
            and "summary" in page
            and page["summary"] != ""
        ):
            # Skip pages that already have metadata
            continue

        tasks.append(
            asyncio.to_thread(
                extract_page_metadata,
                langfuse_client,
                openai_client,
                model,
                page["openai_file_id"],
                page_id,
            ),
        )

    if tasks:
        # Run all tasks concurrently
        logger.info(f"Extracting metadata for {len(tasks)} pages in corpus {corpus_id}")
        results = await asyncio.gather(*tasks)
        for page_id, page_metadata in results:
            pages[page_id]["topic"] = page_metadata["topic"]
            pages[page_id]["summary"] = page_metadata["summary"]

    return pages


async def extract_metadata_pages_from_dataset(
    langfuse_client: Langfuse,
    openai_client: AzureOpenAI,
    model: str,
    data_folder: Path,
):
    # Load metadata of extracted files
    metadata = load_metadata(data_folder)

    # Prepare tasks for all corpus_ids that need page extraction
    for corpus_id in metadata:
        metadata[corpus_id]["pages"] = await extract_pages_info_from_document(
            data_folder,
            corpus_id,
            langfuse_client,
            openai_client,
            model,
        )

        # Save updated metadata
        save_metadata(data_folder, metadata)

        logger.info(
            f"Extracted metadata for corpus {corpus_id} with {len(metadata[corpus_id]['pages'])} pages.",
        )
