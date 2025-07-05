import json
import logging
import time
from pathlib import Path

from langfuse import Langfuse
from loader import load_metadata
from openai import AzureOpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)


def review_answers(
    data_folder: Path,
    langfuse_client: Langfuse,
    openai_client: AzureOpenAI,
    model: str,
):
    # Load questions and answers
    questions_file = data_folder / "generated_data/generated_questions.json"
    answers_file = data_folder / "generated_data/generated_answers.json"

    # Load metadata
    metadata = load_metadata(data_folder)

    with questions_file.open() as f:
        questions = json.load(f)
    with answers_file.open() as f:
        answers = json.load(f)

    # Create the reviewer assistant
    assistant = openai_client.beta.assistants.create(
        name="Faithfulness Reviewer",
        instructions=(
            "You are an expert reviewer. Given a question, an answer, and a list of file_ids (representing the source documents/pages), "
            "your task is to generate a new answer that is more faithful and correct than the original, using only the information that could be found in the referenced file_ids. "
            "If the original answer is already maximally faithful and correct, you may return it unchanged. Otherwise, improve it. "
            "If the answer cannot be improved due to lack of information, state this clearly."
        ),
        model=model,
        tools=[{"type": "file_search"}],
    )

    # Get answer prompt from Langfuse client
    prompt = langfuse_client.get_prompt(
        "answer_review",
        label="production",
    )

    for question_type, questions_info in tqdm(questions.items()):
        for qid, qdata in tqdm(questions_info.items()):
            question = qdata["question"]
            answer_entry = answers[question_type][qid]
            answer = answer_entry["answer"]
            references = answer_entry["references"]

            file_ids = []
            for ref in references:
                if not isinstance(ref, (list | tuple)) or len(ref) < 2:
                    logger.warning(f"Invalid reference format: {ref}")
                    continue

                corpus_id, page_id = ref[0], ref[1]
                if corpus_id in metadata and "pages" in metadata[corpus_id]:
                    if str(page_id) in metadata[corpus_id]["pages"]:
                        file_ids.append(
                            metadata[corpus_id]["pages"][str(page_id)][
                                "openai_file_id"
                            ],
                        )
                    else:
                        logger.warning(
                            f"Page {page_id} not found in corpus {corpus_id}",
                        )
                else:
                    logger.warning(f"Corpus {corpus_id} not found in metadata")

            if not question or not answer:
                continue

            # Get the prompt from langfuse
            compiled_prompt = prompt.compile(
                question=question,
                answer=answer,
            )

            # Create a thread
            thread = openai_client.beta.threads.create()
            # Create message with file attachments
            message = openai_client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=compiled_prompt,
                attachments=[
                    {"file_id": file_id, "tools": [{"type": "file_search"}]}
                    for file_id in file_ids
                ]
                if file_ids
                else None,
            )
            # Run the assistant
            run = openai_client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
            )
            max_wait_time = 60
            start_time = time.time()
            while run.status in ["queued", "in_progress", "cancelling"]:
                if time.time() - start_time > max_wait_time:
                    logger.error(
                        f"Run for Q{qid} timed out after {max_wait_time} seconds.",
                    )
                    break
                time.sleep(1)
                run = openai_client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id,
                )

            reviewed_answer = None
            if run.status == "completed":
                messages = openai_client.beta.threads.messages.list(thread_id=thread.id)
                for message in messages:
                    if message.role == "assistant":
                        if message.content and len(message.content) > 0:
                            reviewed_answer = message.content[0].text.value
                        else:
                            logger.error(f"Empty message content for Q{qid}")
                            continue

                        answers[question_type][qid]["answer"] = reviewed_answer

                        break

            if not reviewed_answer:
                logger.error(f"Run for Q{qid} did not complete successfully.")

    with answers_file.open("w") as f:
        json.dump(answers, f, indent=2)

    logger.info(f"Review complete. Output written to {answers_file.name}")

    # Clean up the assistant
    openai_client.beta.assistants.delete(assistant_id=assistant.id)
    logger.info("Assistant cleaned up successfully")
