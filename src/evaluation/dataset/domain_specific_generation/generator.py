import json
import logging
import random
import re
import time
from pathlib import Path

from langfuse import Langfuse
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


def generate_questions(
    lang_fuse_client: Langfuse,
    open_ai_client: AzureOpenAI,
    folder: Path,
    model: str,
) -> None:
    # Maps prompt to question types
    prompt_to_answer = {
        "recency-bias": {"prompt_name": "evaluation_question_recency_bias"},
        "single-fact": {"prompt_name": "evaluation_question_single_fact"},
        "single-slide": {"prompt_name": "evaluation_question_single_slide"},
        "multi-slide": {"prompt_name": "evaluation_question_multi_slide"},
        "aggregation": {"prompt_name": "evaluation_question_aggregation"},
        "top-level-strategic": {
            "prompt_name": "evaluation_question_top_level_strategic",
        },
    }

    generated_questions: dict = {}

    for question_type, prompt_info in prompt_to_answer.items():
        # Get metadata prompt from Langfuse client
        prompt = lang_fuse_client.get_prompt(
            prompt_info["prompt_name"],
            label="production",
        )
        prompt_to_answer[question_type]["prompt"] = prompt

    assistant = open_ai_client.beta.assistants.create(
        name="PDF Document Analyzer",
        instructions=(
            "You are a helpful assistant that can generate high quality questions based on PDF files. "
            "Use the file_search tool to find relevant information in uploaded PDFs. "
            "Extract key information based on the type of questions requested and generate a high quality question based on the documents content. "
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
    metadata = {}
    with metadata_file.open() as f:
        metadata = json.load(f)

    # Question assignments {qtype: [[[corpus_ids], [file_ids]], ...]}
    recency_bias_questions: list = random.sample(list(metadata.keys()), 50)
    single_fact_questions: list = random.sample(list(metadata.keys()), 50)
    single_slide_questions: list = random.sample(list(metadata.keys()), 50)

    # Use the metadata to assign questions, randomly for now
    question_assignments: dict = {
        "recency-bias": [
            ([corpus_id], [metadata[corpus_id]["openai_file_id"]])
            for corpus_id in recency_bias_questions
        ],
        "single-fact": [
            ([corpus_id], [metadata[corpus_id]["openai_file_id"]])
            for corpus_id in single_fact_questions
        ],
        "single-slide": [
            ([corpus_id], [metadata[corpus_id]["openai_file_id"]])
            for corpus_id in single_slide_questions
        ],
        "multi-slide": [],
        "aggregation": [],
        "top-level-strategic": [],
    }

    # Create multi-file sets for each question type: pick 3 files at random in the metadata, and put them together
    corpus_ids = list(metadata.keys())
    # Cannot have duplicates of tuples of document ids within a set
    set_multi_slide: set = set()
    set_aggregation: set = set()
    set_top_level_strategic: set = set()

    # Do num_questions multi-file questions for each question type
    num_questions = 50
    for question_set in [set_multi_slide, set_aggregation, set_top_level_strategic]:
        i = 0
        failsafe_attempts = 500
        while i < num_questions and failsafe_attempts > 0:
            # Randomly select 3 corpus ids for multi-slide, aggregation, and top-level-strategic questions
            selected_corpus_ids = random.sample(corpus_ids, 3)  # 3 unique corpus ids
            # Try to add them to the set, if they are not already present
            if set(selected_corpus_ids) not in question_set:
                question_set.add(tuple(selected_corpus_ids))
                i += 1

            failsafe_attempts -= 1

    question_assignments["multi-slide"] = [
        [list(current_set), [metadata[cid]["openai_file_id"] for cid in current_set]]
        for current_set in set_multi_slide
    ]
    question_assignments["aggregation"] = [
        [list(current_set), [metadata[cid]["openai_file_id"] for cid in current_set]]
        for current_set in set_aggregation
    ]
    question_assignments["top-level-strategic"] = [
        [list(current_set), [metadata[cid]["openai_file_id"] for cid in current_set]]
        for current_set in set_top_level_strategic
    ]

    logger.info(f"Question assignments created: {question_assignments}")

    qid = 0  # Question ID counter

    # {question_type: [[[corpus_ids], [file_ids]], ...]}
    for question_type, questions in question_assignments.items():
        # Question types
        single_file_questions = ["recency-bias", "single-fact", "single-slide"]
        multi_file_questions = ["multi-slide", "aggregation", "top-level-strategic"]

        generated_questions[
            question_type
        ] = {}  # {{question_id: { files: [[corpus_ids], [file_ids]] }, question: str}, ...}

        # Get file ids for each questions
        if question_type in single_file_questions:
            # questions = [([corpus_ids], [file_ids])]
            logger.info(
                f"Generating {question_type} questions for {len(questions)} files...",
            )
            for i, question_info in enumerate(questions):
                corpus_id = question_info[0][0]

                compiled_prompt = prompt_to_answer[question_type]["prompt"].compile(
                    topic=metadata[corpus_id]["metadata"]["topic"],
                    summary=metadata[corpus_id]["metadata"]["summary"],
                )

                print(
                    f"Generating {question_type} question {i + 1}/{len(questions)}...",
                )

                # Create a thread
                thread = open_ai_client.beta.threads.create()
                print(f"Created thread: {thread.id}")

                # Create message with the PDF attached
                message = open_ai_client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=compiled_prompt,
                )

                # Run the assistant
                run = open_ai_client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant.id,
                )

                max_wait_time = 60  # 1 minute max wait
                start_time = time.time()
                # Wait for completion
                while run.status in ["queued", "in_progress", "cancelling"]:
                    if time.time() - start_time > max_wait_time:
                        logger.error(
                            f"Run for corpus {corpus_id} timed out after {max_wait_time} seconds.",
                        )
                        break
                    time.sleep(1)
                    run = open_ai_client.beta.threads.runs.retrieve(
                        thread_id=thread.id,
                        run_id=run.id,
                    )
                    print(f"Run status: {run.status}")

                if run.status == "completed":
                    # Get the assistant's response
                    messages = open_ai_client.beta.threads.messages.list(
                        thread_id=thread.id,
                    )

                    # Get the latest assistant message
                    response = None
                    for message in messages:
                        if message.role == "assistant":
                            try:
                                response = message.content[0].text.value

                                # {{question_id: { files: [[corpus_ids], [file_ids]] }, question: str}, ...}
                                generated_questions[question_type][str(qid)] = {
                                    "files": question_info,
                                    "question": response,
                                }

                                qid += 1

                                logger.info(
                                    f"Metadata extracted for corpus {corpus_id}.",
                                )
                            except Exception:
                                logger.exception(
                                    f"Run failed for corpus {corpus_id}. Status: {run.status}",
                                )
                                continue
                            else:
                                break

        if question_type in multi_file_questions:
            logger.info(
                f"Generating {question_type} questions for {len(questions)} files...",
            )
            # questions = [([corpus_ids], [file_ids])]
            for i, question_info in enumerate(questions):
                corpus_ids: list = question_info[0]

                topics_and_summaries: str = "\n".join(
                    f"""
                    topic {topic_id}: {metadata[cid]["metadata"]["topic"]}
                    summary {topic_id}: {metadata[cid]["metadata"]["summary"]}
                    """
                    for topic_id, cid in enumerate(corpus_ids)
                )

                compiled_prompt = prompt_to_answer[question_type]["prompt"].compile(
                    topics_and_summaries=topics_and_summaries,
                )

                print(
                    f"Generating {question_type} question {i + 1}/{len(questions)}...",
                )

                # Create a thread
                thread = open_ai_client.beta.threads.create()
                print(f"Created thread: {thread.id}")

                # Create message with the PDF attached
                message = open_ai_client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=compiled_prompt,
                )

                # Run the assistant
                run = open_ai_client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant.id,
                )

                max_wait_time = 60  # 1 minute max wait
                start_time = time.time()
                # Wait for completion
                while run.status in ["queued", "in_progress", "cancelling"]:
                    if time.time() - start_time > max_wait_time:
                        logger.error(
                            f"Run for corpus {corpus_id} timed out after {max_wait_time} seconds.",
                        )
                        break
                    time.sleep(1)
                    run = open_ai_client.beta.threads.runs.retrieve(
                        thread_id=thread.id,
                        run_id=run.id,
                    )
                    print(f"Run status: {run.status}")

                if run.status == "completed":
                    # Get the assistant's response
                    messages = open_ai_client.beta.threads.messages.list(
                        thread_id=thread.id,
                    )

                    # Get the latest assistant message
                    response = None
                    for message in messages:
                        if message.role == "assistant":
                            try:
                                response = message.content[0].text.value

                                # {{question_id: { files: [[corpus_ids], [file_ids]] }, question: str}, ...}
                                generated_questions[question_type][str(qid)] = {
                                    "files": question_info,
                                    "question": response,
                                }

                                qid += 1

                                logger.info(
                                    f"Metadata extracted for corpus {corpus_id}.",
                                )
                            except Exception:
                                logger.exception(
                                    f"Run failed for corpus {corpus_id}. Status: {run.status}",
                                )
                                continue
                            else:
                                break

    # Save the generated questions to a JSON file
    generated_questions_file = folder / "generated_data/generated_questions.json"
    with generated_questions_file.open("w") as f:
        json.dump(generated_questions, f, indent=2)


def generate_answers(
    lang_fuse_client: Langfuse,
    open_ai_client: AzureOpenAI,
    folder: Path,
    model: str,
) -> None:
    generated_answers: dict = {}

    # Get answer prompt from Langfuse client
    prompt_to_answer = lang_fuse_client.get_prompt(
        "evaluation_answer",
        label="production",
    )

    assistant = open_ai_client.beta.assistants.create(
        name="PDF Document Analyzer and Question Answerer",
        instructions=(
            "You are a helpful assistant that can generate high quality answer based on a question and PDF files. "
            "Use the file_search tool to find relevant information in uploaded PDFs. "
            "Extract key information based on the type of questions requested and generate a high quality answer based on the documents content. "
        ),
        model=model,
        tools=[{"type": "file_search"}],
    )

    # Load question assignments
    questions_file = folder / "generated_data/generated_questions.json"

    if not questions_file.exists():
        logger.error(f"Extracted files IDs not found: {questions_file}")
        return

    question_assignments: dict = {}
    with questions_file.open() as f:
        question_assignments = json.load(f)

    for question_type, questions in question_assignments.items():
        # Question types
        single_file_questions = ["recency-bias", "single-fact", "single-slide"]
        multi_file_questions = ["multi-slide", "aggregation", "top-level-strategic"]

        generated_answers[
            question_type
        ] = {}  # {question_id: int {answer:  str, references: list[(document_name, page_number)]}}

        # Get file ids for each questions
        if question_type in single_file_questions:
            # questions = [{files: ([corpus_ids], [file_ids]), question: str}, ...]
            for qid, question_info in questions.items():
                corpus_id = question_info["files"][0][0]
                file_id = question_info["files"][1][0]
                compiled_prompt = prompt_to_answer.compile(
                    question=question_info["question"],
                )

                print(
                    f"Generating answer for {question_type} question with id {qid}...",
                )

                # Create a thread
                thread = open_ai_client.beta.threads.create()
                print(f"Created thread: {thread.id}")

                # Create message with the PDF attached
                message = open_ai_client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=compiled_prompt,
                    attachments=[
                        {"file_id": file_id, "tools": [{"type": "file_search"}]},
                    ],
                )

                # Run the assistant
                run = open_ai_client.beta.threads.runs.create(
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
                                f"Run for corpus {corpus_id} timed out after {max_wait_time} seconds.",
                            )
                            break
                        time.sleep(1)
                        run = open_ai_client.beta.threads.runs.retrieve(
                            thread_id=thread.id,
                            run_id=run.id,
                        )
                        print(f"Run status: {run.status}")

                    if run.status == "completed":
                        # Get the assistant's response
                        messages = open_ai_client.beta.threads.messages.list(
                            thread_id=thread.id,
                        )

                        # Get the latest assistant message
                        response = None
                        for message in messages:
                            if message.role == "assistant":
                                response = message.content[0].text.value

                                try:
                                    json_match = re.search(
                                        r"```json\s*([\s\S]*?)\s*```",
                                        response,
                                    )
                                    if json_match:
                                        # Extract just the JSON part
                                        response_text = json_match.group(1).strip()
                                    else:
                                        response_text = response

                                    parsed_response = json.loads(response_text)

                                    if isinstance(parsed_response, (dict)):
                                        # {question_id: int {answer:  str, references: list[(document_name, page_number)]}}
                                        generated_answers[question_type][qid] = {
                                            "answer": parsed_response["answer"],
                                            "references": parsed_response["references"],
                                        }

                                        json_response_failed = False
                                    else:
                                        # If the response is not valid JSON, ask the assistant to fix it
                                        open_ai_client.beta.threads.messages.create(
                                            thread_id=thread.id,
                                            role="user",
                                            content="Please, ensure the response is in a valid JSON format. Return ONLY that json object.",
                                        )

                                        run = open_ai_client.beta.threads.runs.retrieve(
                                            thread_id=thread.id,
                                            run_id=run.id,
                                        )

                                        attempts -= 1

                                except Exception:
                                    logger.exception(
                                        f"Couldn't extract json object for corpus {corpus_id}: Retrying",
                                    )
                                    attempts -= 1

                                break
                    else:
                        logger.error(
                            f"Run failed for corpus {corpus_id}. Status: {run.status}",
                        )
                        break

                if run.status not in [
                    "queued",
                    "in_progress",
                    "cancelling",
                    "completed",
                ]:
                    logger.error(
                        f"Skipping corpus {corpus_id}.",
                    )
                    continue

        if question_type in multi_file_questions:
            # questions = [{files: ([corpus_ids], [file_ids]), question: str}, ...]
            for qid, question_info in questions.items():
                file_ids: list = question_info["files"][1]

                compiled_prompt = prompt_to_answer.compile(
                    question=question_info["question"],
                )

                print(
                    f"Generating answer for {question_type} question with id {qid}...",
                )

                # Create a thread
                thread = open_ai_client.beta.threads.create()
                print(f"Created thread: {thread.id}")

                # Create message with the PDF attached
                message = open_ai_client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=compiled_prompt,
                    attachments=[
                        {"file_id": file_id, "tools": [{"type": "file_search"}]}
                        for file_id in file_ids
                    ],
                )

                # Run the assistant
                run = open_ai_client.beta.threads.runs.create(
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
                                f"Run for corpus {corpus_id} timed out after {max_wait_time} seconds.",
                            )
                            break
                        time.sleep(1)
                        run = open_ai_client.beta.threads.runs.retrieve(
                            thread_id=thread.id,
                            run_id=run.id,
                        )
                        print(f"Run status: {run.status}")

                    if run.status == "completed":
                        # Get the assistant's response
                        messages = open_ai_client.beta.threads.messages.list(
                            thread_id=thread.id,
                        )

                        # Get the latest assistant message
                        response = None
                        for message in messages:
                            if message.role == "assistant":
                                response = message.content[0].text.value

                                try:
                                    json_match = re.search(
                                        r"```json\s*([\s\S]*?)\s*```",
                                        response,
                                    )
                                    if json_match:
                                        # Extract just the JSON part
                                        response_text = json_match.group(1).strip()
                                    else:
                                        response_text = response

                                    parsed_response = json.loads(response_text)

                                    if isinstance(parsed_response, (dict)):
                                        # {question_id: int {answer:  str, references: list[(document_name, page_number)]}}
                                        generated_answers[question_type][qid] = {
                                            "answer": parsed_response["answer"],
                                            "references": parsed_response["references"],
                                        }

                                        json_response_failed = False
                                    else:
                                        # If the response is not valid JSON, ask the assistant to fix it
                                        open_ai_client.beta.threads.messages.create(
                                            thread_id=thread.id,
                                            role="user",
                                            content="Please, ensure the response is in a valid JSON format. Return ONLY that json object.",
                                        )

                                        run = open_ai_client.beta.threads.runs.retrieve(
                                            thread_id=thread.id,
                                            run_id=run.id,
                                        )

                                        attempts -= 1

                                except Exception:
                                    logger.exception(
                                        f"Unexpected error for corpus {corpus_id}: Retrying",
                                    )
                                    attempts -= 1

                                break
                    else:
                        logger.error(
                            f"Run failed for corpus {[question_info['files'][i][0] for i in range(len(question_info['files']))]}. Status: {run.status}",
                        )
                        break

    # Save the generated questions to a JSON file
    generated_questions_file = folder / "generated_data/generated_answers.json"
    with generated_questions_file.open("w") as f:
        json.dump(generated_answers, f, indent=2)
