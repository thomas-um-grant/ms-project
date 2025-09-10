import json
import logging
import random
import time
from pathlib import Path

from langfuse import Langfuse
from loader import load_metadata
from openai import AzureOpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_assistant_thread(openai_client, assistant_id, compiled_prompt, file_ids=None):
    """
    Helper to create a thread, send a message, run the assistant, poll for completion, and return the run and thread.
    """
    thread = openai_client.beta.threads.create()
    logger.info(f"Created thread: {thread.id}")

    message_kwargs = {
        "thread_id": thread.id,
        "role": "user",
        "content": compiled_prompt,
    }
    if file_ids:
        message_kwargs["attachments"] = [
            {"file_id": file_id, "tools": [{"type": "file_search"}]}
            for file_id in file_ids
        ]
    openai_client.beta.threads.messages.create(**message_kwargs)

    run = openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

    max_wait_time = 60  # 1 minute max wait
    start_time = time.time()
    while run.status in ["queued", "in_progress", "cancelling"]:
        if time.time() - start_time > max_wait_time:
            logger.error(f"Run timed out after {max_wait_time} seconds.")
            break
        time.sleep(1)
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        logger.info(f"Run status: {run.status}")
    return run, thread


def parse_assistant_response(messages):
    """Extract the latest assistant message content from a list of messages."""
    for message in messages:
        if message.role == "assistant":
            if message.content and len(message.content) > 0:
                return message.content[0].text.value
            logger.error("Empty message content in assistant response.")
            return None
    logger.error("No assistant message found in thread.")
    return None


def generate_single_file_questions(
    openai_client,
    assistant,
    prompt_to_answer,
    metadata,
    question_type,
    questions,
    generated_questions,
    qid,
    generated_questions_file,
):
    logger.info(
        f"Generating {question_type} questions for {len(questions)} files...",
    )
    for i, question_info in tqdm(enumerate(questions)):
        if qid % 20 == 0:
            with generated_questions_file.open("w") as f:
                json.dump(generated_questions, f, indent=2)

        corpus_id = question_info[0][0]
        compiled_prompt = prompt_to_answer[question_type]["prompt"].compile(
            topic=metadata[corpus_id]["metadata"]["topic"],
            summary=metadata[corpus_id]["metadata"]["summary"],
        )

        logger.info(
            f"Generating {question_type} question {i + 1}/{len(questions)}...",
        )

        run, thread = run_assistant_thread(openai_client, assistant.id, compiled_prompt)

        if run.status == "completed":
            messages = openai_client.beta.threads.messages.list(thread_id=thread.id)
            response = parse_assistant_response(messages)
            if response is not None:
                generated_questions[question_type][str(qid)] = {
                    "files": question_info,
                    "question": response,
                }
                logger.info(f"Question generated for {question_type} with id {qid}.")
                qid += 1
        else:
            logger.error(f"Run failed for corpus {corpus_id}. Status: {run.status}")
    return qid


def generate_multi_file_questions(
    openai_client,
    assistant,
    prompt_to_answer,
    metadata,
    question_type,
    questions,
    generated_questions,
    qid,
    generated_questions_file,
):
    logger.info(
        f"Generating {question_type} questions for {len(questions)} files...",
    )
    for i, question_info in tqdm(enumerate(questions)):
        if qid % 20 == 0:
            with generated_questions_file.open("w") as f:
                json.dump(generated_questions, f, indent=2)

        corpus_ids = question_info[0]
        topics_and_summaries = "\n".join(
            f"""
            topic {topic_id}: {metadata[cid]["metadata"]["topic"]}
            summary {topic_id}: {metadata[cid]["metadata"]["summary"]}
            """
            for topic_id, cid in enumerate(corpus_ids)
        )
        compiled_prompt = prompt_to_answer[question_type]["prompt"].compile(
            topics_and_summaries=topics_and_summaries,
        )

        logger.info(
            f"Generating {question_type} question {i + 1}/{len(questions)}...",
        )

        run, thread = run_assistant_thread(openai_client, assistant.id, compiled_prompt)

        if run.status == "completed":
            messages = openai_client.beta.threads.messages.list(thread_id=thread.id)
            response = parse_assistant_response(messages)
            if response is not None:
                generated_questions[question_type][str(qid)] = {
                    "files": question_info,
                    "question": response,
                }
                logger.info(f"Question generated for {question_type} with id {qid}.")
                qid += 1
        else:
            logger.error(f"Run failed for corpus {corpus_ids}. Status: {run.status}")
    return qid


def generate_questions(
    langfuse_client: Langfuse,
    openai_client: AzureOpenAI,
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
    generated_questions_file = folder / "generated_data/generated_questions.json"

    for prompt_info in prompt_to_answer.values():
        # Get metadata prompt from Langfuse client
        prompt = langfuse_client.get_prompt(
            prompt_info["prompt_name"],
            label="production",
        )
        prompt_info["prompt"] = prompt

    assistant = openai_client.beta.assistants.create(
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
    metadata = load_metadata(folder)

    # Question assignments {qtype: [[[corpus_ids], [file_ids]], ...]}
    available_keys = list(metadata.keys())
    sample_size = min(50, len(available_keys))
    recency_bias_questions: list = random.sample(available_keys, sample_size)
    single_fact_questions: list = random.sample(available_keys, sample_size)
    single_slide_questions: list = random.sample(available_keys, sample_size)

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

    # Load the cluster keys to know which corpus ids to use together for multiple file questions
    cluster_keys_file = folder / "generated_data/corpuses_clusters.json"
    if not cluster_keys_file.exists():
        logger.error(f"Cluster keys not found: {cluster_keys_file}")
        return

    cluster_keys: dict = {}
    with cluster_keys_file.open() as f:
        cluster_keys = json.load(f)

    # Create multi-file sets for each question type:
    # Pick 3 files at random from one of the clusters, and put them together
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
            selected_cluster = random.choice(list(cluster_keys.items()))

            corpus_ids = selected_cluster[1]["corpus_ids"]
            if len(corpus_ids) < 3:
                logger.warning(
                    f"Cluster has only {len(corpus_ids)} corpus IDs, skipping",
                )
                continue
            selected_corpus_ids = random.sample(corpus_ids, 3)

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
    single_file_questions = ["recency-bias", "single-fact", "single-slide"]
    multi_file_questions = ["multi-slide", "aggregation", "top-level-strategic"]

    for question_type, questions in tqdm(question_assignments.items()):
        generated_questions[question_type] = {}
        if question_type in single_file_questions:
            qid = generate_single_file_questions(
                openai_client,
                assistant,
                prompt_to_answer,
                metadata,
                question_type,
                questions,
                generated_questions,
                qid,
                generated_questions_file,
            )
        elif question_type in multi_file_questions:
            qid = generate_multi_file_questions(
                openai_client,
                assistant,
                prompt_to_answer,
                metadata,
                question_type,
                questions,
                generated_questions,
                qid,
                generated_questions_file,
            )

    # Create a folder for generated data if it doesn't exist
    (folder / "generated_data").mkdir(parents=True, exist_ok=True)

    # Save the generated questions to a JSON file
    with generated_questions_file.open("w") as f:
        json.dump(generated_questions, f, indent=2)


def generate_answers(
    langfuse_client: Langfuse,
    openai_client: AzureOpenAI,
    folder: Path,
    model: str,
) -> None:
    generated_answers: dict = {}
    generated_questions_file = folder / "generated_data/generated_answers.json"

    # Get answer prompt from Langfuse client
    prompt_to_answer = langfuse_client.get_prompt(
        "evaluation_answer",
        label="production",
    )

    assistant = openai_client.beta.assistants.create(
        name="PDF Document Analyzer and Question Answerer",
        instructions=(
            "You are a helpful assistant that can generate high quality answer based on a question and PDF files. "
            "Use the file_search tool to find relevant information in uploaded PDFs. "
            "Extract key information based on the type of questions requested and generate a high quality answer based on the documents content. "
        ),
        model=model,
        tools=[{"type": "file_search"}],
    )

    # Load metadata of extracted files
    metadata = load_metadata(folder)

    # Load question assignments
    questions_file = folder / "generated_data/generated_questions.json"

    if not questions_file.exists():
        logger.error(f"Extracted files IDs not found: {questions_file}")
        return

    question_assignments: dict = {}
    with questions_file.open() as f:
        question_assignments = json.load(f)

    for question_type, questions in tqdm(question_assignments.items()):
        # Question types
        single_file_questions = ["recency-bias", "single-fact", "single-slide"]
        multi_file_questions = ["multi-slide", "aggregation", "top-level-strategic"]

        generated_answers[
            question_type
        ] = {}  # {question_id: int {answer:  str, references: list[(document_name, page_number)]}}

        # Get file ids for each questions
        if question_type in single_file_questions:
            # questions = [{files: ([corpus_ids], [file_ids]), question: str}, ...]
            for qid, question_info in tqdm(questions.items()):
                if qid in generated_answers[question_type]:
                    continue  # Skip if already answered

                # Update question assignments with pages to answer and save answers generated so far
                if int(qid) % 10 == 0:
                    with generated_questions_file.open("w") as f:
                        json.dump(generated_answers, f, indent=2)

                corpus_id = question_info["files"][0][0]
                compiled_prompt = prompt_to_answer.compile(
                    question=question_info["question"],
                )

                # Get page ids to answer the question
                openai_file_ids = []
                references = [
                    (ref["corpus_id"], ref["page_id"])
                    for ref in question_info["files"][2]
                ]
                for corpus_id, page_id in references:
                    # Get the openai_file_id for each page in the corpus
                    openai_file_ids.append(
                        metadata[corpus_id]["pages"][page_id]["openai_file_id"],
                    )

                logger.info(
                    f"Generating answer for {question_type} question with id {qid} using references {openai_file_ids}...",
                )

                # Create a thread
                thread = openai_client.beta.threads.create()
                logger.info(f"Created thread: {thread.id}")

                # Create message with the PDF attached
                message = openai_client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=compiled_prompt,
                    attachments=[
                        {"file_id": file_id, "tools": [{"type": "file_search"}]}
                        for file_id in openai_file_ids
                    ],
                )

                # Run the assistant
                run = openai_client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant.id,
                )

                max_wait_time = 60  # 1 minute max wait
                start_time = time.time()
                # Wait for completion
                while run.status in ["queued", "in_progress", "cancelling"]:
                    if time.time() - start_time > max_wait_time:
                        logger.error(
                            f"Run for question {qid} timed out after {max_wait_time} seconds.",
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
                                logger.error(
                                    f"Empty message content for question {qid}",
                                )
                                continue

                            try:
                                # {question_id: int {answer:  str, references: list[(document_name, page_number)]}}
                                generated_answers[question_type][qid] = {
                                    "answer": response,
                                    "references": references,
                                }

                            except Exception:
                                logger.exception(
                                    f"Unexpected error for corpus {corpus_id}: Retrying",
                                )

                            break
                else:
                    logger.error(
                        f"Run failed for corpus {corpus_id}. Status: {run.status}",
                    )
                    continue

        if question_type in multi_file_questions:
            # questions = [{files: ([corpus_ids], [file_ids]), question: str}, ...]
            for qid, question_info in tqdm(questions.items()):
                if qid in generated_answers[question_type]:
                    continue  # Skip if already answered

                # Update question assignments with pages to answer and save answers generated so far
                if int(qid) % 10 == 0:
                    with generated_questions_file.open("w") as f:
                        json.dump(generated_answers, f, indent=2)

                compiled_prompt = prompt_to_answer.compile(
                    question=question_info["question"],
                )

                # Get page ids to answer the question
                openai_file_ids = []
                references = [
                    (ref["corpus_id"], ref["page_id"])
                    for ref in question_info["files"][2]
                ]
                for corpus_id, page_id in references:
                    # Get the openai_file_id for each page in the corpus
                    openai_file_ids.append(
                        metadata[corpus_id]["pages"][page_id]["openai_file_id"],
                    )

                logger.info(
                    f"Generating answer for {question_type} question with id {qid} using references {openai_file_ids}...",
                )

                # Create a thread
                thread = openai_client.beta.threads.create()
                logger.info(f"Created thread: {thread.id}")

                # Create message with the PDF attached
                message = openai_client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=compiled_prompt,
                    attachments=[
                        {"file_id": file_id, "tools": [{"type": "file_search"}]}
                        for file_id in openai_file_ids
                    ],
                )

                # Run the assistant
                run = openai_client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant.id,
                )

                max_wait_time = 60  # 1 minute max wait
                start_time = time.time()
                # Wait for completion
                while run.status in ["queued", "in_progress", "cancelling"]:
                    if time.time() - start_time > max_wait_time:
                        logger.error(
                            f"Run for question {qid} timed out after {max_wait_time} seconds.",
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
                                logger.error(
                                    f"Empty message content for question {qid}",
                                )
                                continue

                            try:
                                # {question_id: int {answer:  str, references: list[(corpus_id, page_number)]}}
                                generated_answers[question_type][qid] = {
                                    "answer": response,
                                    "references": references,
                                }

                            except Exception:
                                logger.exception(
                                    f"Unexpected error for question {qid}: Retrying",
                                )

                            break
                else:
                    logger.error(
                        f"Run failed for corpus {[question_info['files'][i][0] for i in range(len(question_info['files']))]}. Status: {run.status}",
                    )
                    continue

    # Save the generated questions to a JSON file
    with generated_questions_file.open("w") as f:
        json.dump(generated_answers, f, indent=2)
