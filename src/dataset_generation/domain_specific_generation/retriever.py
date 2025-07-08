import json
import logging
from pathlib import Path

import numpy as np
from loader import load_metadata
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


def get_embedding(
    text: str,
    openai_client: AzureOpenAI,
    embedding_model: str,
) -> list[float]:
    response = openai_client.embeddings.create(input=[text], model=embedding_model)
    return response.data[0].embedding


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def retrieve_top_k_pages(
    data_folder: Path,
    documents: list[str],
    query: str,
    k: int,
    openai_client: AzureOpenAI,
    embedding_model: str,
) -> list[dict]:
    """
    For a list of document paths and a query, returns the top k pages (with doc info) most relevant to the query.
    """
    metadata = load_metadata(data_folder)

    embeddings = {}
    # Load embeddings from JSON file
    embeddings_file = data_folder / "generated_data/pages_embeddings.json"
    if embeddings_file.exists():
        with embeddings_file.open("r") as f:
            embeddings = json.load(f)
    else:
        # If embeddings file does not exist, initialize an empty dict
        embeddings = {}

    all_pages_embeddings = []  # List[Tuple[doc_path, (page_num, page_text)]]

    for corpus_id in documents:
        pages = metadata[corpus_id]["pages"]

        # Get computed page embeddings
        pages_embeddings = embeddings.get(corpus_id, [])

        # Embed all pages using AzureOpenAI client
        if not pages_embeddings:
            for page_id, page in pages.items():
                page_text = f"topic: {page['topic']}\nsummary: {page['summary']}"
                emb = get_embedding(page_text, openai_client, embedding_model)
                pages_embeddings.append({"page_id": page_id, "embedding": emb})

            # Update embeddings for the corpus_id
            embeddings[corpus_id] = pages_embeddings

            embeddings_file = data_folder / "generated_data/pages_embeddings.json"
            with embeddings_file.open("w") as f:
                json.dump(embeddings, f)

        all_pages_embeddings.extend(
            [(corpus_id, page_embedding) for page_embedding in pages_embeddings],
        )

    # Embed query
    query_embedding = get_embedding(query, openai_client, embedding_model)

    # Compute similarities
    similarities = [
        (
            corpus_id,
            emb["page_id"],
            cosine_similarity(query_embedding, emb["embedding"]),
        )
        for corpus_id, emb in all_pages_embeddings
    ]

    # Get top k, sort similarities by similarity score
    top_k_indices = sorted(similarities, key=lambda x: x[2], reverse=True)[:k]
    results = []
    for corpus_id, page_id, score in top_k_indices:
        results.append(
            {
                "corpus_id": corpus_id,
                "page_id": page_id,
                "similarity": score,
            },
        )

    return results


def retrieve_top_k_pages_for_questions(
    data_folder: Path,
    openai_client: AzureOpenAI,
    embedding_model: str,
) -> None:
    """
    For a set of questions, retrieves the top k pages for each question.
    """
    # Load questions from JSON file
    questions_file = data_folder / "generated_data/generated_questions.json"
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    questions_info = {}
    with questions_file.open("r") as f:
        questions_info = json.load(f)

    for question_type, questions in questions_info.items():
        for qid, qdata in questions.items():
            query = qdata["question"]

            if (
                "files" not in qdata
                or not qdata["files"]
                or not isinstance(qdata["files"][0], list)
            ):
                logger.warning(f"Invalid files structure for question {qid}")
                continue
            documents = qdata["files"][0]

            top_k_pages = retrieve_top_k_pages(
                data_folder=data_folder,
                documents=documents,
                query=query,
                k=3
                if question_type in ["recency-bias", "single-fact", "single-slide"]
                else 5,
                openai_client=openai_client,
                embedding_model=embedding_model,
            )

            # Ensure files list has at least 3 elements
            files_list = questions_info[question_type][qid]["files"]
            while len(files_list) < 3:
                files_list.append([])
            # Update the third element (index 2) with retrieved pages
            files_list[2] = top_k_pages

            logger.info(
                f"Retrieved {len(top_k_pages)} top pages for question {qid} of type {question_type}",
            )

        # Save to question assignments
        with questions_file.open("w") as f:
            json.dump(questions_info, f, indent=2)
