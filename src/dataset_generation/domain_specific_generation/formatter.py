import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def format_beir_dataset(folder: Path) -> None:
    """Format the dataset to BEIR format."""
    logger.info("Formatting dataset to BEIR format...")

    # Dataset Format:
    # "test": {
    #     "corpus":  [{"corpus_id": int, "file_name": str, "azure_file_id": str}],
    #     "queries": [{"query_id": str, "query": str, "query_type": str}],
    #     "qrels": [{"query_id": str, "corpus_id": int, "pages": list[int], "score": int}, "answer": str],
    # }

    # Get generate questions and answers from the dataset folder
    generated_questions_file = folder / "generated_data/generated_questions.json"
    generated_answers_file = folder / "generated_data/generated_answers.json"

    if not generated_questions_file.exists() or not generated_answers_file.exists():
        msg = "Generated questions or answers file not found."
        raise FileNotFoundError(msg)

    # Load the generated questions and answers
    generated_questions = {}
    with generated_questions_file.open() as qf:
        generated_questions = json.load(qf)

    generated_answers = {}
    with generated_answers_file.open() as af:
        generated_answers = json.load(af)

    beir_dataset: dict = {
        "test": {
            "corpus": [],
            "queries": [],
            "qrels": [],
        },
    }

    # Get all the corpus_id / doc_id pairs from the generated questions
    corpus_list = []
    queries_list = []
    qrels_list = []

    for question_type, questions_by_type in generated_questions.items():
        for question_id, question_item in questions_by_type.items():
            # Validate file structure
            if "files" not in question_item or len(question_item["files"]) < 2:
                logger.warning(f"Invalid file structure for question {question_id}")
                continue
            if len(question_item["files"][0]) != len(question_item["files"][1]):
                logger.warning(f"Mismatched file lists for question {question_id}")
                continue

            for corpus_id, file_id in [
                (question_item["files"][0][x], question_item["files"][1][x])
                for x in range(len(question_item["files"][0]))
            ]:
                if question_type not in generated_answers:
                    logger.warning(
                        f"No answers found for question type {question_type}. Skipping."
                    )
                    continue
                if question_id not in generated_answers[question_type]:
                    logger.warning(
                        f"No answers found for question {question_id} of type {question_type}. Skipping."
                    )
                    continue
                if "references" not in generated_answers[question_type][question_id]:
                    logger.warning(
                        f"No references found for question {question_id} of type {question_type}. Skipping."
                    )
                    continue

                question = question_item["question"]
                corpus_list.append(
                    {
                        "corpus_id": corpus_id,
                        "file_name": f"{corpus_id}.pdf",
                        "azure_file_id": file_id,
                    },
                )
                queries_list.append(
                    {
                        "query_id": question_id,
                        "query": question,
                        "query_type": question_type,
                    },
                )

                pages = []
                for reference in generated_answers[question_type][question_id][
                    "references"
                ]:
                    if reference[0] == corpus_id:
                        pages.append(reference[1])

                qrels_list.append(
                    {
                        "query_id": question_id,
                        "corpus_id": corpus_id,
                        "pages": pages,
                        "score": 1,
                        "answer": generated_answers[question_type][question_id][
                            "answer"
                        ],
                    },
                )

    # Build the BEIR dataset
    beir_dataset["test"]["corpus"] = corpus_list
    beir_dataset["test"]["queries"] = queries_list
    beir_dataset["test"]["qrels"] = qrels_list

    # Save the BEIR dataset to a JSON file
    beir_dataset_file = folder / "generated_data/beir_dataset.json"
    with beir_dataset_file.open("w") as bdf:
        json.dump(beir_dataset, bdf, indent=4)


def generate_human_evaluation_json(folder: Path) -> None:
    """Generate a JSON file for human evaluation."""
    logger.info("Generating JSON file for human evaluation...")
    qa_pairs = []

    beir_dataset_file = folder / "generated_data/beir_dataset.json"
    if not beir_dataset_file.exists():
        msg = "BEIR dataset file not found. Please run format_beir_dataset first."
        raise FileNotFoundError(msg)

    # Load the BEIR dataset
    with beir_dataset_file.open() as bdf:
        beir_dataset = json.load(bdf)
    # From the BEIR dataset, extract in line using qrels:
    # "query_id", "corpus_id", "doc_id", "question", "answer"
    i = 0
    while i < len(beir_dataset["test"]["qrels"]):
        qrel = beir_dataset["test"]["qrels"][i]

        question_id = qrel["query_id"]
        question_type = ""
        question = ""
        file_name = ""
        pages = qrel["pages"]
        answer = qrel["answer"]

        # Find the question for this query_id
        for query in beir_dataset["test"]["queries"]:
            if query["query_id"] == question_id:
                question = query["query"]
                question_type = query["query_type"]
                break

        for corpus in beir_dataset["test"]["corpus"]:
            if corpus["corpus_id"] == qrel["corpus_id"]:
                file_name = corpus["file_name"]
                break

        # Prepare file_name / files
        files = (
            f"{file_name} (pages: {', '.join(str(p) for p in pages)})"
            if pages and len(pages) > 0
            else ""
        )

        # Write to JSON file
        qa_pairs.append(
            [
                question_id,
                question_type,
                question,
                files,
                answer,
            ],
        )
        i += 1

    # Merge the qa_pairs that have the same question_id
    # In this case, they also have the same question_type, question, and answer
    # We need to concatenate the corpus_ids, file_names, and azure_file_ids
    merged_qa_pairs = {}
    for pair in qa_pairs:
        question_id = pair[0]
        question_type = pair[1]
        question = pair[2]
        files = pair[3]
        answer = pair[4]

        if question_id not in merged_qa_pairs:
            merged_qa_pairs[question_id] = {
                "question_type": question_type,
                "question": question,
                "answer": answer,
                "files": [],
            }

        merged_qa_pairs[question_id]["files"].append(files)

    # Convert the lists into comma-separated strings
    for data in merged_qa_pairs.values():
        data["files"] = ", ".join(f for f in data["files"] if f != "")

    qa_pairs_file = folder / "generated_data/human_evaluation_qa_pairs.json"
    with qa_pairs_file.open("w") as qaf:
        json.dump(merged_qa_pairs, qaf, indent=4)


def convert_to_excel(folder: Path):
    # Read the JSON file
    qa_pairs_file = folder / "generated_data/human_evaluation_qa_pairs.json"
    with qa_pairs_file.open() as f:
        data = json.load(f)

    # Convert to DataFrame - extract the data we want from each entry
    rows = []
    for qid, item in data.items():
        rows.append(
            {
                "Question ID": qid,
                "Question Type": item["question_type"],
                "Question": item["question"],
                "Files": item["files"],
                "Answer": item["answer"],
            },
        )

    formatted = pd.DataFrame(rows)

    # Save as Excel
    excel_path = folder / "generated_data/human_evaluation_qa_pairs.xlsx"
    formatted.to_excel(excel_path, index=False)

    logger.info(f"Excel file saved to {excel_path}")
