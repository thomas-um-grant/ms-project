import logging
import os
import sqlite3

from utils.exceptions import RelationalDBError

from ..dbs_manager import RelationalDB

logger = logging.getLogger(__name__)


def add_corpus(dataset_name: str, doc_id: str, corpus_id: int, corpus_path: str) -> str:
    """
    Add a corpus to the relational database.

    Args:
        dataset_name (str): Name of the dataset.
        doc_id (str): Document ID.
        corpus_id (int): Corpus ID.
        corpus_path (str): Path to the corpus.

    Returns:
        str: The primary key of the inserted corpus.

    """
    try:
        relational_db = RelationalDB(os.getenv("RELATIONAL_DB_NAME", "relational.db"))
        query = """
        INSERT INTO corpus (dataset_name, doc_id, corpus_id, corpus_path)
        VALUES (?, ?, ?, ?);
        """
        params = (dataset_name, doc_id, corpus_id, corpus_path)
        relational_db.execute_query(query, params)

    except RelationalDBError as e:
        return f"Error adding corpus: {e}"

    else:
        return f"Corpus '{dataset_name}-{doc_id}-{corpus_id}' added successfully."


def add_corpuses(corpus_data: list[dict]) -> str:
    """
    Add multiple corpora to the relational database.

    Args:
        corpus_data (list[dict]): List of dictionaries containing corpus data.

    Returns:
        str: Success message.

    """
    try:
        relational_db = RelationalDB(os.getenv("RELATIONAL_DB_NAME", "relational.db"))
        query = """
        INSERT INTO corpus (dataset_name, doc_id, corpus_id, corpus_path)
        VALUES (?, ?, ?, ?);
        """
        for data in corpus_data:
            params = (
                data["dataset_name"],
                data["doc_id"],
                data["corpus_id"],
                data["corpus_path"],
            )
            relational_db.execute_query(query, params)

    except RelationalDBError as e:
        return f"Error adding corpora: {e}"

    else:
        return "Corpora added successfully."


def get_corpus_path(
    dataset_name: str,
    doc_id: str,
    corpus_id: int,
) -> sqlite3.Row | None:
    """
    Get a corpus from the relational database.

    Args:
        dataset_name (str): Name of the dataset.
        doc_id (str): Document ID.
        corpus_id (int): Corpus ID.

    Returns:
        str: The corpus path on disk.

    """
    try:
        relational_db = RelationalDB(os.getenv("RELATIONAL_DB_NAME", "relational.db"))
        query = """
        SELECT corpus_path FROM corpus
        WHERE dataset_name = ? AND doc_id = ? AND corpus_id = ?;
        """
        params = (dataset_name, doc_id, corpus_id)
        result = relational_db.execute_query(query, params, fetch="fetchone")

    except RelationalDBError:
        logger.exception("Error fetching corpus path")

        return None

    else:
        # If result is a Row or tuple, access by index 0
        return result[0] if result is not None else None


def get_corpuses_by_dataset(
    dataset_name: str,
) -> list[sqlite3.Row] | sqlite3.Row | None:
    """
    Get all corpora for a specific dataset.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        list[sqlite3.Row]: List of corpora for the specified dataset.

    """
    try:
        relational_db = RelationalDB(os.getenv("RELATIONAL_DB_NAME", "relational.db"))
        query = """
        SELECT * FROM corpus WHERE dataset_name = ?;
        """
        params = (dataset_name,)
        result = relational_db.execute_query(query, params)

    except RelationalDBError:
        logger.exception("Error fetching corpora by dataset")

        return None

    else:
        return result if result else None
