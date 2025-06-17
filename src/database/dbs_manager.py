import logging
import sqlite3
from pathlib import Path

from utils.exceptions import RelationalDBError

logger = logging.getLogger(__name__)


class RelationalDB:
    """Class for relational databases."""

    def __init__(self, db_name: str = "relational.db") -> None:
        self.db_name = db_name

        # Check if the database file exists or create it
        if (Path(__file__).parent / db_name).exists():
            logger.info(f"Database {db_name} exists, using existing database.")
        else:
            logger.info(f"Database {db_name} does not exist, creating a new database.")
            # Create a new database with the relational schema
            with (
                Path(Path(__file__).parent / "schemas/relational_db_schemas.sql")
            ).open() as db_file:
                db_schema = db_file.read()
                conn = self.get_connection()
                with conn:
                    conn.executescript(db_schema)

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection to the database."""
        conn = sqlite3.connect(Path(__file__).parent / self.db_name)
        conn.row_factory = sqlite3.Row  # To access columns by name
        return conn

    def close_connection(self, conn: sqlite3.Connection | None) -> None:
        """Close the database connection."""
        if conn:
            conn.close()
            logger.info(f"Connection to {self.db_name} closed.")
        else:
            logger.warning("No connection to close.")

    def execute_query(
        self,
        query: str,
        params: tuple | None = None,
        fetch: str = "fetchall",
    ) -> list[sqlite3.Row] | sqlite3.Row | None:
        """Execute a query on the database."""
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                if fetch == "fetchone":
                    return cursor.fetchone()

                if fetch == "fetchmany":
                    return cursor.fetchmany()

                return cursor.fetchall()

        except sqlite3.Error as e:
            msg = f"sqlite error: {e}"
            raise RelationalDBError(msg) from e

        finally:
            self.close_connection(conn)
