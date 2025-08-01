import logging
import sqlite3
from pathlib import Path

import weaviate
from weaviate.classes.config import Configure
from weaviate.util import generate_uuid5

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


class JsonDB:
    """Class for Json databases."""

    def __init__(self, db_name: str = "datasets.json") -> None:
        self.db_name = db_name

        # Check if the database file exists or create it
        if (Path(__file__).parent / db_name).exists():
            logger.info(f"Database {db_name} exists, using existing database.")
        else:
            logger.info(f"Database {db_name} does not exist, creating a new database.")
            # Create a new json file where the data will be stored
            with (Path(__file__).parent / db_name).open("w") as db_file:
                db_file.write("{}")


class VectorDB:
    def __init__(self, client_config: dict | None = None):
        self.client = (
            weaviate.connect_to_local()
            if not client_config
            else weaviate.connect_to_custom(**client_config)
        )

        print(self.client.is_ready())

    async def create_collection(self, collection_name: str, schema: dict) -> None:
        if not self.client.collections.exists(collection_name):
            self.client.collections.create(
                name=collection_name,
                vector_config=[
                    Configure.MultiVectors.self_provided(
                        name="multi_vector",
                    ),
                ],
                properties=schema.get("properties", []),
            )

    async def insert_vectors(
        self,
        collection_name: str,
        corpuses: list[dict],
        batch_size: int = 10,
    ) -> None:
        collection = self.client.collections.get(collection_name)

        with collection.batch.dynamic() as batch:
            for vector_data in corpuses:
                batch.add_object(
                    properties=vector_data["properties"],
                    vector=vector_data["vector"],
                )

        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for corpus in corpuses:
                batch.add_object(
                    properties=corpus["properties"],
                    uuid=generate_uuid5(corpus["properties"]["dataset_name"]),
                    vector={"multi_vector": corpus["vector"]},
                )

        if collection.batch.failed_objects:
            print(f"Number of failed imports: {len(collection.batch.failed_objects)}")
            # print(f"First failed object: {collection.batch.failed_objects[0]}")

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict]:
        collection = self.client.collections.get(collection_name)
        response = collection.query.near_vector(
            near_vector=query_vector,
            target_vector="multi_vector",
            limit=top_k,
        )

        return [
            {
                "id": str(obj.uuid),
                "properties": obj.properties,
                "score": obj.metadata.distance,
            }
            for obj in response.objects
        ]


class GraphDB:
    """Class for Graph databases."""

    # NotImplementedError
