from abc import ABC, abstractmethod

import torch

from database.dbs_manager import VectorDB
from retrieval_pipeline.utils import get_torch_device


class BaseRAG(ABC):
    def __init__(self, vector_db: VectorDB, collection_name: str):
        self.vector_db = vector_db
        self.collection_name = collection_name
        self.device = torch.device(get_torch_device("auto"))

    @abstractmethod
    def _get_collection_schema(self) -> dict:
        """Return the schema for this RAG's collection."""

    async def _setup_collection(self):
        """Setup the vector database collection."""
        schema = self._get_collection_schema()
        await self.vector_db.create_collection(self.collection_name, schema)

    async def initialize(self):
        """Call this after creating the instance to setup the collection."""
        await self._setup_collection()

    @abstractmethod
    async def index(self, corpuses: list):
        """Index images or documents for retrieval."""

    @abstractmethod
    async def retrieve(self, queries: list, top_k: int | None = None):
        """Retrieve the most relevant images or documents for the given queries."""

    @abstractmethod
    async def answer(self, queries: list):
        """Generate answers based on the retrieved images or documents."""
