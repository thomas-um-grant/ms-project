from typing import Any


class Orchestrator:
    def __init__(self):
        self.rags = {}

    def add(self, name: str, rag):
        """Add a RAG to the orchestrator."""
        self.rags[name] = rag

    def index(self, name: str, corpuses: list[Any] | None = None):
        """Index images with the specified RAG."""
        if name in self.rags:
            return self.rags[name].index(corpuses)

        raise RAGNotFoundError(name)

    def retrieve(self, name: str, queries: list[str]):
        """Retrieve relevant documents using the specified RAG."""
        if name in self.rags:
            return self.rags[name].retrieve(queries)

        raise RAGNotFoundError(name)

    def answer(self, name: str, queries: list[str]):
        """Answer queries using the specified RAG."""
        if name in self.rags:
            return self.rags[name].answer(queries)

        raise RAGNotFoundError(name)


class RAGNotFoundError(ValueError):
    """Exception raised when a specified RAG is not found in the orchestrator."""

    def __init__(self, name: str):
        super().__init__(f"RAG '{name}' not found.")
        self.name = name

    def __str__(self):
        return f"RAG '{self.name}' not found."
