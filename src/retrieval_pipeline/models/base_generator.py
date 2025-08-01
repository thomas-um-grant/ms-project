from abc import ABC, abstractmethod
from typing import Any


class BaseGenerationModel(ABC):
    @abstractmethod
    async def generate(self, query, context: list[Any] | None = None):
        """Generate an answer based on the input query and context."""
