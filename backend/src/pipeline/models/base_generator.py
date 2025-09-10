from abc import abstractmethod
from typing import Any

from pipeline.models.base_model import BaseModel


class BaseGenerationModel(BaseModel):
    """Abstract base class for all generation models. Defines the interface for generating answers."""

    @abstractmethod
    async def generate(self, query: str, context: list[Any] | None = None) -> str:
        """Generate an answer based on the input query and context."""

    @abstractmethod
    def _load_model(self) -> None:
        """Abstract method to load the generation model. Each concrete model will implement its specific loading logic here."""
