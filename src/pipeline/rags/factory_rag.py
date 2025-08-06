from pathlib import Path
from typing import ClassVar

from pipeline.rags.base_rag import BaseRAG
from pipeline.rags.multimodal import MultiModalRAG


class RAGFactory:
    """Factory for creating RAG instances based on configuration."""

    _rag_types: ClassVar[dict[str, type[BaseRAG]]] = {
        # "text": TextRAG,
        "multimodal": MultiModalRAG,
        # "graph": GraphRAG,
    }

    @classmethod
    def create_rag(
        cls,
        config: dict,
        data_dir: Path,
    ) -> BaseRAG:
        """
        Create a RAG instance based on configuration.

        Args:
            config: Configuration dict
            data_dir: Base data directory for RAG storage

        Returns:
            Instantiated RAG object

        Raises:
            ValueError: If RAG type is not supported
            KeyError: If required config fields are missing

        """
        # Validate required fields
        cls._validate_config(config)

        rag_type = config["type"]
        name = config["name"]
        configs = config.get("configs", {})

        # Get RAG class
        if rag_type not in cls._rag_types:
            available_types = ", ".join(cls._rag_types.keys())
            raise ValueError(
                f"Unsupported RAG type: '{rag_type}'. "
                f"Available types: {available_types}",
            )

        rag_class = cls._rag_types[rag_type]

        # Create and return RAG instance
        return rag_class(
            name=name,
            data_dir=data_dir,
            configs=configs,
        )

    @classmethod
    def register_rag_type(cls, type_name: str, rag_class: type[BaseRAG]) -> None:
        """
        Register a new RAG type.

        Args:
            type_name: Name to identify the RAG type
            rag_class: RAG class to register

        """
        if not issubclass(rag_class, BaseRAG):
            raise ValueError("RAG class must inherit from BaseRAG")

        cls._rag_types[type_name] = rag_class

    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of available RAG types."""
        return list(cls._rag_types.keys())

    @classmethod
    def _validate_config(cls, config: dict) -> None:
        """
        Validate configuration has required fields.

        Args:
            config: Configuration dictionary

        Raises:
            KeyError: If required fields are missing

        """
        required_fields = ["type", "name", "configs"]

        for field in required_fields:
            if field not in config:
                raise KeyError(f"Missing required configuration field: '{field}'")

        # Type-specific validation could be added here
        rag_type = config["type"]
        if rag_type == "multimodal":
            cls._validate_multimodal_config(config.get("configs", {}))

    @classmethod
    def _validate_multimodal_config(cls, configs: dict) -> None:
        """Validate multimodal-specific configuration."""
        if "chunking_strategy" in configs:
            valid_strategies = [
                "page",
                "section",
                "paragraph",
            ]  # Add your valid strategies
            if configs["chunking_strategy"] not in valid_strategies:
                raise ValueError(
                    f"Invalid chunking_strategy for multimodal RAG: {configs['chunking_strategy']}. "
                    f"Valid options: {valid_strategies}",
                )
