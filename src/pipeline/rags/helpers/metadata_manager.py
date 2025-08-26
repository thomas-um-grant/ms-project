import json
import logging
from pathlib import Path
from typing import Any


class MetadataManager:
    """Handles metadata operations for RAG systems."""

    def __init__(
        self,
        metadata_path: Path,
        json_indent: int = 2,
        fallback_corpus_id: int = -1,
    ) -> None:
        """
        Initialize the MetadataManager.

        Args:
            metadata_path: Path to the metadata file
            json_indent: JSON indentation for pretty printing (default: 2)
            fallback_corpus_id: Default corpus ID when no metadata exists (default: -1)

        """
        self.metadata_path = metadata_path
        self._metadata_cache: dict[str, Any] | None = None
        self._cache_dirty = False

        # Configuration constants
        self.MIN_ID_PARTS = 2
        self.JSON_INDENT = json_indent
        self.FALLBACK_CORPUS_ID = fallback_corpus_id

    def _ensure_cache_loaded(self) -> None:
        """Ensure metadata cache is loaded."""
        if self._metadata_cache is None:
            self._metadata_cache = self._load_from_disk()

    def _load_from_disk(self) -> dict[str, Any]:
        """Load metadata from disk."""
        if self.metadata_path.exists():
            with self.metadata_path.open("r") as f:
                return json.load(f)
        return {}

    def load_metadata(self) -> dict[str, Any]:
        """
        Load metadata from cache or file.

        Returns:
            Dictionary containing metadata, empty if file doesn't exist

        """
        self._ensure_cache_loaded()
        return (
            self._metadata_cache.copy()
        )  # Return copy to prevent external modification

    def save_metadata(self, metadata: dict[str, Any] | None = None) -> None:
        """
        Save metadata to file.

        Args:
            metadata: Dictionary containing metadata to save. If None, saves current cache.

        """
        if metadata is not None:
            self._metadata_cache = metadata
            self._cache_dirty = True

        if self._cache_dirty and self._metadata_cache is not None:
            with self.metadata_path.open("w") as f:
                json.dump(self._metadata_cache, f, indent=self.JSON_INDENT)
            self._cache_dirty = False

    def flush(self) -> None:
        """Force save any pending changes to disk."""
        if self._cache_dirty:
            self.save_metadata()

    def update_metadata_batch(self, entries: list[tuple[Any, ...]]) -> None:
        """
        Update multiple metadata entries in a single operation.

        Args:
            entries: Iterable of tuples. Supported formats (mixed allowed):
                (doc_id, doc_name, description)
                (doc_id, doc_name, description, embedded_flag)

        """
        self._ensure_cache_loaded()

        min_entry_fields = 3  # (doc_id, doc_name, description)
        for entry in entries:
            if len(entry) < min_entry_fields:
                # Skip malformed entries silently (defensive)
                continue
            doc_id, doc_name, description = entry[:3]
            # Parse doc_id to extract corpus and document parts
            id_split = doc_id.split("_")
            if len(id_split) < self.MIN_ID_PARTS:
                corpus_id_str = str(doc_id)
                doc_id_str = ""
            elif len(id_split) == self.MIN_ID_PARTS:
                corpus_id_str, doc_id_str = id_split
            else:
                corpus_id_str, doc_id_str = "_".join(id_split[:-1]), id_split[-1]

            existing = (
                self._metadata_cache.get(doc_id, {}) if self._metadata_cache else {}
            )
            prev_models = (
                existing.get("embedded_models", {})
                if isinstance(existing, dict)
                else {}
            )
            self._metadata_cache[doc_id] = {
                "corpus-id": corpus_id_str,
                "doc-id": doc_id_str,
                "name": doc_name,
                "description": description,
                "embedded_models": prev_models,
            }

        self._cache_dirty = True

    def get_max_corpus_id(self) -> int:
        """
        Get the maximum corpus ID from existing metadata.

        Returns:
            Maximum corpus ID found, or -1 if no metadata exists

        """
        self._ensure_cache_loaded()
        if not self._metadata_cache:
            return self.FALLBACK_CORPUS_ID

        return max(
            (
                int(data["corpus-id"])
                for data in self._metadata_cache.values()
                if data["corpus-id"].isdigit()
            ),
            default=self.FALLBACK_CORPUS_ID,
        )

    def is_document_processed(
        self,
        doc_name: str,
        fallback_description: str = "Image description unavailable",
    ) -> bool:
        """
        Check if a document has already been processed.

        Args:
            doc_name: Name of the document to check
            fallback_description: Description text that indicates processing failed

        Returns:
            True if document has been processed, False otherwise

        """
        self._ensure_cache_loaded()

        return (
            doc_name in self._metadata_cache
            and self._metadata_cache[doc_name]["description"] != fallback_description
        )

    def get_unembedded_documents(self, model_tag: str) -> list[tuple[str, Any]]:
        """
        Get list of documents not yet embedded for the given embedding model.

        Args:
            model_tag: Identifier of the embedding model variant (see BaseRAG.embedding_model_tag)

        Returns:
            List of (doc_id, metadata) tuples requiring embedding for the supplied model.

        """
        self._ensure_cache_loaded()

        for doc_id, doc_data in list(self._metadata_cache.items()):
            if "embedded_models" not in doc_data:
                doc_data["embedded_models"] = {}
                self._metadata_cache[doc_id] = doc_data
                self._cache_dirty = True

        # Collect docs missing embedding for this model_tag
        pending: list[tuple[str, dict[str, Any]]] = []
        for doc_id, doc_data in self._metadata_cache.items():
            embedded_models = doc_data.get("embedded_models", {})
            if not embedded_models.get(model_tag, False):
                pending.append((doc_id, doc_data))
        return pending

    def mark_as_embedded(self, doc_ids: list[str], model_tag: str) -> None:
        """
        Mark documents as embedded for a specific embedding model.

        Args:
            doc_ids: List of document IDs to mark
            model_tag: Embedding model identifier

        """
        self._ensure_cache_loaded()
        updated = 0
        for doc_id in doc_ids:
            if doc_id in self._metadata_cache:
                entry = self._metadata_cache[doc_id]
                if "embedded_models" not in entry:
                    entry["embedded_models"] = {}
                if not entry["embedded_models"].get(model_tag, False):
                    entry["embedded_models"][model_tag] = True
                    updated += 1
        if updated:
            self._cache_dirty = True
            # Flush immediately so subsequent processes see the update (avoids race conditions)
            self.flush()
            logging.getLogger(__name__).debug(
                "Metadata marked %d docs embedded for model_tag=%s",
                updated,
                model_tag,
            )
