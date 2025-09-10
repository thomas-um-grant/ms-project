import json
from pathlib import Path


def load_metadata(data_folder: Path) -> dict:
    """
    Load metadata from a JSON file in the specified data folder.
    """
    metadata_file = data_folder / "extracted_data/metadata.json"
    if metadata_file.exists():
        with metadata_file.open("r") as f:
            return json.load(f)
    else:
        return {}


def save_metadata(data_folder: Path, metadata: dict):
    """
    Save metadata to a JSON file in the specified data folder.
    """
    metadata_file = data_folder / "extracted_data/metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=4)
