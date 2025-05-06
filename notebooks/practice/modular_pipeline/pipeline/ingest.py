from notebooks.practice.modular_pipeline.ingestion.extract import extract_documents

import pandas as pd
import torch

def ingest(files_path: list[str]) -> list[torch.tensor]:
    """Ingest data from various sources, and embed them into vectors.
    Handles .pdf for now, but can be extended to .word and .text.
    
    Return:
        - list[torch.tensor]: List of embeded chunks of information of all files.
    """
    corpus = extract_documents(files_path=files_path)
    corpus_df = pd.DataFrame(corpus)

    print(corpus_df.head())

    return []