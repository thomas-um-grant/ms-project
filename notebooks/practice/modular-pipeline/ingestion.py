"""
Filename: ingestion.py
Author:  Thomas Grant
Description: This file contains methods to ingest data from various sources, and embed them into vectors.
"""

import os
import sys

import fitz # PyMuPDF
from doc2docx import convert
from docx import Document
from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English


FILES_PATHS = [
    "data/math-for-ml-pdf/math_for_ml_lec_0.pdf",
    "data/math-for-ml-pdf/math_for_ml_lec_1.pdf",
    "data/math-for-ml-pdf/math_for_ml_lec_2.pdf",
]

def ingest(files_path: list[str]):
    """Ingest data from various sources, and embed them into vectors.
    Handles .pdf, .word, .txt
    
    Return:
        - list[torch.tensor]: List of embeded chunks of information of all files.
    """
    corpus = _open_and_read_all_files(files_path=files_path)
    corpus_df = pd.DataFrame(corpus)

    print(corpus_df)



def _open_and_read_all_files(files_path: list[str]) -> list[dict]:
    """Read all files and extract text from each pages.
    
    Return:
        - list[dict]: Each dictionary contains the text of a page and its metadata.
    """
    pages_and_texts = []

    for file_path in files_path:
        if file_path.endswith(".pdf"):
            pages_and_texts.extend(_open_and_read_pdf(pdf_path=file_path))
        elif file_path.endswith(".word"):
            # pages_and_texts.extend(_open_and_read_word(word_path=file_path))
            pass
        elif file_path.endswith(".text"):
            pass
        else:
            raise Exception(f"The file ({file_path}) type is not supported. Documents can be .pdf, .word, or .txt")
        
        return pages_and_texts

def _open_and_read_pdf(pdf_path: str) -> list[dict]:
    """Read pdf files and extract text from each pages.
    
    Return:
        - list[dict]: Each dictionary contains the text of a page and its metadata.
    """

    def text_formatter(text: str) -> str:
        """Performs minor formatting on text.
        
        Return:
            - list[dict]: The cleaned up pages and text.
        """
        # TODO: Find papers related to this to optimize the cleanup.
        cleaned_text = text.replace("\n", " ").strip()

        return cleaned_text

    doc = fitz.open(pdf_path)
    pages_and_texts = []

    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text=text)
        pages_and_texts.append({
            "document_name": "pdf_path",
            "page_id": page_number,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4, # 1 token ~= 4 char: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
            "text": text
        })
        
    return pages_and_texts


def _open_and_read_word(word_path: str) -> list[dict]:
    """Read docx files and extract text from each pages.
    
    Return:
        - list[dict]: Each dictionary contains the text of a page and its metadata.
    """
    # TODO

    return []


if __name__ == "__main__":
    ingest(FILES_PATHS)

    # convert("data/drones-information-word/aviator_code_initiative.doc", "data/drones-information-word/aviator_code_initiative.docx")
    # doc = Document("data/drones-information-word/aviator_code_initiative.docx")
    # print(doc.paragraphs)