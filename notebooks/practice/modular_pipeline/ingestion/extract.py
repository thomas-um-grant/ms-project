from doc import Doc

import fitz # PyMuPDF
from tqdm.auto import tqdm

def extract_documents(files_path: list[str]) -> list[dict]:
    """Read all files and extract text from each pages.
    
    Return:
        - list[Doc]: Each dictionary contains the text of a page and its metadata.
    """
    docs = []

    for file_path in files_path:
        if file_path.endswith(".pdf"):
            docs.extend(_open_and_read_pdf(pdf_path=file_path))
        elif file_path.endswith(".word"):
            # docs.extend(_open_and_read_word(word_path=file_path))
            pass
        elif file_path.endswith(".text"):
            # docs.extend(_open_and_read_text(text_path=file_path))
            pass
        else:
            raise Exception(f"The file ({file_path}) type is not supported. Documents can be .pdf, .word, or .txt")
        
        return docs

def _open_and_read_pdf(pdf_path: str) -> list[Doc]:
    """Read pdf files and extract text from each pages.
    
    Return:
        - list[Doc]: Each Doc contains the text of a page and its metadata.
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
    docs = []

    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text=text)
        docs.append(Doc(
            document_name=pdf_path,
            doc_type="pdf",
            doc_id=page_number,
            doc_char_count=len(text),
            doc_word_count=len(text.split(" ")),
            doc_sentence_count=len(text.split(". ")),
            doc_token_count=len(text) / 4, # 1 token ~= 4 char: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
            text=text
        ))
        
    return docs

def _open_and_read_word(word_path: str) -> list[Doc]:
    """Read docx files and extract text from each pages.
    
    Return:
        - list[Doc]: Each Doc contains the text of a page and its metadata.
    """
    # TODO

    return []

def _open_and_read_text(word_path: str) -> list[Doc]:
    """Read txt files and extract text.
    
    Return:
        - list[Doc]: Each Doc contains the text its metadata.
    """
    # TODO

    return []
