from dataclasses import dataclass

# Using a dataclass struct here to enforce all the data types to have the same 
# metadata in addition to the text field which holds the actual content.
@dataclass
class Doc:
    document_name: str
    doc_type: str
    doc_id: int
    doc_char_count: int
    doc_word_count: int
    doc_sentence_count: int
    doc_token_count: float
    text: str