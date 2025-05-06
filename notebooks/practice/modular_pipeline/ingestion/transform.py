from ..ingestion import Doc

from spacy.lang.en import English


def transform_corpus(corpus: list[Doc]):
    pass


def _chunk_docs(corpus: list[Doc]):

    # TODO: Clear up this assumption, find better semantic chunking for all languages?
    # Assuming all sources are in English for now
    nlp = English()

    # Add a sentencizer pipeline to make the  Robust
    nlp.add_pipe("sentencizer")