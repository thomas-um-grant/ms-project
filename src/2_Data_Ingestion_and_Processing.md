# Data Ingestion and Processing

## 1. Document Upload and Initial Handling

- Interface: The Vue.js frontend will feature a component for file uploads, allowing users to drag-and-drop or select one or more files.
- Supported Formats: Initially, we should support common text-based formats: .pdf, .txt, .md, and .json. Support for .docx or images (for multimodal RAG) can be added later.
- API Endpoint: The frontend will send the files to a backend endpoint like POST /api/v1/ingest. This request will also include the selected ingestion configuration (see section 4).

## 2. Document Parsing and Chunking

Once the backend receives a file, it begins processing.

- Parsing: The system will use libraries to extract clean text. A good choice would be the unstructured library, as it can handle various file types automatically. For PDFs, pypdf is another solid alternative.

- Chunking Strategy: This is a key configuration setting. The backend logic should support multiple strategies, which the user can select in the UI:

  - Fixed-Size: Splitting text into chunks of a fixed number of characters or tokens (e.g., 1000 characters) with a defined overlap (e.g., 100 characters). Simple and fast.
  - Recursive Character Splitting: A more advanced technique that tries to split text along natural separators (paragraphs, sentences, words) to keep related content together. This is generally the preferred method.
  - Semantic Chunking: An advanced option that uses an embedding model to decide where to split chunks, grouping semantically related sentences together. This can yield the best results but is computationally more expensive.

- Chunk Metadata: Every chunk must be stored with essential metadata. This is crucial for tracing back answers to their source. The metadata for each chunk should include at least:
  - source_document_id
  - page_number (if applicable)
  - chunk_index (e.g., chunk 3 of 50)

## 3. Populating the Databases

After chunking, the data is sent to the appropriate databases.

- Milvus (Vector Storage):

  - Embedding Generation: For each text chunk, the backend will use an embedding model to create a vector representation. This model should be configurable (e.g., text-embedding-ada-002 from OpenAI, nomic-embed-text via Ollama, etc.).
  - Storage: The backend will insert the data into a Milvus collection. Each entry will consist of the vector embedding, the original text chunk, and its metadata.

- Memgraph (Graph Storage):

  - Entity/Relationship Extraction: If GraphRAG is enabled in the configuration, the backend will process each text chunk to identify entities and their relationships. This step typically requires a powerful LLM.

  - LLM Call with Guidance: For each chunk, the backend will call an LLM (via LiteLLM) using a Guidance template. The template will instruct the LLM to extract a list of triples in the format (Subject, Relationship, Object).

    - Example Text: "The backend is written in Python and communicates with the Vue.js frontend."
    - Extracted Triples: ("Backend", "IS_WRITTEN_IN", "Python"), ("Backend", "COMMUNICATES_WITH", "Vue.js frontend").

  - Storage: These triples are then inserted into Memgraph as nodes and edges, building a knowledge graph from the document's content.

- SQLite (Metadata & Relational Storage):

  - Purpose: To act as the central ledger for the ingestion process.
  - Schema: We'll need a few tables to track everything:
    - documents: Stores information about each uploaded file (doc_id, filename, status, upload_time).
    - ingestion_configs: Stores the settings used for a particular ingestion run (config_id, chunk_strategy, embedding_model, etc.).
    - chunks: Stores the metadata for each chunk, linking everything together (chunk_id, doc_id, config_id, milvus_vector_id, text_preview).

## 4. Configurable Ingestion Pipeline

To make the system flexible, the user should be able to define an ingestion configuration in the UI. This configuration would be sent to the backend with the documents.

Example Configuration (as JSON):

```json
{
  "chunk_strategy": "recursive_character",
  "chunk_size": 1000,
  "chunk_overlap": 150,
  "embedding_model": "ollama/mxbai-embed-large",
  "graph_extraction": {
    "enabled": true,
    "llm_model": "gemini/gemini-1.5-flash"
  }
}
```
