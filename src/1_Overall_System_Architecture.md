# Overall System Architecture

## 1. High-Level Overview

The system is designed as a classic client-server application. The Vue.js frontend is the user's entry point, running entirely in their browser. It communicates with the Python backend via a REST API. The backend is the central orchestrator, handling all business logic, data processing, and communication with the various databases and external LLM services.

The core idea is to decouple the components:

- The frontend doesn't know which database or LLM is being used; it just sends requests and displays data.
- The backend uses LiteLLM as a unified interface, so it doesn't need to write separate code for Gemini, OpenAI, or Ollama.

## 2. Component Breakdown & Responsibilities

Here's the role of each piece of the tech stack:

### Frontend (Vue.js / TypeScript)

- Responsibility: Provides the user interface (UI).
- Tasks: Renders interactive forms for uploading documents and configuring pipelines. Sends user queries to the backend. Visualizes evaluation results using D3.js to create dynamic charts and tables.

### Backend (Python / FastAPI)

- Responsibility: The brain of the application.
- Tasks:
  - Exposes API endpoints for the frontend (e.g., /ingest, /query, /evaluate).
  - Orchestrates the data ingestion pipeline (parsing, chunking, embedding).
  - Executes retrieval logic by querying Milvus and/or Memgraph.
  - Communicates with LLMs through LiteLLM.
  - Uses Guidance to enforce structured, reliable JSON outputs from the LLMs.
  - Manages evaluation workflows and stores results in SQLite.

### Databases & Storage

- SQLite (Relational DB):

  - Purpose: Stores structured metadata and application state.
  - Data: User configurations (e.g., saved pipeline settings), pointers to raw documents, evaluation history, and QA dataset contents.

- Milvus (Vector DB):

  - Purpose: Fast semantic search.
  - Data: Stores numerical vector embeddings of the document chunks. Essential for traditional and multimodal RAG.

- Memgraph (Graph DB):

  - Purpose: Storing and querying relationships.
  - Data: Stores extracted entities (like people, places, concepts) as nodes and their relationships as edges. Key for GraphRAG.

### LLM & Development Services

- LiteLLM (Model Endpoint Manager):

  - Purpose: A unified API layer for all LLMs.
  - Workflow: The Python backend sends a request to LiteLLM, specifying the model (gemini-pro, gpt-4, ollama/llama3). LiteLLM handles the specific API call and returns the response in a standardized format.

- Guidance (Structured Output):

  - Purpose: To control the LLM's output format.
  - Workflow: After the backend gets context from the databases, it creates a "Guidance program" (a template) that is sent to the LLM (via LiteLLM). This forces the model to return a well-structured JSON, preventing errors.

- UV & Ruff (Tooling):
  - Purpose: These are for development efficiency. UV will manage the Python packages, and Ruff will keep the code clean and consistent. They don't run in production but are vital for building the backend.

## 3. Core Interaction Flows

Here’s how it all comes together in practice.

- Flow A: Data Ingestion

  - User uploads a document (e.g., document.pdf) via the Vue.js UI.
  - Frontend sends the file to the Backend API (e.g., POST /ingest).
  - Backend receives the file:
  - Parses the text and/or images.
  - Chunks the content into smaller pieces.
  - Generates vector embeddings for each chunk using an embedding model (via LiteLLM or a local model).
  - Stores embeddings in Milvus.
  - For Graph RAG, extracts entities and relationships and stores them in Memgraph.
  - Saves metadata (e.g., file name, chunk IDs) to SQLite.
  - Backend sends a success response to the Frontend.

- Flow B: RAG Query

  - User types a question ("What is GraphRAG?") into the Vue.js UI and selects a retrieval configuration.
  - Frontend sends the query to the Backend API (e.g., POST /query).
  - Backend receives the query:
  - Embeds the user's question.
  - Retrieval Step:
  - For Traditional RAG: Queries Milvus with the question embedding to find relevant chunks.
  - For GraphRAG: Queries Memgraph to find related entities and context.
  - Augmentation Step: Compiles the retrieved context into a prompt.
  - Generation Step:
  - Uses a Guidance template to structure the final prompt.
  - Sends the prompt to the desired LLM (e.g., Gemini) via the LiteLLM interface.
  - LiteLLM returns a structured JSON response to the Backend.
  - Backend sends the final answer and source information to the Frontend.
  - Frontend displays the answer and uses D3.js to show sources or other relevant data.
