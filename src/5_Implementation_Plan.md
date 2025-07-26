# Implementation Plan

## Phase 1: Backend Foundation & Core Service Setup (1-2 weeks)

Goal: Establish the project's skeleton and ensure all services can communicate.

- Project Initialization: Set up the Python project using UV as the package manager. Configure Ruff for linting and formatting from day one.
- API Scaffolding: Create a basic web server using a framework like FastAPI. This is a great choice for its speed and automatic API documentation.
- Dockerize Services: Create a docker-compose.yml file to easily launch and manage your databases (Milvus, Memgraph) and local LLM provider (Ollama). This makes your development environment portable and consistent.
- Database Models: Using an ORM like SQLAlchemy, define the initial SQLite tables for Documents, Chunks, and PipelineConfigurations.
- Health Check Endpoint: Create a /health API endpoint that confirms the backend can successfully connect to SQLite, Milvus, and Memgraph. This is a simple way to verify your setup.

## Phase 2: Minimum Viable Ingestion Pipeline (1 week)

Goal: Be able to upload a text document and have it processed for vector search.

- Create Ingestion Endpoint: Implement the POST /api/v1/ingest endpoint.
- Implement Basic Parsing: Add logic to handle simple .txt and .md file uploads.
- Implement Chunking: Integrate a basic recursive character splitting strategy.
- Generate & Store Embeddings: Use a sentence-transformer model (via Ollama or a local library) to generate embeddings for each chunk.
- Populate Databases: Write the code to insert the embeddings into Milvus and the corresponding metadata into SQLite. Graph ingestion will be skipped for now.

## Phase 3: Minimum Viable Retrieval Pipeline (1-2 weeks)

Goal: Ask a question and get a basic RAG answer.

- Create Query Endpoint: Implement the POST /api/v1/query endpoint.
- Implement Traditional RAG: Add the core logic: embed the query, search Milvus to get the top k chunks.
- Integrate LiteLLM: Connect to LiteLLM to route requests to a single model running on Ollama (e.g., Llama 3). This keeps things simple and free initially.
- Basic Generation: Construct a simple prompt using the retrieved context and the user's question. Call the LLM and return the raw text response. We'll skip Guidance and structured output for now.

## Phase 4: Frontend Foundation & End-to-End Test (2 weeks)

Goal: Create a simple UI to interact with the backend and prove the end-to-end flow works.

- Setup Vue.js Project: Initialize the frontend with Vue.js and TypeScript.
- Build Ingestion UI: Create a simple page with a file uploader that calls the /ingest endpoint.
- Build Query UI: Create a chat-like interface where a user can type a question. This UI will call the /query endpoint and display the raw text answer returned by the backend.
- Connect & Test: Wire up the frontend and backend. At the end of this phase, you should have a working, albeit basic, "ask questions about your document" web app.

## Phase 5: Advanced RAG & Evaluation Pipeline (2-3 weeks)

Goal: Implement GraphRAG and the system for measuring performance.

- Implement Graph Ingestion: Enhance the ingestion pipeline to extract entities/relationships using an LLM and populate Memgraph.
- Implement Graph Retrieval: Add the GraphRAG logic to the query endpoint as a selectable option.
- Integrate Guidance: Refactor the generation step to use Guidance to produce structured JSON output (answer + sources). Update the frontend to parse and display this correctly.
- Implement Evaluation Backend: Build the logic for the dataset generation and evaluation pipelines.
- Integrate Closed-Source Models: Add API keys for Gemini and OpenAI to LiteLLM, making them selectable options in the UI.

## Phase 6: Visualization and UI Polish (1-2 weeks)

Goal: Make the evaluation results understandable and improve the user experience.

- Integrate D3.js: Create a new page in the frontend for evaluation results. Use D3.js to build interactive tables and charts that compare the performance of different RAG configurations.
- Refine UI: Improve the overall user experience with loading indicators, error handling, and a more polished design.
- Configuration Management: Build the UI for creating and saving different ingestion and retrieval configurations.
