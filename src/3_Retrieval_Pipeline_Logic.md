# General Retrieval Flow (R-A-G)

- Retrieve: Based on the user's query and the chosen RAG strategy, the system fetches the most relevant context from one or more databases.
- Augment: The retrieved context is combined with the original user query and formatted into a detailed prompt for a large language model.
- Generate: The LLM receives the augmented prompt and generates a final, human-readable answer, which is then sent back to the user with source citations.

Here’s how this flow is implemented for each of your required retrieval strategies.

## A. Traditional RAG (Vector Search)

This is the standard, foundational RAG approach.

- 1. Query Embedding: The user's query (e.g., "What are the core components of the system?") is converted into a vector embedding. Crucially, this must use the exact same embedding model that was used during data ingestion.
- 2. Vector Search: The backend takes this query vector and queries the Milvus collection to find the most similar vectors. This is an Approximate Nearest Neighbor (ANN) search that returns the top-k most relevant text chunks (e.g., the top 5).
- 3. Context Assembly: The text content from these top-k chunks is collected to form the context.

## B. GraphRAG (Knowledge Graph Search)

This method excels at answering questions about relationships and complex entities.

- 1. Entity Extraction from Query: The backend first sends the user's query to an LLM (via LiteLLM) to identify the key entities. A Guidance template can ensure the output is a clean list, e.g., for "Tell me about Memgraph's role," the LLM would extract ["Memgraph"].
- 2. Graph Traversal: The extracted entities are used to construct a query in Memgraph's query language (Cypher). The query searches for these entities and their direct neighbours in the graph.
  - Example Cypher Query: MATCH (e {name: 'Memgraph'})-[r]-(neighbor) RETURN e, r, neighbor;
- 3. Contextualization: The results from the graph (a set of nodes and relationships) are converted back into natural language sentences. For instance, (Memgraph)-[:STORES]->(Entities) becomes "Memgraph stores entities." This text becomes the context.

## C. Multimodal RAG (Text + Image)

This extends vector search to include images. This assumes you've used a multimodal embedding model (like CLIP) during ingestion.

- 1. Query Embedding: The user's text query is embedded using the multimodal model.
- 2. Unified Vector Search: The query vector is used to search in Milvus. Because both text chunks and images were embedded into the same vector space, this search can retrieve a mix of relevant text and relevant images.
- 3. Context Assembly: The context will include the text from the retrieved chunks. The retrieved images can be passed to a multimodal LLM (like Gemini) in the generation step or simply displayed alongside the text answer in the UI.

## Final Stage: Augment & Generate (Common to All Strategies)

After the retrieval step (whether from Milvus or Memgraph), the process is the same:

- Augment Prompt: The backend constructs a final prompt using the retrieved context.

  - Example Template:

  ```txt
      You are a helpful AI assistant. Answer the following user's question based *only* on the provided context below. Cite the source for each piece of information you use.

      [CONTEXT]
      {retrieved_text_chunks_and_graph_data}
      [/CONTEXT]

      User Question: {user_query}
  ```

- Generate with Guidance: This entire prompt is then passed to the generative LLM (e.g., Gemini, GPT-4) through LiteLLM. To ensure a reliable and parseable response, a Guidance program wraps the call, forcing the LLM to return a JSON object.

  - Required Output Structure:
    ```json
    {
      "answer": "This is the final generated answer...",
      "sources": [
        {
          "document_id": "doc_123",
          "chunk_id": "chunk_abc",
          "content": "The text snippet from the source document..."
        }
      ]
    }
    ```

- Send to Frontend: The backend receives this structured JSON and forwards it to the Vue.js app, which can then beautifully render the answer and its interactive sources.
