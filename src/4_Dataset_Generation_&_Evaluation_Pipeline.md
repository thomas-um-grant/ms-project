# Dataset Generation & Evaluation Pipeline

## A. Dataset Generation Pipeline

This process automates the creation of a "golden" dataset to test your retriever.

- Initiation: The user selects a collection of ingested documents to serve as the source material for the dataset.
- Automated Q&A Generation: The backend kicks off a process that iterates through the text chunks of the selected documents.

  - For each chunk, a request is sent to a powerful LLM (e.g., Gemini Pro or GPT-4 via LiteLLM).
  - The prompt instructs the LLM to act as a "test creator." It asks the model to generate a question that can be answered exclusively from the content of that specific chunk.
  - A second LLM call is then made, providing both the chunk and the newly generated question, to produce a concise, "ground truth" answer.

- Data Structure: Each generated Q&A pair is stored as a structured object. This is your evaluation "test case."

```json
{
  "question_id": "qid_001",
  "question": "What is the primary function of the LiteLLM component?",
  "ground_truth_answer": "The primary function of the LiteLLM component is to act as a unified API layer, allowing the backend to communicate with various LLM providers without writing provider-specific code.",
  "source_chunk_id": "chunk_abc_123",
  "source_document": "system_architecture.pdf"
}
```

- Storage: The entire collection of these Q&A pairs is saved as a distinct dataset in the SQLite database, ready to be used for evaluation.

## B. Evaluation Pipeline

This pipeline executes the tests and measures the results.

- Configuration: The user selects two things in the UI:

  - A QA Dataset to test with.
  - A Retrieval Pipeline Configuration to test (e.g., chunk size 512, text-embedding-3-small model, Traditional RAG, Gemini 1.5 Flash for generation).

- Automated Execution: The backend runs each question from the dataset through the selected RAG pipeline. For each question, it produces a final generated_answer and also keeps a record of the retrieved_context chunks.

- Metrics Calculation: After the run is complete, the system compares the pipeline's output against the ground truth data for each question to calculate key metrics. This often involves using an LLM as a "judge."

  - Context Precision: Did the retrieved context include the original source chunk? (Score: 1 if yes, 0 if no). This measures the retriever's accuracy.
  - Answer Faithfulness: Does the generated_answer contain any information not present in the retrieved_context? This checks for hallucinations.
  - Answer Relevancy: How well does the generated_answer address the user's question?
  - Answer Correctness: How semantically similar is the generated_answer to the ground_truth_answer?

- Results Storage & Visualization:
  - The aggregated scores for the entire run (e.g., average Context Precision of 85%, average Answer Relevancy of 92%) are stored in SQLite, linked to the specific dataset and RAG configuration used.
  - The Vue.js frontend can then query an API endpoint for these results.
  - D3.js will be used to render the data into insightful visualizations:
    - A comparison table showing different RAG configurations side-by-side.
    - Bar charts visualizing the scores for each metric, making it easy to see which configuration performs best.
