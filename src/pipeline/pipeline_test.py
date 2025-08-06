import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

load_dotenv()

from retrieval_pipeline.rags.multimodal import MultiModalRAG


async def main():
    """Main function demonstrating the usage of MultiModalRAG."""
    # Configuration
    rags_data_dir = os.getenv("RAGS_DATA_DIR")
    if rags_data_dir is None:
        # Fallback to default if environment variable is not set
        rags_data_dir = "/Users/thomas/repos/imperial/project/ms-project/src/data/rags"
        print(f"RAGS_DATA_DIR not set, using default: {rags_data_dir}")

    data_dir = Path(rags_data_dir)

    # Device configuration options (now passed through configs):
    # preferred_device = None  # Auto-detect best option
    preferred_device = "cpu"  # Force CPU (most stable)
    # preferred_device = "mps"      # Force Apple Silicon GPU
    # preferred_device = "cuda"     # Force NVIDIA GPU

    configs = {
        "preferred_device": preferred_device,
        "batch_size": 8,
    }

    # Initialize the RAG
    rag = MultiModalRAG(
        name="multimodal_testing",
        data_dir=data_dir,
        embedding_model="colqwen2_embed",
        generation_model="colqwen2_ollama_gen",
        configs=configs,
    )

    try:
        # Extract documents (PDFs) into images and embeddings
        # print("Extracting documents...")
        # documents = [
        #     Path("/Users/thomas/Desktop/RAG_M1.pdf"),
        #     Path("/Users/thomas/Desktop/RAG_M2.pdf"),
        # ]
        # await rag.extract(documents)
        # # print("Extraction completed!")

        print("Starting indexing...")
        await rag.index()
        print("Indexing completed!")

        # # Perform retrieval using async method
        # print("\nPerforming single query retrieval...")
        # results = await rag.retrieve(
        #     "What is the name of the employee getting an offer?",
        #     top_k=3,
        # )

        # # Display results
        # print("\nSingle Query Retrieval Results:")
        # for md, sc in results:
        # print(f"  doc {md['corpus-id']}_{md['doc-id']} - score {sc}")
        # print("  description:", md.get("description", "..."))
        # print()

        # # Demonstrate batch retrieval with multiple queries
        # batch_queries = [
        #     "What is the first offer amount?",
        #     "What is the second offer amount?",
        #     "Did the offer include a signing bonus?",
        # ]
        # print("Performing batch retrieval...")
        # batch_results = await rag.retrieve(batch_queries, top_k=3)

        # print("\nBatch Retrieval Results:")
        # # Since results are now flattened, we need to process them differently
        # # Each query returns top_k results, so we can group them
        # results_per_query = len(batch_queries)
        # top_k_per_query = 2

        # for i in range(results_per_query):
        #     start_idx = i * top_k_per_query
        #     end_idx = start_idx + top_k_per_query
        #     query_results = batch_results[start_idx:end_idx]

        #     print(f"Query {i + 1}: '{batch_queries[i]}'")
        #     for md, sc in query_results:
        #         print(f"  doc {md['corpus-id']}_{md['doc-id']} - score {sc:.4f}")
        #         print("  description:", md.get("description", "..."))
        #     print()

        # # Test answer generation
        # print("Generating answers for queries...")
        # for query in batch_queries:
        #     answer = await rag.answer(query)
        #     print(f"Answer for '{query}': {answer}")

        # Test answer generation
        test_query = (
            "What was the example used to better understand the bag of words concept?"
        )
        answer, results = await rag.answer(test_query)
        print(f"Question:\n{test_query}")
        print("Retrieved documents:")
        for md, sc in results:
            print(f"  doc {md['corpus-id']}_{md['doc-id']} - score {sc}")

        print(
            f"Answer:\n{answer}",
        )

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
