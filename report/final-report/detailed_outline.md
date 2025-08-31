# Report

## Title - "A Hybrid Retrieval Framework for RAG: Combining Textual, and Multimodal Information via Reranking"

## Abstract

Context:

Retrieval-Augmented Generation (RAG) is crucial for grounding LLMs, but no single retrieval method (traditional, graph, or multimodal) excels in all scenarios.

Problem:

This fragmentation necessitates a way to combine the strengths of these disparate retrievers to improve overall retrieval quality. Furthermore, evaluating such systems is hampered by a lack of robust, domain-specific benchmarks.

Our Contribution:

This project introduces "MultiRAG," a modular framework that merges outputs from traditional, and multimodal RAG pipelines using a final reranking step. We also propose and implement a novel methodology for generating large-scale, high-quality evaluation datasets for specialised domains.

Methodology & Results:

We implemented original RAG archetypes and our MultiRAG system. Through extensive ablation studies on our generated "Consulting" dataset, we demonstrate that MultiRAG improves retrieval relevance (e.g., by X% in nDCG@10) compared to any single RAG approach.

Conclusion:

Our findings validate that a hybrid, reranking-based approach is a potent strategy for RAG optimization, and our dataset generation technique provides a valuable tool for future research.

## Acknowledgements

### Supervisor Nihir, and Chiraag

## Contents pages

## Introduction

### LLMs and VLMs revolution

- Rapid progress in LLMs (GPT, Llama, Mistral, etc.) and VLMs (CLIP, BLIP2, PaliGemma)
- Empowerment: LLMs excel at text generation, reasoning, and knowledge tasks
- VLMs enable understanding and generation across text and images

#### Challenges

- Limited context window and memory
- Hallucination and lack of grounding in external facts
- Inability to access up-to-date or external knowledge
- Poor performance on domain-specific or multimodal tasks
- Difficulty with reasoning over complex, structured, or multi-hop information

### Rise of RAG

#### General introduction

- Retrieval-Augmented Generation (RAG): combines LLMs with external retrieval to ground responses in relevant documents
- Pipeline: retrieve relevant chunks, then feed to LLM for answer generation

#### Overview of its potential

- Improves factuality, reduces hallucination
- Improves Explainability
- Enables up-to-date and domain-specific knowledge access
- Supports multi-hop, complex, and multimodal queries
- Can use text, images, graphs, or hybrid retrieval
- Reranking strategies

#### Challenges

- Important Tradeoffs (Latency, High Memory / RAM usage)

### Research project purpose

#### Introducing a new enhancement approach to retrieval combining multiple retrievers

- Introduce latest RAG approaches (Multimodal and Graph)
- Combine best practices from traditional, graph-based, and multimodal RAG
- Modular pipeline for easy experimentation and benchmarking

#### Introducing a new benchmark generation for domain specific knowledge (Consulting)

- Address lack of robust, real-world, and multimodal evaluation datasets
- Propose new methods for dataset creation and benchmarking on specific domain

## Literature review and related work (Shouldn't be too technical here, detail the only specific work used in the body)

### Attention with Transformers

- What are transformers and how they are used for query answering.
- The limitations of LLMS to answer queries on domain specific and contextual data.

### Traditional RAG

#### The building blocks

- Text extraction
- Document chunking and preprocessing (text splitters, cleaning, metadata extraction)
- Embedding generation (models like Sentence Transformers, OpenAI, etc.)
- Vector store/database (FAISS, Weaviate, Milvus) for fast similarity search
- Retriever module (dense, sparse, hybrid)
- Reranker module to improve retrieval quality
- Generator module (LLM, e.g., GPT-4, Llama)

#### Challenges

- Hallucination and factuality: LLMs may ignore retrieved context or hallucinate
- Retrieval quality: Embedding drift, poor chunking, context window limits
- Scalability: Indexing and updating large corpora, latency
- Multilingual and domain adaptation challenges

### Graph RAG

#### Purpose and potential

- Addresses limitations of vanilla RAG for multi-hop, global, or broad queries
- Enables structured reasoning over entities and relationships
- Supports modular, hierarchical, and community-based summarization

#### Notable existing frameworks / research

- GraphRAG: Converts corpus into a knowledge graph, uses community detection for hierarchical summaries, excels at global queries
- LightRAG: Lightweight, dual-level retrieval (local/global), incremental updates, efficient for evolving datasets
- PathRAG: Uses relational paths for multi-hop reasoning, decay pruning for concise path selection, interpretable rationale
- KG2RAG: Uses knowledge graphs to expand and organize context, focuses on preprocessing and chunk expansion

Notable techniques:

- DRIFT method: Pruning based on path length and semantic decay
- Late chunking: Efficiently incorporates new data without full reindexing

#### Challenges

- High upfront cost for graph construction (entity/relation extraction, LLM calls)
- Sensitivity to prompt engineering and extraction quality (hallucinated or missing entities/edges)
- Scalability: Graph size, path explosion, memory/computation for large corpora
- Maintenance: Updating graphs as data changes (Updating without rebuilding, but also removing time sensitive relation that change over time) -> Zep / Graphitti?

#### Current research to watch -> Move to Body or Conclusion / Future work

- Hybrid retrieval: Combining graph and vector-based methods
- Multi-modal graph construction (images, tables, code)
- Evaluation metrics for cross-modal and multi-hop reasoning

Other paths

- Explainability and trust in graph-based reasoning
- Agentic workflows (planner/critic agents) for graph RAG
- Real-time or streaming updates to graphs

### Multimodal RAG

#### OCR Approach

- Traditional: OCR + text chunking, layout detection, captioning
- Limitations: Slow, error-prone for complex layouts, fails on handwriting or poor scans

#### PaliGemma

- Vision-language model for image and text embedding
- Uses SIGLIP with Pali3 for improved multi-modal retrieval

#### ColPali

- Bypasses OCR, directly encodes document pages as images
- Late interaction retrieval: Multi-vector image embeddings, fine-grained matching

#### Other apporaches

VDocRAG:

- Dual-encoder for queries and document images, dynamic high-res image encoding, multi-hop reasoning, context-independent questions, OpenDocVQA dataset

ViDoRAG:

- Multi-modal hybrid retrieval (visual + textual), dynamic result adjustment (GMM prior), agent-based generation (seeker, inspector, answer agent), ViDoSeek dataset

VisRAG:

- Query-centric multi-modal retrieval, retrieval-augmented generation, modular and memory-efficient, strong on VQA benchmarks

mmGraphRAG:

- Dual DB (graph + vector), incorporates object detection, spatial relationships, and image features

#### Challenges

#### Multi Vector size and scalability

#### Current research to watch -> Move to Future work

- Muvera

### Reranking strategies

- Current reranking strats that exist
- Why are they useful, and why are they different from our approach
- Give some context to our approach if needed

## Methodology and System Implementation (8 - 10 pages)

### The MultiRAG Framework Architecture:

Start with a high-level diagram showing the three parallel RAG pipelines feeding into a merger/reranker module, which then feeds the generator. This visual anchor is invaluable.

Explain the philosophy: leverage specialist retrievers and use a reranker as an expert "judge" to select the best context.

### Component Implementation: Individual RAG Pipelines:

#### Traditional RAG Pipeline:

Detail the choices. (e.g., "Text was extracted using PyMuPDF, chunked semantically using a sentence-transformer model, embedded with bge-large-en-v1.5, and indexed in a FAISS vector store.")

#### In-Depth techniques -> Enhancement possible - TODO Nihir: Should this go in future work?

- Extraction: OCR for scanned docs, entity extraction, metadata parsing
- Transformation: Text splitting (by sentence, paragraph, semantic units), context windowing
- Embedding: Dense (transformers), sparse (BM25), hybrid
- Storing: Vector DBs, hybrid stores (text + graph)
- Retrieving: vector based search with kNN search, hybrid retrieval, query expansion, dual encoders
- Reranking: Cross-encoder rerankers, contrastive reranking, in-context reranking

#### Graph RAG Pipeline:

talk about graph. (e.g., "Entities and relationships were extracted using OpenAI's GPT-4 with a specific prompt structure. The graph was built using NetworkX and community detection was performed using the Leiden algorithm, similar to the original GraphRAG paper.")

#### Multimodal RAG Pipeline:

Detail the approach. (e.g., "We implemented a multimodal pipeline based on the ColPali principle, directly encoding document page images using PaliGemma-3b to create multi-vector embeddings...")

### The Fusion Strategy: Multi-Retrieval Reranking:

This is a core part of the novelty. Explain the algorithm precisely.

How did I gather the chunks from the 3 pipelines? (e.g., "We retrieve the top-k=20 chunks from each pipeline.")

How do I handle duplicates?

How are the candidates reranked? (e.g., "The combined set of up to 60 chunks is then passed to a bge-reranker-large cross-encoder, and the final top-k=5 chunks are selected.")

### Evaluation Dataset Generation Methodology:

This is the second major contribution. Detail it like a scientific recipe.

#### Corpus Selection:

Explain the source (e.g., "publicly available consulting reports from...") and preprocessing.

#### Document Clustering & Metadata:

"Documents were clustered by topic using BERTopic to ensure thematic diversity in question generation."

#### Synthetic Question-Answer Generation:

"For each cluster, a fine-tuned Mixtral 8x7B model was prompted to generate 5 types of questions (e.g., simple fact-finding, comparative, multi-hop reasoning)..."

#### Ground Truth Establishment & LLM Peer Review:

"Generated QA pairs were validated by a separate GPT-4 agent acting as a reviewer, which checked for factuality against the source document and answerability. Pairs failing validation were discarded."

#### Reproducibility

The large-scale 'Consulting' dataset was generated following the methodology detailed in this chapter. The generation process was performed utilizing computational resources provided by [Company Name]. Due to the proprietary nature of the source documents and the terms of the collaboration, this dataset remains the intellectual property of the company and cannot be publicly released. However, the dataset generation pipeline developed for this project is a core contribution of this work and can be applied to any corpus to create similar high-quality evaluation benchmarks.

### Use Case: An Interactive Demonstration & Analysis Platform

This title frames the app not just as a demo, but as a tool used for the own analysis, which strengthens its academic relevance.

How to Structure the Section
Break down the description into four key parts: Purpose, Architecture, Features, and its Role in the Research.

#### Purpose and Objectives

Start by explaining why building the application. What was its goal in the context of the Master's project?

To Provide a Tangible Demonstration: State that the primary objective was to create an interactive interface to demonstrate the end-to-end functionality of the implemented RAG systems.

To Enable Comparative Analysis: A key goal was to allow for a direct, side-by-side comparison of the outputs from the Traditional, Graph, Multimodal, and the proposed MultiRAG systems for any given query. This visual comparison is much more powerful than looking at metrics tables alone.

To Facilitate Qualitative Evaluation: The application served as a crucial tool for the research, allowing to perform qualitative error analysis and identify specific examples where one system excelled or failed, which can then discuss in the "Results" section.

Example Text:

"To demonstrate the practical application and facilitate comparative analysis of the different RAG pipelines, a full-stack web application was developed. The platform's primary objectives were: 1) to provide an interactive interface for querying the implemented RAG systems against a selected document corpus, 2) to enable a direct, side-by-side comparison of the generated answers and retrieved sources, and 3) to serve as a tool for the qualitative evaluation and error analysis presented in the 'Evaluation' chapter of this report."

#### System Architecture

This is where to detail the technical stack. A simple architecture diagram TODO.

Diagram: Create a simple block diagram with three columns: Frontend, Backend (API Server), and RAG Services.

Text Description:

Frontend: Describe the technology and its purpose.

Stack: (e.g., React, Next.js, TypeScript, Tailwind CSS)

Why chose it: "The frontend was built using Next.js (a React framework) and TypeScript. This stack was chosen for its ability to create a fast, responsive, and type-safe user interface, ideal for presenting complex information like source chunks and generated text in a clear, component-based manner."

Backend (API Server): This is the intermediary between the user interface and Python RAG logic.

Stack: (e.g., Python with FastAPI or Flask)

Why chose it: "A backend API server was developed using FastAPI in Python. FastAPI was selected for its high performance, asynchronous capabilities (essential for handling potentially long-running RAG queries without blocking), and its automatic generation of interactive API documentation (Swagger UI), which aided in development and debugging."

RAG Services: This is the core research code.

How it's integrated: "The backend server wraps the core RAG logic. Each of the four RAG systems (Traditional, Graph, Multimodal, MultiRAG) was exposed via a distinct API endpoint. When a request is received from the frontend, the backend invokes the selected RAG pipeline, which performs the retrieval and generation, and then returns the final answer along with source metadata as a JSON object."

#### Key Features and User Workflow

Describe how a user interacts with the application. This is the perfect place to include screenshots.

Corpus Selection: "The user begins by selecting a document from a pre-loaded corpus based on our generated 'Consulting' dataset." (Add a screenshot of this interface).

Query Interface: "The user then inputs their question into a text field and can select which RAG systems to run in parallel using a set of checkboxes: 'Traditional', 'Graph', 'Multimodal', and 'MultiRAG'."

Results Display: "Upon submission, the application presents the results in a side-by-side comparison view." Describe what is shown for each result:

Generated Answer: The final text generated by the LLM.

Retrieved Sources: The list of source chunks used to generate the answer.

Source Metadata: For each chunk, display crucial metadata like the source document name, page number, and the retrieval score.

Visual Context: "Crucially, for the Multimodal and Traditional systems, the application displays a thumbnail image of the document page from which the chunk was extracted, with the chunk's bounding box highlighted. For the Graph RAG, a simplified visualization of the retrieved knowledge subgraph is rendered."

(Include a well-composed screenshot showing the side-by-side comparison of two different RAG outputs for the same query).

#### Role in the Research Project

Finally, explicitly link the app back to the thesis.

"Beyond being a demonstration tool, the application was instrumental in the research process. It was used extensively during the qualitative analysis phase to identify the strengths and weaknesses of each approach. For instance, the ability to visually compare the retrieved chunks for a multi-hop query immediately highlighted the superiority of the Graph RAG pipeline in that context. The specific examples used in the 'Discussion' section of this report were sourced directly through interaction with this platform."

## Experiments and Results"

#### OLD - Purpose

- What / why is evaluation important
- Need for robust, real-world, and multimodal evaluation
- Assess retrieval, generation, and reasoning quality

### Experimental Setup:

Datasets: Describe the generated "Consulting" dataset (size, #QA pairs, types of questions) and any standard benchmarks used for comparison (e.g., "To validate our dataset's robustness, we also report performance on a subset of the DocVQA benchmark").

Metrics: Define the evaluation metrics clearly. For retrieval: nDCG@k, Mean Reciprocal Rank (MRR). For generation: ROUGE-L, BERTScore, and perhaps a Factuality Score using an LLM judge.

Baselines: State clearly: "We compare our MultiRAG system against three baselines: our standalone implementations of Traditional RAG, Graph RAG, and Multimodal RAG."

#### OLD - Exisiting work

- ViDoRe benchmark v1 and v2
- MTEB benchmarks
- BEIR format

#### OLD - Challenges

- Lack of standardized, multimodal, and challenging benchmarks
- Difficulty in evaluating multi-hop and cross-modal reasoning
- Dataset bias, annotation quality, and coverage
- No noise, limited size to assess at scale

#### OLD - Metrics

- Factuality, relevance, comprehensiveness, diversity, logicality, coherence
- Retrieval accuracy, answer correctness, multi-hop reasoning

### Results:

Ablation Study: Individual RAG Performance: Present a table showing the performance of each of the 3 individual pipelines on the dataset. This is essential to show that different RAGs excel at different question types, justifying the hybrid approach.

Main Result: MultiRAG Performance: Present the key results table comparing MultiRAG against the three baselines. This table should be the centerpiece of the evaluation. Use bold to highlight the winning scores.

Dataset Quality Analysis: If I benchmarked the dataset generation process itself, present those results here.

### Discussion and Error Analysis:

Don't just present numbers. Interpret them. Why did MultiRAG perform better? Show an example where Traditional RAG failed, Graph RAG found a key connection, and MultiRAG's reranker correctly prioritized the graph context.

Discuss the limitations. When does MultiRAG fail? (e.g., "When all three retrievers fail to find relevant context, the reranker has nothing to work with.")

## Conclusions and future work

Conclusion:

Briefly restate the problem, the solution (MultiRAG + dataset methodology), and summarize the main finding (e.g., "Our results confirm that combining retriever strengths through reranking provides a statistically significant improvement in retrieval quality...").

Limitations:

Be honest about the weaknesses (e.g., "The MultiRAG system has higher computational overhead and latency due to running three parallel retrieval systems.").

Future Work:

- On MultiRAG:
  "Future work could explore more sophisticated fusion techniques, such as adaptive weighting of retrievers based on query type, or using a learning-to-rank model instead of a static cross-encoder."

- On Dataset Generation:
  "The dataset generation process could be enhanced by incorporating more complex, non-textual question types (e.g., 'What is the trend shown in the chart on page 5?')."

## References

## Declarations

I declare that during the course of this research project, I received in-kind support from [Company Name] in the form of computational resources for the generation of an experimental dataset. Under the terms of this informal collaboration, the resulting dataset is the intellectual property of [Company Name]. The aggregated results and findings from the analysis of this dataset are presented in this thesis with their full permission. No financial compensation or formal employment relationship was part of this arrangement.

## Appendices
