# Report

## Title - "Dive into complex RAG pipelines - Combining strength for relevant gains"

## Abstract

### Basis for the project

#### Information Knowledge and Retrieval

#### Diversification of approaches for specific use cases

### Project coverage

#### Dataset generation

#### Evaluation pipeline

#### RAG systems

## Acknowledgements

### Supervisor Nihir, and Chiraag

### Sherpa team

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
- Enables up-to-date and domain-specific knowledge access
- Supports multi-hop, complex, and multimodal queries
- Can use text, images, graphs, or hybrid retrieval

### Research project purpose

#### Introducing a new benchmark generation framework for domain specific knowledge (Consulting)

- Address lack of robust, real-world, and multimodal evaluation datasets
- Propose new methods for dataset creation and benchmarking on specific domain

#### Introducing a new RAG framework from combined best practices

- Introduce latest RAG approaches (Multimodal and Graph)
- Combine best practices from traditional, graph-based, and multimodal RAG
- Modular pipeline for easy experimentation and benchmarking

## Literature review and related work

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

#### In-Depth techniques

- Extraction: OCR for scanned docs, entity extraction, metadata parsing
- Transformation: Text splitting (by sentence, paragraph, semantic units), context windowing
- Embedding: Dense (transformers), sparse (BM25), hybrid
- Storing: Vector DBs, hybrid stores (text + graph)
- Retrieving: vector based search with kNN search, hybrid retrieval, query expansion, dual encoders
- Reranking: Cross-encoder rerankers, contrastive reranking, in-context reranking

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
- Maintenance: Updating graphs as data changes

#### Current research to watch

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

#### Current research to watch

- Muvera

### Evaluation benchmark

#### Exisiting work

- ViDoRe benchmark v1 and v2
- MTEB benchmarks
- BEIR format

#### Challenges

- Lack of standardized, multimodal, and challenging benchmarks
- Difficulty in evaluating multi-hop and cross-modal reasoning
- Dataset bias, annotation quality, and coverage
- No noise, limited size to assess at scale

## Body of report

### Generating an evaluation dataset

- How we address the challenges in our new approach
- Metadata extraction
- Document Clustering
- Question generation
- Corpus selection for answering
- Answer generation
- LLM Peer review

### Combining an enhanced RAG framework

- Integrate best practices from literature
- Experiment with hybrid approaches
- Explain how to assess whether combining different approaches is relevant
- Explain why it is rational to assemble such methods and why it should work

### Use case

#### Existing and potential use cases

#### Our Framework

## Evaluation

### Purpose

- What / why is evaluation important
- Need for robust, real-world, and multimodal evaluation
- Assess retrieval, generation, and reasoning quality

#### Metrics

- Factuality, relevance, comprehensiveness, diversity, logicality, coherence
- Retrieval accuracy, answer correctness, multi-hop reasoning

### Dataset generation

- Benchmark on existing and new datasets to assess dataset quality

### Enhanced RAG framework results

## Conclusions and future work

Dataset gen:

- New benchmarks for multimodal and graph-based RAG
- Better metrics for cross-modal and multi-hop evaluation
- Further automation for dataset generation and annotation

## References

## Declarations

## Appendices
