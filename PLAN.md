# Research strategy 
## 🔭 Phase 1: Exploration & Planning 
Goal: Define scope, do background research, and decide on evaluation metrics.
✔️ Key Tasks:
* Define the hypothesis:“Graph-based representations in RAG can improve retrieval quality by structuring and filtering context, especially in multi-modal scenarios.”
* Conduct a literature review:
    * GraphRAG, Hierarchical Retrieval, Retrieval-Augmented Multi-modal Generation, etc...
    * Embedding strategies (e.g., co-embedding images/text).
    * Instruction-agnostic vs. instruction-aware retrieval.
* Pick evaluation benchmarks:
    * HotpotQA, MultiModalQA, or my own constructed dataset.
* Decide on baseline:
    * Start with vanilla RAG (FAISS + cosine + chunked input).
* Set up tools & environment:
    * LangChain / LlamaIndex / Haystack for pipelines.
    * Open-source LLM (Mistral, LLaMA), or GPT APIs?
    * Weaviate / Neo4j / NetworkX for graph components.
    * CLIP / BLIP2 / LLaVA for multimodal encoding.

## 🧪 Phase 2: Prototyping & Data
Goal: Build minimal working prototypes of Graph-based retrieval and multi-modal integration.
✔️ Key Tasks:
* Create a small knowledge base (with text and images, if multi-modal).
* Prototype 3 retrieval variants for comparison?
    1. Vanilla RAG (chunked + cosine).
    2. Graph-based RAG (documents as nodes, edges as relationships).
    3. Multi-modal Graph-RAG (image nodes, text/image cross-links).
* Experiment with instruction separation or other simple techniques:
    * Strip prompt instructions and match only content-relevant parts during retrieval.

## 🔬 Phase 3: Experimentation & Tuning
Goal: Gather comparative data and refine the techniques.
✔️ Key Tasks:
* Define metrics:
    * Retrieval Quality: Recall@K, MRR.
    * Generation Quality: BLEU, ROUGE, human evals?
* Iterate on graph construction:
    * Try clustering, coreference resolution, semantic edges.
    * Add inter-modal connections.
* Try re-ranking methods (check if it is kind of cheating?):
    * Post-retrieval filtering using local LLM scoring.
* Start logging errors and examples of hallucinations.

## 📈 Phase 4: Analysis & Insights
Goal: Derive insights, consolidate findings, and refine the hypothesis.
✔️ Key Tasks:
* Compare methods:
    * Look for precision/recall trade-offs when context is more structured but smaller.
* Create visualizations:
    * Graph diagrams, context length vs. hallucination, modal influence (text/image).
* Extract insights:
    * Where do graphs help?
    * When does multi-modality help or confuse?

## 🧾 Phase 5: Report & Finalization
Goal: Polish the report and prepare for submission and/or presentation.
📄 Report Structure:
1. Abstract – Problem, method, results, significance.
2. Introduction – RAG overview, why it needs improvement.
3. Related Work – Graph RAG, multi-modal retrieval, evaluation techniques.
4. Methodology – The graph construction, retrieval pipeline, models used.
5. Experiments – Setup, datasets, metrics.
6. Results & Discussion – Insights, tradeoffs, limitations.
7. Conclusion & Future Work – Summary and next steps.