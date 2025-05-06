# From Local to Global - A Graph RAG Approach to Query-Focused Summarization

## Reference

https://arxiv.org/pdf/2404.16130

## Summary

GraphRAG query is focused on the summarization of large documents (corpora).
It derives a knowledge graph by creating hierarchical community summaries (of each chunk).
It outperforms standard RAG on comprehensiveness and diversity for big-picture questions, but it has higher indexing costs and can be less direct.

Core Problem:
Traditional RAG systems retrieve and concatenate small text chunks (“local” retrieval) based on semantic similarity, then feed them into a LLM to answer user queries. This is effective for questions where answers can be found a specific part of the doc, but it struggles with “global” or “broad” queries (e.g., “What are the main themes across all documents?”).

Keypoints:
- Graph construction:
	  Convert the entire corpus into a knowledge-graph-like structure.
	  It prompts an LLM to extract Entities (nodes), Relationships (edges), and Optional Covariates?

- Hierarchical community detection:
	  A community detection algorithm (e.g., Leiden) groups closely related nodes into communities. This forms a hierarchy of topic clusters (from broad clusters at the top level to more fine-grained clusters at deeper levels).

- Pre-summarization:
	  Generate concise “community summaries” at each level of the hierarchy. Each summary covers the entities, relationships, and any covariates found within that community.

- Query-Time summarization:
	  First, each relevant community summary is used to generate a partial answer.
	  These partial answers are then combined (“reduced”) into a final one (coherent summary).

## Advantages

- Better for “global” or “broad” queries:
Very good when a user asks a question that spans the entire dataset.

- More comprehensive and diverse answers:
Since the corpus is pre-indexed into communities, the system naturally covers many aspects or subtopics.

- Modularity and summaries at different granularities
The graph structure can be subdivided at various community levels (broad top-level themes to more detailed sub-communities) which means users can choose the best “level” for a question.

- Scalability for iterative questions
The graph index and summaries can be reused repeatedly for different queries.

## Disadvantages

- Upfront index-building cost
Building the knowledge graph is more complex than simply chunking and embedding documents. It involves multiple LLM calls (e.g., entity extraction, relationship extraction, repeated “gleaning” steps). So it is very expensive and might not be needed for small queries.

- Complexity and potential noise
Entity extraction (with LLM) can miss or merge entities incorrectly. Relationship edges can contain hallucinations or ambiguous references. So the robustness depends on prompt design for extraction and summarization.

- Higher risk of summarization drift
Summaries of summaries can introduce drift or inaccuracy if the LLM “over-abstracts” or incorrectly fuses details. It is challenging to maintain factual consistency at each stage.

- Maintenance over time
The pipeline must be re-run to update the graph if the corpus changes. Traditional RAG only requires re-embedding new documents, which can be simpler to manage.

## Questions

- Adaptive indexing and updates
How does Graph RAG handle continuous updates or additions to the corpus? Is there an incremental way to update the knowledge graph and community summaries without rebuilding from scratch?

- Entity/Relationship accuracy
What mechanisms can be put in place to detect or correct hallucinated relationships or entities introduced by LLM-based extraction?

- Handling noisy or heterogeneous data
Does the approach remain robust when data is extremely diverse (e.g., social media posts, forum chats, multi-lingual corpora)? Would specialized prompts help reduce noise?

- Quality control for summaries
How can we ensure that each layer of summarization (from entity-level to community-level to final answer) preserves key facts faithfully rather than compounding errors?

- Graph “scale” trade-offs
At what point (query / data size) is it worth building a detailed knowledge graph, and when does it become more cost-efficient than naive RAG queries?
Is there a known tipping point where the indexing overhead pays off?

- Combining retrieval strategies
Could a hybrid method retrieve from both raw text chunks (standard RAG) and from the graph-based summaries?

- Automating community-level granularity
How can the system choose which level of the community hierarchy to use for a given user query?

- User-centric evaluations
Do advance users truly find it more ‘empowering’ and helpful for insight, and do they trust it more or less than naive chunk-based methods?

- Multi-modal extensions
Can the concept be extended beyond text (e.g., images, tables, or knowledge graphs from codebases)? Would LLMs reliably handle cross-modal indexing and summarization?

- Impact on Hallucinations vs. Citation
Does building a knowledge graph reduce final hallucinations by forcibly grounding each stage, or do multi-level summaries risk introducing new unverified ‘facts’?