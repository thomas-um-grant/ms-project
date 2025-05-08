# PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths
## Reference
https://arxiv.org/abs/2502.14902

Additional explanation:
https://www.youtube.com/watch?v=oetP9uksUwM

## Summary

This paper presents Relational Reasoning Paths, which is a way to represents relations in between concepts extracted from the original corpus. This enables PathRAG to outperform [[LightRAG]], enriching the semantic and comprehension of the model for higher quality response.
It still relies on vector database to retrieve indexed nodes.

## Advantages

- Improved semantic reasoning: Multi-hop relational paths introduce explicit relational knowledge that improves answer comprehension and reasoning.

- Efficient pruning: The proposed α-decay pruning strategy effectively selects concise and meaningful paths, reducing redundancy.

- Better performance: Demonstrates superior performance on open-domain QA benchmarks compared to standard RAG and LightRAG.

- Interpretable rationale: Offers transparency in the reasoning process by explicitly showing the paths between concepts.

- Adaptability: The method can be layered over any base retriever-reader architecture with minimal changes.

## Disadvantages

- Dependency on graph quality: Relies heavily on the quality and completeness of the constructed knowledge graph; errors or omissions in entity linking and relation extraction can degrade performance.

- Additional computation: Path retrieval and pruning introduce extra overhead compared to standard RAG models.

- Scalability constraints: As the number of nodes and potential paths increases, computational complexity in path scoring and selection becomes a challenge.

- Hyperparameter sensitivity: Model effectiveness is influenced by the number of nodes/paths retrieved and the decay rate α, requiring careful tuning.

## Notes

Method:
- Node retrieval:
	- A dense retriever (e.g., DPR) selects top-k relevant documents or nodes from a vector index based on the user query.
- Path retrieval:
	- For each pair of retrieved nodes, multi-hop paths are extracted from a pre-constructed knowledge graph. Paths are scored using a semantic decay function that penalizes longer paths to prioritize concise, informative paths. Pruning is applied to retain top-m paths based on this scoring.
- Answer generation:
	- Retrieved nodes and their relational paths are serialized into a prompt structure.
	- A decoder-only language model (like LLaMA or similar) generates the final answer using this enriched input.

Hyperparams:
- Number of retrieved nodes
	- Find the right amount of retrieved nodes to limit context while not loosing key info
- Number of retrieved paths
	- Reach a peak early on
- Decay rate
	- prioritizing shorter paths with a proper α serves as effective prior knowledge for the pruning process.

Ablation:
- Path ordering:
	- Random ordering vs Hop-first ordering
- Prompt format:
	- Flat organization

Evaluation dimensions:
- Comprehensiveness:
	- How much detail does the answer provide to cover all aspects and details of the question?
- Diversity:
	- How varied and rich is the answer in providing different perspectives and insights on the question?
- Logicality:
	- How logically does the answer respond to all parts of the question?
- Relevance:
	- How relevant is the answer to the question, staying focused and addressing the intended topic or issue?
- Coherence:
	- How well does the answer maintain internal logical connections between its parts, ensuring a smooth and consistent structure?