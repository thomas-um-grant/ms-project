# Knowledge Graph-Guided Retrieval Augmented Generation
## Reference

https://arxiv.org/abs/2502.06864

## Summary

KG2RAG claims to be better than GraphRAG, the approach is a bit different, it is more focused on the preprocessing aspect, using KG to expand the chunks retrieved, then using the expanded context to do KG organization.

## Notes

Steps:
- Semantic Search on chunks:
	- The chunks with the top-k highest similarities to the query are selected as the retrieved chunks
- Graph-guided Expansion:
	- Link one chunk to other chunks through the overlapping or connected entities that they contain for retrieved chunk expansion.
- KG-based Context Organization:
	- A post-processing stage before response generation of LLMs, motivated by the following two considerations.
		- Serving as a Filter
			- Calculate the semantic similarities between the expanded chunks with the user query
			- Retain only the most relevant linking information between entities and eliminate redundant edges
		- Serving as an Arranger
			- Integrate the retrieved chunks into intrinsically related and self consistent paragraphs with the KG as the skeleton.