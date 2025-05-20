# ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
## Reference

https://arxiv.org/abs/2004.12832

## Summary

ColBERT introduces a novel approach to information retrieval by balancing the expressiveness of deep language models like BERT with the efficiency required for large-scale search tasks.

Traditional methods either:
- Representation-focused: Encode queries and documents into single vectors, enabling fast retrieval but often lacking nuanced understanding.
- Interaction-focused: Model detailed interactions between query and document tokens, offering higher accuracy but at significant computational costs.

ColBERT bridges this gap through a late interaction mechanism, achieving both high effectiveness and efficiency.

## Notes

Novelty:
Late Interaction Architecture:
- Independent Encoding:
	- Queries and documents are independently encoded using BERT, allowing document representations to be precomputed and stored offline.
- MaxSim Operator:
	- At query time, ColBERT computes the similarity between each query token and all document tokens, selecting the maximum similarity score for each query token. The final relevance score is the sum of these maxima
- This approach captures fine-grained interactions without the computational overhead of full cross-attention mechanisms.

Dimensionality Reduction:
- To reduce storage and computation, ColBERT projects BERT's 768-dimensional output to a lower-dimensional space (e.g., 128 dimensions) using a linear layer. This maintains performance while improving efficiency.

Efficient Indexing and Retrieval:
- Offline Indexing:
	- Document embeddings are computed once and stored, facilitating rapid retrieval.
- Vector Similarity Search:
	- Utilizes tools like FAISS with Inverted File and Product Quantization (IVFPQ) to enable scalable and fast similarity searches across large corpora.

Implementation Details:
- Special Tokens:
	- Queries and documents are prefixed with Q and D tokens, respectively, helping the model distinguish between them during encoding.
- L2 Normalization:
	- Embeddings are L2-normalized to ensure cosine similarity computations are efficient and meaningful.
- Query Augmentation:
	- Short queries are padded with MASK tokens to a fixed length, allowing the model to learn to expand or re-weight query terms during training.