# GRAG: Graph Retrieval-Augmented Generation
## Reference

https://arxiv.org/abs/2405.16506

## Summary

They propose a divide-and-conquer strategy that leverages K-hop ego-graphs and soft pruning to approximate the optimal textual subgraph.

Consists of four main stages:
- Indexing of k-hop ego-graphs
- Graph retrieval
- Soft pruning to mitigate the impact of irrelevant entities
- Generation with pruned textual subgraphs.

GRAG's core workflow-retrieving textual subgraphs followed by soft pruning-efficiently identifies relevant subgraph structures while avoiding the computational infeasibility typical of exhaustive subgraph searches, which are NP-hard.

They propose a novel prompting strategy that achieves lossless conversion from textual subgraphs to hierarchical text descriptions.

## Notes

Using Soft Pruning and Ranking to chose which nodes to keep and to return as part of the context.

Claim to outperform SoTa on multi-hop reasoning benchmarks