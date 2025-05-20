# ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents
## Reference

https://arxiv.org/abs/2502.18017

## Summary

This is another dataset (ViDoSeek - 1.2k questions) tailored for Multimodal retrieval systems. It criticize other benchmarks for 2 main reasons:
- Inefficient and Variable Retrieval Performance (lack of an effective method to integrate visual and textual features)
- Insufficient Activation of Reasoning Capabilities during Generation (only emphasizing on the quantity of knowledge without providing further reasoning guidance presents certain limitations.)

They also present ViDoRAG, a Multi-Modal Hybrid Retrieval, which combines both visual and textual features and dynamically adjusts results distribution based on Gaussian Mixture Models (GMM) prior. This achieves the optimal retrieval distribution for each query, enhancing generation efficiency by reducing unnecessary computations

## Notes

During Generation, there are 3 agents used:
- The seeker rapidly scans thumbnails and selects relevant images with feedback from the inspector.
- The inspector reviews, then provides reflection and offers preliminary answers.
- The answer agent ensures consistency and gives the final answer.

Novelty to note:
They introduce a GMM-based multi-modal hybrid retrieval strategy to effectively integrate visual and textual pipelines.