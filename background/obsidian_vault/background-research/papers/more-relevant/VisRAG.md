# VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents
## Reference

https://doi.org/10.48550/arXiv.2410.10594

## Summary

This paper introduces VisRAG, a novel framework for Visual Question Answering (VQA) that enhances answer generation by incorporating retrieval-augmented generation. Unlike traditional VQA systems that rely solely on the input image and question, VisRAG retrieves relevant external information (from image-text pairs) to help answer complex or knowledge-based queries.

The method involves two main components:
- Query-Centric Multi-Modal Retrieval: A retrieval module that fetches related image-text pairs based on the query and visual content.
- Retrieval-Augmented Generation: A generative model that fuses the retrieved information with the original question and image features to generate more accurate answers.

Experiments on OK-VQA and A-OKVQA datasets show that VisRAG significantly outperforms previous methods, especially in handling knowledge-intensive questions. The model also includes ablation studies and qualitative examples to highlight the importance of retrieval quality and fusion strategies.

Interesting notes:
- Datasets and Evaluation: A lot of VQA datasets are provided
- Methodology:
	- Document parsing: [[PP-OCR]] and [[MiniCPM-V]] are used
	- Retrieval Experiments: [[SigLIP]] as the vision encoder and [[MiniCPM]] as the language model.
## Advantages

- Better retrieval performances overall on VQA benchmarks.
- Modular framework, easy to replace different parts / models to fine tune.
- Strong generation performance
- Memory efficient compared to [[ColPali]]

## Disadvantages

- Might not generalize well with more text based documents ?
- More complex framework to implement

## Questions

- Could fine-tuning the retrieval module on downstream VQA tasks further improve performance?