# Benchmarking Multimodal RAG through a Chart-based Document Question-Answering Generation Framework
## Reference

https://arxiv.org/abs/2502.14864

## Summary

Propose CHARt-based document question-answering GEneration (CHARGE), a framework that produces evaluation data through structured keypoint extraction, crossmodal verification, and keypoint-based generation. By combining CHARGE with expert validation, they construct Chart-MRAG Bench (267 documents spanning 8 domains, 8 types of questions, 1,283 paragraphs, and 627 charts), a comprehensive benchmark for chart-based MRAG evaluation, featuring 4,738 question-answering pairs across 8 domains from real-world documents.

## Notes

CHARGE operates in three stages:
1) extracting self-contained keypoints from both textual and visual content.
2) verifying the modality authenticity of extracted keypoints through crossmodal verification.
3) generating diverse QA pairs by combining related keypoints across documents and modalities.

