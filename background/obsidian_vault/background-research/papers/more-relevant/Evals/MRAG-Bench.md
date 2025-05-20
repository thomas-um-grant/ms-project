# MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models
## Reference

https://openreview.net/forum?id=Usklli4gMc

## Summary

MRAG-BENCH consists of 16,130 images and 1,353 human-annotated multiple-choice questions across 9 distinct scenarios. Previous benchmarks mainly focused on retrieving from textual knowledge. However, there are scenarios where retrieving correct textual knowledge is hard and sometimes not as useful as visual knowledge.

- It focuses on real-world scenarios where visually augmented information is useful
- It incorporates 9 diverse multimodal RAG scenarios covering various types of image objects
- It features cleaned ground-truth images for each question that align with human knowledge
- It provides robust evaluation settings for deterministic evaluations.

## Notes

Perspective understanding aspect:
- ANGLE evaluates the ability of models to utilize visual knowledge of common shooting angles to identify and reason about less common, long-tailed viewpoints of visual entities.

- PARTIAL evaluates the ability of models to use complete appearance knowledge to identify and reason when only a partial image of the visual entities is available.

- SCOPE evaluates the ability of models to leverage high-resolution, detailed images for identifying and reasoning about visual entities in longer-scoped, low-resolution images.

- OCCLUSION evaluates the ability of models to use ground-truth image knowledge to identify and reason when visual entities are occluded or partially hidden in natural scenes.

Transformative understanding aspect:
- TEMPORAL evaluates the ability of models to use familiar image knowledge to identify and reason about visual entities undergoing temporal changes that may not be represented in the model’s knowledge base.

- DEFORMATION evaluates the ability of models to use intact physical appearance knowledge to identify and reason when visual entities undergo deformation not captured in the model’s knowledge base.

- INCOMPLETE evaluates the ability of models to compare and contrast the complete layout and structure of image knowledge to identify and reason about missing parts and the correct layout of visual entities.

- BIOLOGICAL evaluates the ability of models to utilize image knowledge after biological transformations of the visual entities.

All the data was manually scraped, with multiple quality control stages.

There is a deep comparison of proprietary models vs open source ones. It seems proprietary models can better utilize retrieved images and are generally better at disregarding noisy images that might degrade the response.


