# An Easy Introduction to Multimodal Retrieval-Augmented Generation

## Reference

https://developer.nvidia.com/blog/an-easy-introduction-to-multimodal-retrieval-augmented-generation/

## Summary

Mentions different approaches for multimodal retrieval:

- Embed all modalities into the same vector space:
	- Use [[CLIP]] for instance to encode both text and images in the same vector space
	- Replace LLM by a MLLM for all question and answering.
	- 
- Ground all modalities into one primary modality:
	- Pick a primary modality based on the focus of the application and ground all other modalities in the primary modality. 
		- If the application revolves around text-based Q&A over PDFs, process text normally and create text descriptions and metadata for images.
		- Retrieval works off of the text description and metadata for the images, and regular text.
		- Key benefit: Helpful in answering objective questions. No need to tune a new model for embedding images and rank results from across different modalities.
		- Disadvantages: Preprocessing costs and losing some nuance from the image.

- Have separate stores for different modalities:
	- Have separate stores for different modalities, query all to retrieve top-_N_ chunks, use multimodal re-ranker to provide the most relevant chunks.
	- Simplifies the modeling process, but adds complexity (re-ranker).


Considers [[Pix2Struct]] for image embeddings
Explains how to build a basic pipeline for multimodal RAG
Mentions [[DePlot]] as well to better handle information on charts
