# Learning Transferable Visual Models From Natural Language Supervision
## Reference

https://arxiv.org/abs/2103.00020

## Summary

This  introduces a novel approach to learning visual representations by leveraging natural language supervision. It is built by jointly training an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. It is much more efficient at zero-shot transfer than the image caption baseline.

## Notes

Novelty:
- Contrastive Pre-training Objective
	- Contrastive learning framework where it learns to associate images with their corresponding textual descriptions.

- Dual Encoder Architecture
	- Image Encoder:
		- Processes images using architectures like ResNet or Vision Transformers (ViT) to produce image embeddings.
	- Text Encoder:
		- Processes text using a Transformer-based model to produce text embeddings.

- Zero-Shot Transfer Capability
	- Image Classification:
		- By comparing the image embedding to embeddings of textual class descriptions (e.g., "a photo of a cat"), CLIP can classify images in a zero-shot manner.
	- Cross-modal Retrieval:
		- CLIP can retrieve relevant images based on text queries and vice versa.
	- Image Captioning and Generation:
		- CLIP's embeddings can be used to guide models for generating captions or images.

Limitations:
- Fine-Grained Classification:
	- CLIP may struggle with tasks requiring fine-grained distinctions between similar classes.
- Biases:
	- As CLIP is trained on internet data, it can inherit and even amplify societal biases present in the data.
- Performance on Specific Datasets:
	- CLIP underperforms on datasets like MNIST, which differ significantly from its training data.