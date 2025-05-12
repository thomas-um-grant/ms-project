# SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images
## Reference

https://arxiv.org/abs/2301.04883

## Summary

Propose a VQA dataset containing 2.6k+ slide decks (each consisting of 20 slides = 52k+ slide images) and 14.5k questions about a slide deck. It requires complex reasoning, including single-hop, multi-hop, and numerical reasoning, and also provides annotated arithmetic expressions of numerical answers for enhancing the ability of numerical reasoning. 

## Notes

Answer types:
- Single-span:
	- Is a contiguous sequence of tokens in the reading order extracted from the image
- Multi-span:
	- Is formed from multiple spans from the image.
- Non-span:
	- Is not extracted and is composed of numerical values and visual appearances.

Data collection:
They recruited crowd workers located in English-speaking countries.

- Selected and downloaded 25,327 slide decks composed of more than 20 slides from slideshare and covering 39 topics. Only kept the first 20 slides each time and truncated the rest.
- Workers filtered the decks that did not meet the following criteria:
	- The main language is English
	- The content is easy for workers to understand
	- The decks must contain one or more graphs, tables, figures, or numerical data to avoid creating questions requiring only text-level understanding.
	- Workers annotated bounding boxes around different part of slides (title, caption, diagram, image, etc...)

Single-hop QA creation:
- Workers created 12,466 QA pairs by selecting a single slide image from a slide deck. The selected slide can be used as evidence to tell whether a system arrived at the right answer for the right reasons. Encouraged questions that needed numerical reasoning. They avoided creating questions that:
	- contained selected page numbers
	- required external knowledge
	- were common to all of the slides (e.g., “What is the title?”).

Multi-hop QA creation:
- Created 2,018 QA pairs for multi-hop reasoning by editing the single-hop questions created in the previous step. For example, a question is rephrased to enforce bridges in between multiple documents to answer a question.

Another set of workers were asked to respond to the questions and selecting relevant slides, to evaluate the human performance on the dataset.

