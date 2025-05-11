# InfographicVQA
## Reference

https://arxiv.org/abs/2104.12756

## Summary

InfographicVQA, comprising 30035 questions over 5485 images, from 2594 distinct
web domains. The images were extracted from the internet simply searching using the keyword "infographics". Then the images were filtered and workers annotated them.

## Notes

Annotation process:
2 steps:
- 1) Annotators are asked to create questions-answers pairs for each of the images
- 2) Some questions were asked to other annotators, they had to answer of flag the question if not relevant enough for an inforgraphics. The annotators were also tasked with given the question-answer a type (can have multiple types): Answer-source, Evidence or Operation:
	- Answer-source: Image-span, Question-span, Multi-span and Non-extractive.
	- Evidence: indicates the kind of evidence behind the answer.
	- Operation: captures the kind of discrete operation(s) required to arrive at an answer (Counting, Arithmetic or Sorting).

Interesting to have human also answer to evaluate their performances compare to models to ensure the legitimacy of the dataset.