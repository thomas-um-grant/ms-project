# DocVQA: A Dataset for VQA on Document Images
## Reference

https://arxiv.org/abs/2007.00398

## Summary

DocVQA is a large-scale dataset created to enable Visual Question Answering (VQA) on document images, requiring both textual and structural understanding. It comprises 50,000 questions defined on over 12,000 document images, primarily sourced from the UCSF Industry Documents Library across five industries. The goal is to simulate ad hoc queries about real-world documents, promoting a "purpose-driven" approach to document understanding.

## Advantages

- Covers diverse document types (forms, tables, handwritten notes, reports, etc.).

- Includes layout-sensitive questions requiring spatial reasoning.

- Supports open-ended, extractive answers, not constrained by a closed vocabulary.

- Promotes structure-aware and multi-modal reasoning, advancing beyond pure text extraction.

- Offers fine-grained evaluation with human baselines and metrics like ANLS to account for OCR variability.

## Disadvantages

- Documents mainly span 1960–2000, limiting temporal and domain diversity (tobacco, food, drug, fossil fuel and chemical).

- Heavy reliance on OCR; OCR errors can limit model performance and answer accuracy.

- Lacks support for longer, compositional reasoning chains, unlike some recent datasets.

## Notes

Questions-Answers:
Annotation process, using remote workers (Amazon mainly) through web based annotation tool:
- Stage 1:
	- Workers were shown a document image and asked to define at most 10 question-answer pairs on it.
	- Encouraged to extract answers verbatim and include multiple valid answers if needed.

- Stage 2: 
	- Verification step: new annotators re-answer questions without seeing stage-1 answers.
	- Assign one or more question types per QA pair; can flag invalid/ambiguous items.
	
- Stage 3:
	- Questions with no answer match between stages 1 & 2 are reviewed and edited by authors.

Metrics used in Experiments:
- ANLS (Average Normalized Levenshtein Similarity):
	- Measures soft similarity, useful when minor OCR/formatting issues exist.
- Accuracy:
	- Measures exact match; more strict.

