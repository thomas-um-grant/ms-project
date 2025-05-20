# VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents
## Reference

https://arxiv.org/abs/2504.09795

## Summary

Present a new framework using VLMs like ColPali. Stating that the current approaches have trouble understanding diverse real world documents due to the limitations of their datasets and training strategies. ViDoRe is also acknowledged and seem to contain questions that might not require retrieval and handles a limited number of document types, resulting in a gap between real-world scenarios.

The dataset they introduce (OpenDocVQA) is verified by humans and more robust to context independent conditions.

To train the VLM used for the embedding, they focus on pre-training tasks that transfer the understanding and generation capabilities of LVLMs to retrievers.
## Notes

Dataset Collection:
- Filtering of DocumentVQA datasets:
	- Applied heuristic rules to automatically select likely context-independent questions, reducing the pool by 20.9%. Then, we manually reviewed and verified the remaining examples to ensure their context independence.
- Reformulation of TableQA dataset:
	- Took the screenshot images of tables from the correspondingWikipedia pages to reformulate the task as the OpenDocVQA.
- Creation of new multi-hop questions:
	- Semi-automatically created a multi-hop DocumentVQA dataset, MHDocVQA, using the single-hop QA pairs collected in the previous steps.
		- First used spaCy to identify a bridge entity in the answer to a single-hop question and then searched for this entity in other single-hop questions.
		- Next, used Mixtral model to combine the two single-hop questions.
		- Then filtered the generated multi-hop questions using another LLM (GPT-4o), which answered the questions based on the context of the two initial single-hop questions and their answers.
			- Validated only if If the predicted answer was the same as the answer to the second single-hop question.
		- Manually reviewed the filtered questions for quality.
- Negative candidates mining:
	- Produced negative image candidates for retrievers to sift through for every question, used only during inference.
		- Extracted OCR text from images in the COYO-700M dataset
		- Mined negative images where the OCR text exhibits high lexical overlap with the question but does not contain the correct answer.

3 key advantage of the dataset:
- First large-scale collection of open-domain DocumentVQA datasets to address open document types (ViDoRe considers six document types for only the retrieval task).
- The questions are context independent and require visual semantic search, whereas ViDoRe’s questions are context-dependent. Therefore it better reflects real-world scenarios.
- Requires multi-hop reasoning with extractive (e.g., span, list) and abstractive (e.g., arithmetic, counting, no answer) answer types, providing a more challenging setting.

Model Architecture:
- Dynamic high-resolution image encoding (split into chunks of 336x336)
- VDocRetriever (LVLM-based dual-encoder architecture that encodes queries and document images independently.)
- VDocGenerator (adapts LVLM to generate answers for a question and the retrieved k documents obtained from VDocRetriever.)
- This is passed to the LLM for the final generation

Pre-training tasks:
- Representation Compression via Retrieval (RCR).
- Representation Compression via Generation (RCG).