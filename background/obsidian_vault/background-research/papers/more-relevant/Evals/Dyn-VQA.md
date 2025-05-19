# Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Planning Agent
## Reference

https://arxiv.org/abs/2411.02937

## Summary

They propose a new Dyn-VQA dataset to benchmark multimodal RAG systems (1,452 dynamic questions that require complex multimodal knowledge retrieval for solution). 

In addition, they propose "OmniSearch", a self-adaptive planning agent for multimodal retrieval.
Issues: Non-adaptive Retrieval Queries -> The embedding always transforms all modalities into one primary one, which might be counter productive for certain types of questions.

## Notes

3 types of dynamic questions:
- Questions with rapidly changing answers:
	- Since the context knowledge of such question updates frequently, the retrieved content may be mixed with outdated and newer knowledge that is difficult to distinguish. This requires mARG methods to flexibly plan additional retrievals based on feedback from current retrieved content for further comprehension, rather than merely a one-time retrieval. 
- Questions requiring multi-modal knowledge:
	- The knowledge necessary by Dyn-VQA spans various modalities. This demands that mRAG methods retrieve knowledge across diverse modalities with tailored retrieval APIs, differing from most VQA datasets limited in seeking textual knowledge with multimodal questions.
- Multi-hop questions:
	- Questions in Dyn-VQA necessitate varied reasoning hops for solution, which entails that mRAG methods conduct various retrieval steps. While existing VQA datasets primarily focus on two-hop question, i.e., identifying visual concepts via text and then answering single-hop textual question.

OmniSearch:
The overall framework consists of three module. The planning agent is the core module that formulates sub-questions and plans the subsequent retrieval action based on real-world feedback (i.e., retrieved content or solver output). Actual retrieval action is execute by the retriever. Then, sub-question solver generates feedback on sub-question based on the retrieval content and update it to the planner.