# ColPali - Efficient Document Retrieval With Vision Language Models
## Reference

https://arxiv.org/pdf/2407.01449

## Summary

ColPali leverages visual document structure. It outperforms traditional RAG pipelines in visually rich domains.
It simplifies preprocessing and improving retrieval quality. However, it has storage, computational costs, and interpretability challenges.

Core Problem:
Current document retrieval systems focus primarily on text, using text embeddings to match queries with relevant documents. However, many documents contain rich visual elements (e.g., figures, tables, page layouts, fonts) that text-based models struggle to leverage effectively. Text-only retrieval pipelines rely on OCR (Optical Character Recognition), layout detection, and captioning models to extract structured text (quite slow).

Keypoints:
- Direct image embeddings:
	ColPali embeds document pages as images, bypassing traditional OCR-based text extraction.

- Multi-vector representations:
	Uses multi-vector embeddings per document page instead of a single dense vector, capturing fine-grained visual and textual details.

- Late interaction retrieval:
	Inspired by [[ColBERT]], it enables precise matching between query vectors and document embeddings, improving retrieval accuracy.

## Advantages

- No need for OCR or text preprocessing
It bypasses text extraction entirely by directly encoding document pages as images.

- Handles complex visual documents
It can retrieve pages where key information is not in text form but in figures, tables, and infographics.

- More efficient and faster indexing
Traditional retrieval requires text chunking, OCR, captioning, and layout detection, which are slow and expensive.
This skips these steps and directly indexes documents from images, making it faster and easier to deploy at scale.

- Multi-vector representations for better matching
Traditional RAG uses single-vector embeddings per chunk, losing fine-grained details.
This matches specific image regions to query terms, improving retrieval accuracy.

- End-to-End trainable
Traditional pipelines separate models for OCR, embedding, and retrieval.
This trains all components together, making it more robust and adaptable.

- Multilingual & domain-agnostic
Standard RAG struggles with non-English texts due to OCR limitations.
It learns from raw visual data, making it language-independent and better suited for diverse document types.


## Disadvantages

- Higher storage and memory requirements
It stores multi-vector embeddings per image, leading to larger index sizes than single-vector text embeddings.
It requires efficient compression techniques to scale to very large corpora.

- More computationally intensive retrieval
It uses late interaction, which requires computing many similarity scores per query.
It might have higher latency at large scale without optimization.

- Limited explainability
It retrieves images, which may require extra steps to interpret how a document matches a query.

- Training complexity
It requires pre-training and fine-tuning on document-image datasets, which may not be as readily available as text datasets.
Model performance is tied to the quality of its multi-modal vision-language training.

- Not ideal for purely text-based queries
If a dataset consists only of text with minimal visual structure, it won't perform as well.

## Questions

- Scalability and efficiency
How does ColPali perform at massive scale (millions of documents)?
Does multi-vector retrieval cause bottlenecks in large-scale applications like enterprise search or legal archives?
What are the best compression strategies for multi-vector embeddings?
Could techniques like clustering, quantization, or knowledge distillation reduce storage while keeping retrieval quality high?
Can ColPali be hybridized with traditional RAG for optimal performance?
Could it retrieve text and image-based representations together, choosing the best strategy based on query type?

- Generalization and Adaptability
How well does ColPali handle handwritten or scanned documents?
Traditional OCR often fails on handwritten text or poor-quality scans. Does ColPali perform better?
Can ColPali extend beyond documents to multi-modal RAG?
Could it integrate with speech, video, and structured data to improve retrieval in multi-modal RAG applications?
How does ColPali handle dynamic document updates?
If new documents arrive frequently, does the image-based approach require costly re-indexing?

- Accuracy and Fine-Grained Retrieval
Can ColPali explain why a document was retrieved?
Unlike text-based models that show highlighted words, could ColPali provide visual heatmaps to indicate relevant image regions?
Does late interaction improve retrieval accuracy across all document types?
Are there cases where bi-encoder (single vector) models outperform ColPali due to their efficiency?

- Future RAG Integration
Can ColPali support retrieval in long-context generative models?
Could it feed structured image-based context into long-context LLMs for better document question answering?
How would ColPali perform in an interactive, conversational RAG system?
Could users ask follow-up questions to refine retrieval, leveraging ColPali’s multi-vector outputs?

## Notes

ViDoRe benchmark - Practical tasks:
Topic-specific retrieval benchmarks spanning multiple domains (Energy, Government, Healthcare, AI, Shift project) beyond repurposed QA datasets.
They collect publicly accessible PDF documents and generate queries pertaining to document pages using Claude-3 Sonnet. 1,000 document pages per topic, which are associated with 100 queries extensively filtered for quality and relevance by human annotators.

Methodology:
1) Use a web crawler to collect publicly available documents on various themes and sources
	- GPT-3.5 Turbo to brainstorm related topics and subtopics.
	- GPT-3.5 Turbo to generate diverse search queries from each subtopic.
	- Query set consumed by parallel workers to fetch the most relevant documents, using SerpAPI17 along with a filetype filter (PDF documents only) to programmatically scrape Google Search rankings.
	- Each file is hashed and stored in a Bloom filter shared among workers to avoid duplicate documents in the final corpus.
	- Unique scraped files are downloaded, and inserted into a SQLite db with metadata.
	- Collected approximately 100 documents per topic ("energy", "government reports", "healthcare industry", "artificial intelligence")
2) Convert these PDFs into a series of images, one per page
	- Removed all documents containing any private information.
3) Generate queries related to each image using a VLM.
	- Generate at most 3 questions per image. From all the documents, randomly sample 10,000 images per theme and call Claude-3 Sonnet with the following prompt:

```
You are an assistant specialized in Multimodal RAG tasks.

The task is the following: given an image from a pdf page, you will have to
generate questions that can be asked by a user to retrieve information from
a large documentary corpus.
The question should be relevant to the page, and should not be too specific
or too general. The question should be about the subject of the page, and
the answer needs to be found in the page.

Remember that the question is asked by a user to get some information from a
large documentary corpus that contains multimodal data. Generate a question
that could be asked by a user without knowing the existence and the content
of the corpus.
Generate as well the answer to the question, which should be found in the
page. And the format of the answer should be a list of words answering the
question.
Generate at most THREE pairs of questions and answers per page in a dictionary
with the following format, answer ONLY this dictionary NOTHING ELSE:

{
	"questions": [
		{
			"question": "XXXXXX",
			"answer": ["YYYYYY"]
		},
		{
			"question": "XXXXXX",
			"answer": ["YYYYYY"]
		},
		{
			"question": "XXXXXX",
			"answer": ["YYYYYY"]
		},
	]
}
where XXXXXX is the question and ['YYYYYY'] is the corresponding list of answers
that could be as long as needed.

Note: If there are no questions to ask about the page, return an empty list.
Focus on making relevant questions concerning the page.
Here is the page:
```

- Human validation, manually validate every single one synthetically:
	- Randomly assign document-pair queries to 4 volunteer annotators and instruct them to filter out queries that do not fit the prompt criteria.
	- Instruct annotators to flag docs containing PII information or not suited for an academic benchmark. (No flag raised during the entirety of the process, validating the prior PDF collection strategy.)
	- 100 queries per topic collected in this manner. Each annotator spent approximately 3 hours filtering the larger query set down to 100 high-quality queries per topic.