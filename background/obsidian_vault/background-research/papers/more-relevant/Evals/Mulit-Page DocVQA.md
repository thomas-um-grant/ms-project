# Hierarchical multimodal transformers for Multi-Page DocVQA
## Reference

https://arxiv.org/abs/2212.05935

## Summary

MP-DocVQA is an extension of the [[DocVQA]] dataset where the questions are posed on documents with between 1 and 20 pages. 46K questions posed over 48K images of scanned
pages that belong to 6K industry documents. 
Propose a new encoder-decoder approach Hi-VT5, which slightly outperform [[BERT]].

## Notes

Process:
During the annotation, they realized most questions are answered looking at a specific block within document, and annotators asked to respond to a query based on multiple documents seemed to not capture the essence of the queries as they were unnatural. Therefore they decided to create the new MP-DocVQA dataset by taking every image-question pair
from [[DocVQA]] and added to every image the previous and posterior pages of the document downloaded from the original source. 

So basically it is showing the performance decrease with more noise, and the encoder-decoder reduces the performance drop. This is not mentioning evaluation with queries spanning multiple documents.
