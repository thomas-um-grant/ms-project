# ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning
## Reference

https://arxiv.org/abs/2203.10244

## Summary

Benchmark 9,608 human-written questions as well as 23,111 questions generated from human-written chart summaries using a T5 model and manually validating a subset of it for quality assurance. 20,882 charts which are curated from four different online sources (Statista- statista.com; The Pew research - pewresearch.org; OWID - ourworldindata.org; OECD - oecd.org) to ensure variety in visual styles and topics.

## Notes

Data annotation:
- Collect human-authored QA pairs using Amazon Mechanical Turk (AMT)
- Generate QA pairs from the Statista human-written summaries.

Identifying to question types:
- Compositional: contain at least two logical operations like sum, difference and average.
- Visual: visual attributes such as color, height, and length of graphical marks (e.g., bars).

Two workers answer the same question, if the answers do not match, then the answer is not considered correct and dismissed.

A lot of manual work is done to clean the dataset

