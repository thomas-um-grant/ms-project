# PlotQA: Reasoning over Scientific Plots
## Reference

https://arxiv.org/abs/1909.00997

## Summary

28.9 million question-answer pairs over 224,377 plots on data from real world sources and questions based on crowd-sourced question templates. Roughly 80.76% of the questions have answers which are not present in the plot or in a fixed vocabulary.

Questions are generated based on 74 templates extracted from 7,000 crowd-sourced questions asked by workers on a sampled set of 1,400 plots.

Questions are categorized into (3x3) cells based on the question type:

| Structural Understanding | Data Retrieval | Reasoning |

and with the answer type: 

| Yes/No | From Fixed Vocabulary | Out Of Vocabulary (OOV) |

## Notes

Stages to build the set:
1) Curating data such as year-wise rainfall statistics, country-wise mortality rates, etc. Basically, extracting metrics that can be plotted, from various sources.
2) Creating different types of plots with a variation in the number of elements, legend positions, fonts, etc.
3) Crowd-sourcing to generate questions: Sampled 1400 plots across different types and asked workers to create questions for these plots. Showed each plot to 5 different workers resulting in a total of 7000 questions.
4) Extracting templates from the crowd-sourced questions and instantiating these templates using appropriate phrasing suggested by human annotators. The questions were manually analyzed  and divided into a total of 74 templates.

Question types:
- Structural Understanding:
	- The overall structure of the plot. Examples: “How many different colored bars are there?”
- Data Retrieval:
	- Seek data item for a single element in the plot. Examples: “What is the number of tax payers in Myanmar in 2015?”.
- Reasoning:
	- Require numeric reasoning over multiple plot elements or a comparative analysis of different elements of the plot, or a combination of both. Examples: “In which country is the number of threatened bird species minimum?”

Additionally paraphrased a bunch of questions manually, to end up with 28,952,641 questions.



