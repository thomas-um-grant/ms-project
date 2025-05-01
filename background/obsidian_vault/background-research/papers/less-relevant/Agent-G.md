# Agent-G: An Agentic Framework for Graph Retrieval Augmented Generation
## Reference

https://openreview.net/forum?id=g2C947jjjQ

## Abstract

Given two knowledge sources, one containing unstructured documents and the other comprising structured graph knowledge bases, how can we effectively retrieve the relevant information to answer user questions? While Retrieval-Augmented Generation (RAG) retrieves documents to assist the large language model (LLM) in question answering, Graph RAG (GRAG) uses graph knowledge bases as an additional knowledge source. However, there are many questions that require information from both sources, which complicates the scenario and makes hybrid retrieval essential. The goal is to effectively leverage both sources to provide better answers to the questions. Therefore, we propose Agent-G, a unified framework for GRAG, composed of an agent, a retriever bank, and a critic module. Agent-G has the following advantages:

- Agentic: it automatically improves the agent's action with self-reflection,
- Adaptive: it solves questions that require hybrid knowledge source with a single unified framework,
- Interpretable: it justifies decision making and reduces hallucinations, and
- Effective: it adapts to different GRAG settings and outperforms all baselines.

The experiments are conducted on two real-world GRAG benchmarks, namely STaRK and CRAG. In STaRK, Agent-G shows relative improvements in Hit@1 of 47% in STaRK-MAG and 55% in TaRK-Prime. In CRAG, Agent-G increases accuracy by 35% while reducing hallucination by 11%, both relatively.
