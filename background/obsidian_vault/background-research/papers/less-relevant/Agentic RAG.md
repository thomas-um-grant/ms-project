# Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG
## Reference
https://arxiv.org/abs/2501.09136

## Summary

This survey explores Agentic Retrieval-Augmented Generation (Agentic RAG), which integrates autonomous AI agents into the RAG pipeline. By leveraging design patterns such as reflection, planning, tool use, and multi-agent collaboration, Agentic RAG systems dynamically manage retrieval strategies and adapt workflows to meet complex task requirements, offering enhanced flexibility and context awareness across various applications.

## Advantages

• Autonomous Decision-Making:
	Agents independently evaluate and manage retrieval strategies based on query complexity.
• Iterative Refinement:
	Incorporates feedback loops to improve retrieval accuracy and response relevance.
• Workflow Optimization:
	Dynamically orchestrates tasks, enabling efficiency in real-time applications

MultiAgent:
• Modularity: 
	Each agent operates independently, allowing for seamless addition or removal of agents based on system requirements.
• Scalability: 
	Parallel processing by multiple agents enables the system to handle high query volumes efficiently.
• Task Specialization:
	Each agent is optimized for a specific type of query or data source, improving accuracy and retrieval relevance.
• Efficiency:
	By distributing tasks across specialized agents, the system minimizes bottlenecks and enhances performance for complex workflows.
• Versatility:
	Suitable for applications spanning multiple domains, including research, analytics, decision-making, and customer support

## Disadvantages

• Coordination Complexity:
	Managing interactions between agents requires sophisticated orchestration mechanisms.
• Computational Overhead:
	The use of multiple agents increases resource requirements for complex workflows.
• Scalability Limitations:
	While scalable, the dynamic nature of the system can strain computational resources for high query volumes

MultiAgent:
• Coordination Complexity:
	Managing inter-agent communication and task delegation requires sophisticated orchestration mechanisms.
• Computational Overhead:
	Parallel processing of multiple agents can increase resource usage.
• Data Integration:
	Synthesizing outputs from diverse sources into a cohesive response is non-trivial and requires advanced LLM capabilities.

## Questions

- Agent Behavior & Emergent Dynamics
	- How do agentic behaviors (e.g., reflection or planning) interact over time in open-ended workflows?
	- What emergent behaviors arise when agents reflect on each other’s outputs iteratively?
	- Can agent collectives develop implicit norms or “division of labor” without explicit orchestration?

- Control vs. Autonomy
	- What is the right balance between centralized control (e.g., orchestrators) and fully autonomous agent operation in dynamic environments?
	- How can we ensure consistency or factuality in outputs when agents follow divergent paths or tools?
    
- Modularity and Composability
	- What mechanisms are needed to make Agentic RAG systems compositional, such that agents can be reused or recombined across domains?
	- Can agents learn to adapt their roles over time in changing task environments without retraining?
	    
- Evaluation & Benchmarking
	- How do we measure "effective collaboration" or "agent contribution" in multi-agent RAG systems?
	- What are good unit tests for evaluating the agentic aspects—like planning effectiveness, reflection depth, or reasoning loops?
    
- Cognitive Modeling
	- Can Agentic RAG architectures serve as models for distributed cognition or theory of mind in LLMs?
	- How do agentic patterns map to human problem-solving strategies in real-world domains (e.g., law, science, negotiation)?
    
- Ethics, Alignment, and Governance
	- How can we align decentralized agentic systems with human goals when no single agent has full oversight?
	- Who is accountable when autonomous agents generate decisions through distributed reasoning and no individual component is clearly at fault?
    
- Infrastructure and Ecosystem Gaps
	- What infrastructure is needed for real-time, cross-agent memory sharing without introducing race conditions or information leakage?
	- How can we simulate complex real-world environments (e.g., real-time web, evolving knowledge graphs) to train and test Agentic RAG agents?
    
- Learning and Adaptation
	- Can agents learn to improve not just their outputs but also their collaboration protocols over time?
	- How can Agentic RAG systems support continual learning without catastrophic forgetting or loss of task coherence?