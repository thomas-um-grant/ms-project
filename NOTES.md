# Additional notes regarding the process of this project

## 📓 Documentation & Logging Tools
💬 Daily Research Log – Obsidian folder, under `/background`
Will be keeping a journal:
* What was done today?
* What worked or failed?
* What’s the next step?

Structure (example):
```text
April 8
- Finished implementing graph chunk linker.
- Tried retrieval with 5-hop max. Performance dropped.
- TODO: Try re-ranking by local language model.
```

📊 Experiment Tracker – Weights & Biases (wandb) or MLflow
* Automatically log: configs, results, model versions.
* Add tags to each run: "graph_rag-v1", "baseline", "clip_retrieval".
* Helps with reproducibility and visual comparison.

## 🛠️ Core Libraries & Tools
For RAG:
* LangChain or LlamaIndex – pipelines, vector stores, document splitting.
* FAISS, Weaviate, or Milvus – vector retrieval.
* Neo4j or NetworkX – building and querying graphs.
For Multi-modal:
* CLIP, BLIP2, LLaVA – encode image/text, do joint embeddings.
* transformers by Hugging Face – for LLMs, tokenizers, etc.
* OpenAI / Anthropic APIs – for reliable baselines (e.g., GPT-4).
Visualization:
* Graphviz, NetworkX + Matplotlib, or Neo4j Bloom – for graph views.
* TensorBoard or Weights & Biases – log embeddings or loss curves.

## 🛎️ Task & Time Management
* Use Obsidian Kanban to track milestones & subtasks.
* Break down the work weekly into:
    * Literature Review
    * Graph Construction Prototype
    * Baseline Retrieval Eval
💡 Tip: Set one “high impact” goal per week. Avoid multitasking during deep work blocks.


## 🧠 Habits to Stay Effective
✅ Weekly Routine
Day	Focus
Mon–Tue	Build / experiment
Wed	Log insights, fix bugs
Thu	Compare results, tweak design
Fri	Summarize learnings & journal
Weekend	Read 1–2 papers or prototype
📌 Weekly Review Template
* What did I accomplish?
* What problems did I face?
* What’s working well?
* What should I change next week?

## 🧾 Final Advice
* Treat the repo and notes as if someone else will read and continue the work.
* Every experiment should answer a specific hypothesis.
* Start building the final report early — don’t wait till the end.


