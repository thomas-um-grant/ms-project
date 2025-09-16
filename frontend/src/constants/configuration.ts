/**
 * Centralized configuration constants for the RAG evaluation platform
 * This file contains all configuration options, defaults, and static data
 * to avoid duplication across components.
 */

import type { PipelineType } from "../types/retriever";

export interface DropdownOption {
  label: string;
  value: string | number;
  description?: string;
}

export interface ConfigurationOption extends DropdownOption {
  value: string;
}

/**
 * Pipeline type options for the RAG system
 */
export const PIPELINE_OPTIONS: ConfigurationOption[] = [
  {
    label: "Traditional RAG",
    value: "traditional",
    description: "Standard RAG pipeline",
  },
  {
    label: "Multi-Modal RAG",
    value: "multimodal",
    description: "Text and image RAG based on ColPali",
  },
  {
    label: "Multi RAG",
    value: "multi",
    description: "Combination of RAG systems",
  },
];

/**
 * Available embedding model options
 */
export const EMBEDDING_MODEL_OPTIONS: ConfigurationOption[] = [
  {
    label: "Nomic Embed Text (Ollama)",
    value: "nomic-embed-text",
    description:
      "High-performance text embedding model with strong retrieval capabilities and context length",
  },
  {
    label: "ColQwen2 (ColPali)",
    value: "vidore/colqwen2-v1.0",
    description:
      "Multi-modal vision-language model optimized for document understanding and visual question answering",
  },
];

/**
 * Available LLM model options
 */
export const LLM_MODEL_OPTIONS: ConfigurationOption[] = [
  {
    label: "LLama 3 (Ollama)",
    value: "llama3.2:3b",
    description:
      "Meta's latest 3B parameter model with improved instruction following and efficiency",
  },
  {
    label: "Qwen2 Instruct (Ollama)",
    value: "Qwen/Qwen2-VL-7B-Instruct",
    description:
      "Alibaba's 7B vision-language model with strong multimodal reasoning and code capabilities",
  },
];

/**
 * Text chunking strategy options
 */
export const CHUNKING_OPTIONS: ConfigurationOption[] = [
  {
    label: "Fixed Size",
    value: "fixed",
    description: "Split text into fixed-size chunks",
  },
  {
    label: "Semantic",
    value: "semantic",
    description: "Split based on semantic boundaries",
  },
  {
    label: "Recursive",
    value: "recursive",
    description: "Recursively split text maintaining structure",
  },
];

/**
 * Retrieval strategy options
 */
export const RETRIEVAL_OPTIONS: ConfigurationOption[] = [
  {
    label: "Vector",
    value: "vector",
    description: "Cosine similarity retrieval",
  },
  {
    label: "BM25",
    value: "bm25",
    description: "Sparse vector (BM25-style) retrieval",
  },
  {
    label: "Hybrid",
    value: "hybrid",
    description: "Combination of vector and BM25 retrieval",
  },
];

/**
 * Reranking strategy options
 */
export const RERANKING_OPTIONS: ConfigurationOption[] = [
  {
    label: "No Reranking",
    value: "none",
    description: "Skip reranking step",
  },
  {
    label: "LLM Reranking",
    value: "llm",
    description: "LLM based reranking",
  },
  {
    label: "Jina Reranking",
    value: "jina",
    description: "Jina based reranking",
  },
];

/**
 * Default configuration values
 */
export const DEFAULT_CONFIG = {
  selectedPipeline: "traditional" as PipelineType,
  selectedEmbeddingModel: "nomic-embed-text",
  selectedLLMModel: "llama3.2:3b",
  retrievalStrategy: "hybrid",
  rerankingStrategy: "none",
  chunkingStrategy: "page",
  chunkSize: 512,
  chunkOverlap: 50,
  topK: 5,
  temperature: 0.8,
} as const;

/**
 * Available datasets for evaluation
 */
export const AVAILABLE_DATASETS = [
  "msmarco",
  "nfcorpus",
  "scidocs",
  "arxivqa",
  "docvqa",
  "infovqa",
  "tabfquad",
  // "tatdqa",
  "esg_reports_v1",
  "esg_reports_v2",
  "economics_reports_v1",
  "economics_reports_v2",
  "biomedical_lectures_v1",
  "biomedical_lectures_v2",
  "consulting_light_v1",
  "consulting_light_v2",
  "consulting_v1",
  "consulting_v2",
];

/**
 * Default datasets to be selected on page load for evaluation tables
 * You can modify this list to change which datasets are selected by default
 */
export const DEFAULT_SELECTED_DATASETS = [
  // "msmarco",
  "nfcorpus",
  "scidocs",
  // "arxivqa",
  // "docvqa",
  "infovqa",
  "tabfquad",
  // "tatdqa",
  // "esg_reports_v1",
  "esg_reports_v2",
  // "economics_reports_v1",
  "economics_reports_v2",
  // "biomedical_lectures_v1",
  "biomedical_lectures_v2",
  "consulting_light_v1",
  "consulting_light_v2",
  "consulting_v1",
  "consulting_v2",
];

/**
 * Default RAG systems to be selected for detailed analysis dashboard
 * You can modify this list to change which RAG systems are selected by default
 * in the "Key Metrics Across RAG Systems" section
 */
export const DEFAULT_SELECTED_RAG_SYSTEMS = [
  "traditional_hybrid",
  "traditional_hybrid_rerank_jina",
  "multimodal",
  "multimodal_rerank_jina",
  "multi_rank_max",
  "multi_norm_avg_l2",
  "multi_rank_fuse_l2",
  // "multi_rrf",
  "multi_rrf_30",
];

/**
 * Default dataset for detailed analysis dashboard
 * You can modify this to change which dataset is selected by default
 * in the "Key Metrics Across RAG Systems" section
 */
export const DEFAULT_ANALYSIS_DATASET = "consulting_light_v2";

/**
 * Available evaluation metrics
 */
export const AVAILABLE_METRICS: DropdownOption[] = [
  {
    label: "nDCG at 1",
    value: "ndcg_at_1",
  },
  {
    label: "nDCG at 3",
    value: "ndcg_at_3",
  },
  {
    label: "nDCG at 5",
    value: "ndcg_at_5",
  },
  {
    label: "nDCG at 10",
    value: "ndcg_at_10",
  },
  {
    label: "nDCG at 20",
    value: "ndcg_at_20",
  },
  {
    label: "nDCG at 50",
    value: "ndcg_at_50",
  },
  {
    label: "nDCG at 100",
    value: "ndcg_at_100",
  },
  {
    label: "MAP at 1",
    value: "map_at_1",
  },
  {
    label: "MAP at 3",
    value: "map_at_3",
  },
  {
    label: "MAP at 5",
    value: "map_at_5",
  },
  {
    label: "MAP at 10",
    value: "map_at_10",
  },
  {
    label: "MAP at 20",
    value: "map_at_20",
  },
  {
    label: "MAP at 50",
    value: "map_at_50",
  },
  {
    label: "MAP at 100",
    value: "map_at_100",
  },
  {
    label: "Recall at 1",
    value: "recall_at_1",
  },
  {
    label: "Recall at 3",
    value: "recall_at_3",
  },
  {
    label: "Recall at 5",
    value: "recall_at_5",
  },
  {
    label: "Recall at 10",
    value: "recall_at_10",
  },
  {
    label: "Recall at 20",
    value: "recall_at_20",
  },
  {
    label: "Recall at 50",
    value: "recall_at_50",
  },
  {
    label: "Recall at 100",
    value: "recall_at_100",
  },
  {
    label: "Precision at 1",
    value: "precision_at_1",
  },
  {
    label: "Precision at 3",
    value: "precision_at_3",
  },
  {
    label: "Precision at 5",
    value: "precision_at_5",
  },
  {
    label: "Precision at 10",
    value: "precision_at_10",
  },
  {
    label: "Precision at 20",
    value: "precision_at_20",
  },
  {
    label: "Precision at 50",
    value: "precision_at_50",
  },
  {
    label: "Precision at 100",
    value: "precision_at_100",
  },
  {
    label: "MRR at 1",
    value: "mrr_at_1",
  },
  {
    label: "MRR at 3",
    value: "mrr_at_3",
  },
  {
    label: "MRR at 5",
    value: "mrr_at_5",
  },
  {
    label: "MRR at 10",
    value: "mrr_at_10",
  },
  {
    label: "MRR at 20",
    value: "mrr_at_20",
  },
  {
    label: "MRR at 50",
    value: "mrr_at_50",
  },
  {
    label: "MRR at 100",
    value: "mrr_at_100",
  },
];

/**
 * Configuration validation constraints
 */
export const CONFIG_CONSTRAINTS = {
  topK: { min: 1, max: 20 },
  chunkSize: { min: 100, max: 2000 },
  chunkOverlap: { min: 0, max: 500 },
  temperature: { min: 0, max: 2, step: 0.1 },
} as const;
