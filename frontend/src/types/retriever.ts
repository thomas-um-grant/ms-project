// Types for pipeline configurations and retrieval strategies

export interface PipelineConfig {
  pipeline_type: PipelineType;
  embedding_model: string;
  llm_model: string;
  chunk_size?: number;
  chunk_overlap?: number;
  top_k?: number;
  temperature?: number;
  additional_params?: Record<string, any>;
}

export type PipelineType = "traditional" | "multimodal" | "multi";

export interface PipelineOption {
  label: string;
  value: PipelineType;
  description: string;
}

export interface ModelOption {
  label: string;
  value: string;
  provider: string;
}

export interface MetricDefinition {
  name: string;
  description: string;
  range: string;
  higher_is_better: boolean;
}

export interface BenchmarkDataset {
  name: string;
  description: string;
  domain: string;
  total_queries: number;
  total_documents: number;
  metrics_available: string[];
}

export interface ConfigurationState {
  selectedPipeline: PipelineType;
  selectedEmbeddingModel: string;
  selectedLLMModel: string;
  chunkSize: number;
  chunkOverlap: number;
  topK: number;
  temperature: number;
}
