// API request and response types

import type { PipelineConfig } from "./retriever";

export interface IngestRequest {
  documents: Document[];
  pipeline_config: PipelineConfig;
}

export interface QueryRequest {
  query: string;
  pipeline_config: PipelineConfig;
  top_k?: number;
}

export interface QueryResponse {
  answer: string;
  retrieved_documents: RetrievedDocument[];
  metadata: {
    pipeline_used: string;
    query_time: number;
    total_documents: number;
  };
}

export interface EvaluateRequest {
  dataset_name: string;
  pipeline_configs: PipelineConfig[];
  metrics: string[];
}

export interface EvaluationResult {
  pipeline_name: string;
  dataset_name: string;
  metrics: {
    [metric: string]: number;
  };
  timestamp: string;
}

export interface EvaluateResponse {
  results: EvaluationResult[];
  summary: {
    total_queries: number;
    evaluation_time: number;
  };
}

export interface Document {
  id: string;
  content: string;
  metadata?: Record<string, any>;
}

export interface RetrievedDocument {
  id: string;
  content: string;
  score: number;
  metadata?: Record<string, any>;
}

export interface ChatMessage {
  id: string;
  type: "user" | "assistant";
  content: string;
  timestamp: Date;
  retrieved_documents?: RetrievedDocument[];
}

export interface Dataset {
  name: string;
  description: string;
  total_queries: number;
  total_documents: number;
}
