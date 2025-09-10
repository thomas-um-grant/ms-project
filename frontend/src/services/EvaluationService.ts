// Service for handling evaluation-related API calls

import { apiClient } from "./ApiClient";
import { MOCK_DATASETS, MOCK_METRICS } from "../utils/mockData";
import type {
  EvaluateRequest,
  EvaluateResponse,
  EvaluationResult,
} from "../types/api";
import type { BenchmarkDataset, MetricDefinition } from "../types/retriever";

/**
 * Service class for evaluation operations
 * Handles both real API calls and mock data for development
 */
export class EvaluationService {
  /**
   * Evaluates pipelines using the provided configuration
   */
  async evaluatePipelines(request: EvaluateRequest): Promise<EvaluateResponse> {
    return apiClient.evaluate(request);
  }

  /**
   * Gets available benchmark datasets
   * Returns mock data for development
   */
  async getAvailableDatasets(): Promise<BenchmarkDataset[]> {
    // In production, this would be a real API call
    // return apiClient.getDatasets();

    // For development, return centralized mock data
    return MOCK_DATASETS;
  }

  /**
   * Gets available evaluation metrics
   * Returns mock data for development
   */
  async getAvailableMetrics(): Promise<MetricDefinition[]> {
    // In production, this would be a real API call
    // return apiClient.getMetrics();

    // For development, return centralized mock data
    return MOCK_METRICS;
  }

  /**
   * Gets evaluation history from previous runs
   * Currently returns empty array
   */
  async getEvaluationHistory(): Promise<EvaluationResult[]> {
    // This would fetch from a backend endpoint that stores evaluation results
    // In a real implementation, this might include pagination, filtering, etc.
    return [];
  }
}

export const evaluationService = new EvaluationService();
