/**
 * Mock data generation utilities for development and testing
 * Provides consistent mock data across components without duplication
 */

import type {
  EvaluationResult,
  RetrievedDocument,
  ChatMessage,
} from "../types/api";
import type { BenchmarkDataset, MetricDefinition } from "../types/retriever";

/**
 * Generates mock evaluation results for the stats page
 */
export function generateMockEvaluationResults(
  datasets: string[],
  metrics: string[],
  pipelines: string[] = ["traditional", "multimodal"]
): EvaluationResult[] {
  const results: EvaluationResult[] = [];

  for (const dataset of datasets) {
    for (const pipeline of pipelines) {
      const metricsData: Record<string, number> = {};

      for (const metric of metrics) {
        // Generate realistic scores between 0.3 and 0.95
        metricsData[metric] =
          Math.round((0.3 + Math.random() * 0.65) * 100) / 100;
      }

      results.push({
        pipeline_name: pipeline,
        dataset_name: dataset,
        metrics: metricsData,
        timestamp: new Date().toISOString(),
      });
    }
  }

  return results;
}

/**
 * Generates mock retrieved documents for chat
 */
export function generateMockRetrievedDocuments(
  count: number = 3
): RetrievedDocument[] {
  const mockDocuments = [
    {
      content:
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn and improve from experience without being explicitly programmed.",
      metadata: {
        source: "ml_fundamentals.pdf",
        page: 1,
        section: "Introduction",
      },
    },
    {
      content:
        "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information using connectionist approaches.",
      metadata: {
        source: "neural_networks.pdf",
        page: 15,
        section: "Architecture",
      },
    },
    {
      content:
        "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
      metadata: {
        source: "deep_learning_guide.pdf",
        page: 3,
        section: "Overview",
      },
    },
    {
      content:
        "Natural language processing (NLP) combines computational linguistics with machine learning and deep learning to help computers understand human language.",
      metadata: {
        source: "nlp_handbook.pdf",
        page: 8,
        section: "Introduction",
      },
    },
    {
      content:
        "Retrieval-Augmented Generation (RAG) combines information retrieval with language generation to produce more accurate and contextually relevant responses.",
      metadata: { source: "rag_systems.pdf", page: 12, section: "Methodology" },
    },
  ];

  return Array.from({ length: count }, (_, index) => {
    const doc = mockDocuments[index % mockDocuments.length];
    return {
      id: `doc_${Date.now()}_${index}`,
      content: doc.content,
      score: Math.round((0.7 + Math.random() * 0.3) * 100) / 100, // Score between 0.7-1.0
      metadata: doc.metadata,
    };
  });
}

/**
 * Generates a mock chat response
 */
export function generateMockChatResponse(query: string): {
  answer: string;
  retrieved_documents: RetrievedDocument[];
} {
  const responses = [
    "Based on the retrieved documents, I can provide you with comprehensive information about this topic. The sources indicate several key points that are relevant to your question.",
    "According to the documentation, this concept involves multiple interconnected components that work together to achieve the desired outcome.",
    "The retrieved sources suggest that this approach has been proven effective in various applications and contexts.",
    "From the available literature, it appears that this methodology combines several established techniques to provide improved results.",
  ];

  return {
    answer: responses[Math.floor(Math.random() * responses.length)],
    retrieved_documents: generateMockRetrievedDocuments(),
  };
}

/**
 * Simulates async API delay for realistic UX
 */
export function simulateDelay(
  min: number = 500,
  max: number = 2000
): Promise<void> {
  const delay = Math.floor(Math.random() * (max - min + 1)) + min;
  return new Promise((resolve) => setTimeout(resolve, delay));
}

/**
 * Available benchmark datasets (mock data)
 */
export const MOCK_DATASETS: BenchmarkDataset[] = [
  {
    name: "scifact",
    description: "Scientific fact verification dataset",
    domain: "Science",
    total_queries: 300,
    total_documents: 5183,
    metrics_available: ["ndcg@10", "recall@10", "precision@10", "map"],
  },
  {
    name: "nfcorpus",
    description: "Nutrition facts corpus",
    domain: "Health",
    total_queries: 323,
    total_documents: 3633,
    metrics_available: ["ndcg@10", "recall@10", "precision@10", "map"],
  },
  {
    name: "trec-covid",
    description: "COVID-19 information retrieval dataset",
    domain: "Medical",
    total_queries: 50,
    total_documents: 171332,
    metrics_available: ["ndcg@10", "recall@10", "precision@10", "map"],
  },
  {
    name: "arguana",
    description: "Argument retrieval dataset",
    domain: "Argumentation",
    total_queries: 1406,
    total_documents: 8674,
    metrics_available: ["ndcg@10", "recall@10", "precision@10", "map"],
  },
];

/**
 * Available evaluation metrics (mock data)
 */
export const MOCK_METRICS: MetricDefinition[] = [
  {
    name: "accuracy",
    description: "Overall accuracy of the responses",
    range: "0.0 - 1.0",
    higher_is_better: true,
  },
  {
    name: "relevance",
    description: "Relevance of retrieved documents",
    range: "0.0 - 1.0",
    higher_is_better: true,
  },
  {
    name: "completeness",
    description: "Completeness of the generated answers",
    range: "0.0 - 1.0",
    higher_is_better: true,
  },
  {
    name: "ndcg_10",
    description: "Normalized Discounted Cumulative Gain at 10",
    range: "0.0 - 1.0",
    higher_is_better: true,
  },
];
