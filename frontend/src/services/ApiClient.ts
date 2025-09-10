// Base API client for communicating with the backend

import type {
  IngestRequest,
  QueryRequest,
  QueryResponse,
  EvaluateRequest,
  EvaluateResponse,
} from "../types/api";

class ApiClient {
  private baseURL: string;

  constructor(baseURL: string = "http://localhost:8000") {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;

    const config: RequestInit = {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("API request failed:", error);
      throw error;
    }
  }

  async ingest(data: IngestRequest): Promise<{ message: string }> {
    return this.request("/api/v1/ingest", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async query(data: QueryRequest): Promise<QueryResponse> {
    return this.request("/api/v1/query", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async evaluate(data: EvaluateRequest): Promise<EvaluateResponse> {
    return this.request("/api/v1/evaluate", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }
  // NOTE: getSettings / updateSettings methods were removed as they were unused.
  // Reintroduce if backend settings endpoints are implemented.
}

// Create and export a singleton instance
export const apiClient = new ApiClient();
export default ApiClient;
