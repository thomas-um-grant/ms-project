// Base API client for communicating with the backend

import type {
  IngestRequest,
  QueryRequest,
  QueryResponse,
  EvaluateRequest,
  EvaluateResponse,
  RetrievedDocument,
} from "../types/api";

class ApiClient {
  private baseURL: string;

  constructor(baseURL?: string) {
    // Prefer env var, else fallback to relative path so Vite proxy handles dev
    const envBase = (import.meta as any).env?.VITE_API_BASE_URL as
      | string
      | undefined;
    this.baseURL = baseURL ?? envBase ?? "";
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

  // Backend: GET /health
  async health(): Promise<{ status: string; state?: any }> {
    return this.request("/health", { method: "GET" });
  }

  // Backend: POST /rag/select
  async selectRag(data: {
    config: string;
    knowledge_base: string;
  }): Promise<any> {
    return this.request("/rag/select", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // Backend: POST /index
  async index(data: {
    config: string;
    knowledge_base: string;
    documents?: string[];
    folder_path?: string;
    set_current?: boolean;
  }): Promise<any> {
    return this.request("/index", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // Backend: POST /retrieve
  async retrieve(data: { query: string; top_k?: number }): Promise<{
    rag: string;
    knowledge_base: string;
    top_k: number;
    results: Array<{ metadata: Record<string, any>; score: number }>;
  }> {
    return this.request("/retrieve", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async ingest(data: IngestRequest): Promise<{ message: string }> {
    return this.request("/api/v1/ingest", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // Keep existing method name for UI, but route to backend /answer and adapt the response
  async query(data: QueryRequest): Promise<QueryResponse> {
    const started = Date.now();
    const payload = {
      query: data.query,
      top_k: data.top_k ?? data.pipeline_config?.top_k,
    } as { query: string; top_k?: number };

    const backend = await this.request<{
      rag: string;
      knowledge_base: string;
      top_k: number;
      results: Array<{ metadata: Record<string, any>; score: number }>;
      answer: string;
    }>("/answer", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    const toContent = (meta: Record<string, any>): string => {
      // Try common fields, otherwise show a compact metadata preview
      return (
        meta.content ?? meta.response ?? (meta.title ? `${meta.title}` : null)
      );
    };

    const retrieved: RetrievedDocument[] = (backend.results || []).map(
      (r, i) => ({
        id: String(r.metadata?.id ?? r.metadata?.doc_id ?? i),
        content: toContent(r.metadata || {}),
        score: r.score ?? 0,
        metadata: r.metadata || {},
      })
    );

    const elapsed = Date.now() - started;

    // Only show textual part of the answer: extract response='...'
    const extractResponse = (ans: string): string => {
      if (typeof ans !== "string") return String(ans ?? "");
      // Prefer strict pattern: response="..." followed by space and thinking=None
      const strict = ans.match(
        /response=[" || '](.*?)[" || ']\s+thinking=None/
      );
      if (strict && strict[1]) return strict[1].trim();
      // Fallback: any response="..."
      const loose = ans.match(/response="([\s\S]*?)"/);
      if (loose && loose[1]) return loose[1].trim();
      return ans.trim();
    };

    const cleanAnswer = extractResponse(backend.answer);
    const decodeEscapes = (s: string): string => {
      try {
        // Wrap in quotes and escape only quotes to allow JSON to decode escapes like \n, \t, \uXXXX
        const jsonLiteral = `"${s.replace(/"/g, '\\"')}"`;
        return JSON.parse(jsonLiteral);
      } catch {
        // Fallback: minimal common escapes
        return s
          .replace(/\\n/g, "\n")
          .replace(/\\r/g, "\r")
          .replace(/\\t/g, "\t")
          .replace(/\\"/g, '"')
          .replace(/\\'/g, "'")
          .replace(/\\\\/g, "\\");
      }
    };
    const displayAnswer = decodeEscapes(cleanAnswer).trim();

    // Adapt to existing front-end QueryResponse shape used by components
    return {
      answer: displayAnswer,
      retrieved_documents: retrieved,
      metadata: {
        pipeline_used: backend.rag ?? "unknown",
        query_time: elapsed,
        total_documents: retrieved.length,
      },
    };
  }

  async evaluate(data: EvaluateRequest): Promise<EvaluateResponse> {
    return this.request("/api/v1/evaluate", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // Backend: POST /open-path (helper for browser environments)
  async openPath(payload: {
    path: string;
    reveal?: boolean;
  }): Promise<{ status: string }> {
    return this.request("/open-path", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }
  // NOTE: getSettings / updateSettings methods were removed as they were unused.
  // Reintroduce if backend settings endpoints are implemented.
}

// Create and export a singleton instance
export const apiClient = new ApiClient();
export default ApiClient;
