/**
 * Mapping between frontend selections and backend RAG config files.
 *
 * Resolves which backend config filename to send when the user
 * selects a pipeline type and a knowledge base.
 */

import type { PipelineType } from "../types/retriever";

export type KnowledgeBaseId = string; // e.g. "vidore/infovqa_test_subsampled_beir", "beir/nfcorpus"

/**
 * map: (pipeline, knowledge base) -> backend config filename
 */
export const RAG_CONFIG_MAP: Record<
  PipelineType,
  Partial<Record<KnowledgeBaseId, string>>
> = {
  traditional: {
    // InfoVQA (Vidore)
    "vidore/infovqa_test_subsampled_beir": "traditional_infovqa.json",
    // NF Corpus (BEIR)
    "beir/nfcorpus": "traditional_nfcorpus.json",
  },
  multimodal: {
    // InfoVQA (Vidore)
    "vidore/infovqa_test_subsampled_beir": "multimodal_infovqa.json",
    // NF Corpus (BEIR)
    "beir/nfcorpus": "multimodal_nfcorpus.json",
  },
  multi: {
    // InfoVQA (Vidore)
    "vidore/infovqa_test_subsampled_beir": "multi_infovqa.json",
    // NF Corpus (BEIR)
    "beir/nfcorpus": "multi_nfcorpus.json",
  },
};

/**
 * Resolve a backend config filename from the tuple (pipeline, knowledgeBase).
 * Returns null if there's no mapping so the caller can handle it (e.g. show
 * a message to the user or pick a different KB).
 */
export function resolveConfigFile(
  pipeline: PipelineType,
  knowledgeBase: KnowledgeBaseId
): string | null {
  return RAG_CONFIG_MAP[pipeline]?.[knowledgeBase] ?? null;
}

/**
 * Optionally allow dynamic registration at runtime (e.g. after indexing a new KB).
 */
export function registerConfigMapping(
  pipeline: PipelineType,
  knowledgeBase: KnowledgeBaseId,
  configFile: string
): void {
  RAG_CONFIG_MAP[pipeline] = {
    ...(RAG_CONFIG_MAP[pipeline] || {}),
    [knowledgeBase]: configFile,
  };
}

/**
 * Get a default config for a pipeline (first available mapping), or null.
 */
export function getDefaultConfigForPipeline(
  pipeline: PipelineType
): string | null {
  const map = RAG_CONFIG_MAP[pipeline] || {};
  const first = Object.values(map).find(Boolean);
  return first ?? null;
}
