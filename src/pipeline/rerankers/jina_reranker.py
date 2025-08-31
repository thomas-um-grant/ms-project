from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from pipeline.models.embedding_models import JinaV4Model
from utils.device import DeviceConfig

logger = logging.getLogger(__name__)


class JinaReranker:
    """Rerank candidates using existing Jina retrieval embeddings."""

    def __init__(
        self,
        *,
        embeddings_path: Path,
        embeddings_ids_path: Path,
        device_config: DeviceConfig | None = None,
        embedding_model: JinaV4Model | None = None,
    ) -> None:
        self.embeddings_path = embeddings_path
        self.embeddings_ids_path = embeddings_ids_path
        self.device_config = device_config
        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else JinaV4Model(device_config=device_config)
        )

        self._corpus_embeddings: list[torch.Tensor] | None = None
        self._corpus_ids: list[str] | None = None
        self._id_to_idx: dict[str, int] = {}

        logger.info(
            "Initialized JinaReranker using store embeddings (%s / %s)",
            self.embeddings_path.name,
            self.embeddings_ids_path.name,
        )

    # ------------------------------------------------------------------
    # Loading existing embeddings
    # ------------------------------------------------------------------
    def load_store_embeddings(self, *, force: bool = False) -> bool:
        if self._corpus_embeddings is not None and not force:
            return True
        if not (self.embeddings_path.exists() and self.embeddings_ids_path.exists()):
            logger.warning(
                "JinaReranker: retrieval embedding files missing (%s / %s)",
                self.embeddings_path,
                self.embeddings_ids_path,
            )
            return False
        try:
            self._corpus_embeddings = torch.load(
                self.embeddings_path,
                map_location="cpu",
                weights_only=True,
            )
            ids: list[str] = []
            with self.embeddings_ids_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Each line is JSON encoded ID string
                    try:
                        ids.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Fallback: raw line
                        ids.append(line)
            self._corpus_ids = ids
            self._id_to_idx = {cid: i for i, cid in enumerate(ids)}
            logger.info("Loaded %d Jina embeddings for reranking", len(ids))
            return True
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed loading store embeddings for Jina reranker: %s", e)
            self._corpus_embeddings = None
            self._corpus_ids = None
            self._id_to_idx = {}
            return False

    # ------------------------------------------------------------------
    async def _embed_queries_batched(
        self,
        queries: list[str],
        batch_size: int = 8,
    ) -> list[torch.Tensor]:
        if not queries:
            return []
        out: list[torch.Tensor] = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            try:
                if hasattr(self.embedding_model, "embed_queries"):
                    embs = await self.embedding_model.embed_queries(batch)  # type: ignore[attr-defined]
                else:
                    embs = await self.embedding_model.embed_texts(batch)
                out.extend(embs)
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "Query batch embedding failed (%s); attempting fallback",
                    e,
                )
                for q in batch:
                    try:
                        if hasattr(self.embedding_model, "embed_queries"):
                            single = await self.embedding_model.embed_queries([q])  # type: ignore[attr-defined]
                        else:
                            single = await self.embedding_model.embed_texts([q])
                        out.extend(single)
                    except Exception as se:  # pragma: no cover
                        logger.error("Single query embedding failed: %s", se)
                        if self._corpus_embeddings:
                            dim = self._corpus_embeddings[0].shape[0]
                            out.append(torch.zeros(dim))
        return out

    # ------------------------------------------------------------------
    async def rerank(
        self,
        queries: str | list[str],
        retrieved_candidates: list[list[tuple[dict[str, Any], float]]],
        top_k: int | None = None,
    ) -> list[list[tuple[dict[str, Any], float]]]:
        if not self.load_store_embeddings():  # Guard: bypass if embeddings absent
            logger.warning("JinaReranker: embeddings unavailable, bypassing rerank")
            return retrieved_candidates

        query_list = [queries] if isinstance(queries, str) else queries
        if len(query_list) != len(retrieved_candidates):
            raise ValueError(
                f"Queries ({len(query_list)}) and candidate lists ({len(retrieved_candidates)}) length mismatch",
            )

        query_embs = await self._embed_queries_batched(query_list)
        norm_queries = [qe / (qe.norm(p=2) + 1e-9) for qe in query_embs]

        results: list[list[tuple[dict[str, Any], float]]] = []
        for q_emb, candidates in zip(norm_queries, retrieved_candidates, strict=True):
            if not candidates:
                results.append([])
                continue
            scored: list[tuple[dict[str, Any], float]] = []
            for meta, orig_score in candidates:
                corpus_id = meta.get("corpus-id", "")
                doc_id = meta.get("doc-id", "")
                possible_ids = [f"{corpus_id}_{doc_id}"] if doc_id else []
                possible_ids.append(corpus_id)
                idx = -1
                for pid in possible_ids:
                    if pid in self._id_to_idx:
                        idx = self._id_to_idx[pid]
                        break
                if idx >= 0:
                    d = self._corpus_embeddings[idx]
                    d = d / (d.norm(p=2) + 1e-9)
                    sim = float(torch.dot(q_emb, d).item())
                    scored.append((meta, sim))
                else:
                    scored.append((meta, orig_score))
            scored.sort(key=lambda x: x[1], reverse=True)
            if top_k is not None:
                scored = scored[:top_k]
            results.append(scored)
        return results

    # ------------------------------------------------------------------
    def get_embedding_stats(self) -> dict[str, Any]:
        if not self._corpus_embeddings:
            return {"status": "missing"}
        return {
            "status": "ready",
            "num_embeddings": len(self._corpus_embeddings),
            "embedding_dimension": self._corpus_embeddings[0].shape[0],
            "source_embeddings": str(self.embeddings_path),
        }


class JinaRerankerFactory:
    """Factory returning a JinaReranker bound to existing embedding store."""

    @staticmethod
    def create_reranker(
        *,
        embeddings_path: Path,
        embeddings_ids_path: Path,
        device_config: DeviceConfig | None = None,
        embedding_model: JinaV4Model | None = None,
    ) -> JinaReranker:
        return JinaReranker(
            embeddings_path=embeddings_path,
            embeddings_ids_path=embeddings_ids_path,
            device_config=device_config,
            embedding_model=embedding_model,
        )
