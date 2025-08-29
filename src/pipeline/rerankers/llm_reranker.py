from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from google import genai  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


class LLMReranker:
    """
    Gemini-based LLM reranker with optional full-document and image inlining.

    Args:
    model_name: str
        Gemini model to use (e.g. "gemini-2.0-flash").
    api_key: str | None
        Explicit API key (falls back to GOOGLE_API_KEY / GEMINI_API_KEY env vars).
    max_docs: int
        Max number of top docs from initial retrieval to send to the LLM.
    temperature: float
        Sampling temperature (0.0 for deterministic ranking).
    request_timeout: float | None
        Per-call timeout (seconds) for the LLM request (async wait wrapper).
    corpus_dir: str | Path | None
        Optional directory containing source documents (filenames match metadata 'name').
    max_chars_per_doc: int
        Max characters of text to include per document when inlining.
    max_retries: int
        Number of retry attempts for Gemini API calls (default 3, 0 disables retries).
    backoff_base: float
        Base (seconds) for exponential backoff (actual wait is uniform[0, base * 2^n]).
    backoff_max: float
        Maximum cap (seconds) applied to the exponential backoff.

    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str | None = None,
        max_docs: int = 20,
        temperature: float = 0.0,
        request_timeout: float | None = 65.0,
        corpus_dir: str | Path | None = None,
        max_chars_per_doc: int = 10000,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 8.0,
    ) -> None:
        self.model_name = model_name
        self.api_key = (
            api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            msg = "LLMReranker requires a Google Gemini API key (set GOOGLE_API_KEY or pass api_key)."
            raise ValueError(msg)
        self.max_docs = max_docs
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.corpus_dir = Path(corpus_dir) if corpus_dir else None
        self.max_chars_per_doc = max_chars_per_doc
        self.max_retries = max(0, max_retries)
        self.backoff_base = max(0.0, backoff_base)
        self.backoff_max = max(backoff_base, backoff_max)

        # Schema for structured output (Gemini response_schema)
        self.response_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "corpus_id": {"type": "string"},
                            "score": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["corpus_id", "score"],
                    },
                },
            },
            "required": ["query", "results"],
        }
        try:
            self.client = genai.Client(api_key=self.api_key)
        except (ValueError, OSError) as exc:  # pragma: no cover - defensive
            msg = f"Failed to initialize Gemini client: {exc}"
            raise RuntimeError(msg) from exc

    async def rerank(
        self,
        queries: str | list[str],
        retrieved_candidates: list[list[tuple[dict[str, Any], float]]],
        top_k: int | None = None,
    ) -> list[list[tuple[dict[str, Any], float]]]:
        """Rerank candidate documents for each query using the LLM."""
        query_list = [queries] if isinstance(queries, str) else queries
        if len(query_list) != len(retrieved_candidates):
            msg = f"Query count ({len(query_list)}) != candidate list count ({len(retrieved_candidates)})"
            raise ValueError(msg)

        all_results: list[list[tuple[dict[str, Any], float]]] = []
        for q, candidates in zip(query_list, retrieved_candidates, strict=True):
            if not candidates:
                all_results.append([])
                continue

            slice_candidates = candidates[: self.max_docs]
            try:
                new_order = await self._rank_single_query(q, slice_candidates)
                merged = self._merge_new_order(
                    new_order,
                    slice_candidates,
                    candidates,
                )
            except (ValueError, RuntimeError):  # fail open
                merged = candidates

            # Always cap results: prefer explicit top_k else max_docs
            cap = top_k if top_k is not None else self.max_docs
            merged = merged[:cap]
            all_results.append(merged)

        return all_results

    def _merge_new_order(
        self,
        new_order: list[tuple[str, float]],
        slice_candidates: list[tuple[dict[str, Any], float]],
    ) -> list[tuple[dict[str, Any], float]]:
        id_to_meta = {
            m.get("corpus-id", m.get("name", "")): (m, s) for m, s in slice_candidates
        }
        ordered: list[tuple[dict[str, Any], float]] = []
        used = set()
        for cid, score in new_order:
            meta_score = id_to_meta.get(cid)
            if meta_score:
                meta, _ = meta_score
                ordered.append((meta, score))
                used.add(cid)
        for meta, score in slice_candidates:
            cid = meta.get("corpus-id", meta.get("name", ""))
            if cid not in used:
                ordered.append((meta, score))
        return ordered

    async def _rank_single_query(
        self,
        query: str,
        candidates: list[tuple[dict[str, Any], float]],
    ) -> list[tuple[str, float]]:
        """Call Gemini and return list of (corpus_id, score)."""
        prompt = self._build_prompt(query, candidates)
        loop = asyncio.get_event_loop()

        def _call_llm() -> str:
            last_exc: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={
                            "temperature": self.temperature,
                            "response_mime_type": "application/json",
                            "response_schema": self.response_schema,
                        },
                    )
                    # Response may contain .text or .candidates; handle generically
                    if hasattr(response, "text") and response.text:
                        return response.text
                    try:
                        # Fallback attempt: join parts
                        return "\n".join(
                            p.text
                            for c in getattr(response, "candidates", [])
                            for p in getattr(c, "content", []).parts
                            if getattr(p, "text", None)
                        )
                    except AttributeError:  # pragma: no cover - unexpected shape
                        return str(response)
                except Exception as exc:  # pragma: no cover - network / API failures
                    last_exc = exc
                    if attempt >= self.max_retries:
                        logger.error(
                            "Gemini generate_content failed after %s attempts: %s",
                            attempt + 1,
                            exc,
                        )
                        raise
                    # Exponential backoff with jitter
                    delay = min(
                        self.backoff_base * (2**attempt),
                        self.backoff_max,
                    )
                    # Full jitter (AWS style)
                    sleep_for = random.uniform(0, delay) if delay > 0 else 0
                    logger.warning(
                        "Gemini call failed (attempt %d/%d): %s. Retrying in %.2fs",
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
            # If loop exits without return (should not), raise last exception
            if last_exc:  # pragma: no cover - defensive
                raise last_exc
            raise RuntimeError(
                "Gemini call failed without exception but no response returned",
            )  # pragma: no cover

        raw = await asyncio.wait_for(
            loop.run_in_executor(None, _call_llm),
            timeout=self.request_timeout,
        )
        return self._parse_llm_output(raw)

    def _build_prompt(
        self,
        query: str,
        candidates: list[tuple[dict[str, Any], float]],
    ) -> str | list[dict[str, Any]]:
        """
        Construct instruction + documents.

        Returns either a structured list of Gemini content parts. (txt and imgs)
        """
        # Full document inclusion (multimodal capable). We build a list of parts.
        parts: list[Any] = [
            {
                "text": "System: Rank ALL provided documents for the query by relevance (0-1). Return ONLY JSON matching schema with every document exactly once.",
            },
            {"text": f"Query: {query}"},
        ]

        for idx, (meta, score) in enumerate(candidates, start=1):
            cid = meta.get("corpus-id") or meta.get("name") or f"doc_{idx}"
            name = meta.get("name", "unknown")
            base_header = (
                f"DOC {idx} | ID: {cid} | NAME: {name} | ORIGINAL_SCORE: {score:.4f}"
            )
            content_added = False
            # Try to read file if corpus_dir given
            if self.corpus_dir and name:
                file_path = self.corpus_dir / name
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    if suffix in {".txt", ".md"}:
                        try:
                            raw = file_path.read_text(encoding="utf-8", errors="ignore")
                            trimmed = raw[: self.max_chars_per_doc]
                            parts.append({"text": base_header + "\n" + trimmed})
                            content_added = True
                        except OSError:
                            pass
                    elif suffix in {".png", ".jpg", ".jpeg"}:
                        try:
                            with file_path.open("rb") as f:
                                b64 = base64.b64encode(f.read()).decode()
                            mime = "image/png" if suffix == ".png" else "image/jpeg"
                            parts.append({"text": base_header})
                            parts.append(
                                {"inline_data": {"mime_type": mime, "data": b64}},
                            )
                            content_added = True
                        except OSError:
                            pass
            if not content_added:
                # Fallback when no file content or image available; do not use metadata description.
                parts.append(
                    {"text": base_header + "\nSNIPPET: (NO_CONTENT_AVAILABLE)"},
                )

        return [
            {
                "role": "user",
                "parts": parts,
            },
        ]

    def _parse_llm_output(self, text: str) -> list[tuple[str, float]]:
        """Parse JSON from model output (schema-enforced where possible)."""
        try:  # Fast path: direct JSON expected
            data = json.loads(text)
        except json.JSONDecodeError:
            json_str = self._extract_json(text)
            data = self._load_json(json_str, text)
        results_field = data.get("results")
        if not isinstance(results_field, list):
            msg = "Parsed JSON missing 'results' list"
            raise TypeError(msg)
        return self._collect_scores(results_field)

    def _extract_json(self, text: str) -> str:
        if not text:
            msg = "Empty LLM response"
            raise ValueError(msg)
        trimmed = text.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            return trimmed
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0)
        msg = "No JSON object found in LLM output"
        raise ValueError(msg)

    def _load_json(self, json_str: str, raw_text: str) -> dict[str, Any]:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:  # pragma: no cover - malformed output
            msg = f"Failed to decode JSON from LLM output: {exc}\nRaw: {raw_text[:500]}"
            raise ValueError(msg) from exc

    def _collect_scores(self, items: list[Any]) -> list[tuple[str, float]]:
        parsed: list[tuple[str, float]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            cid = item.get("corpus_id") or item.get("id")
            score = item.get("score")
            if not cid or not isinstance(score, int | float):
                continue
            score_f = float(max(0.0, min(1.0, score)))
            parsed.append((str(cid), score_f))
        if not parsed:
            msg = "No valid (corpus_id, score) pairs parsed"
            raise ValueError(msg)
        return parsed


class LLMRerankerFactory:
    """Factory for creating LLM rerankers (mirrors Jina factory style)."""

    @staticmethod
    def create_reranker(**kwargs: Any) -> LLMReranker:
        return LLMReranker(**kwargs)


__all__ = ["LLMReranker", "LLMRerankerFactory"]
