from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from google import genai  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases for readability
# ---------------------------------------------------------------------------
CandidateMeta = dict[str, Any]
Candidate = tuple[CandidateMeta, float]
LLMOrder = list[tuple[str, float]]  # (corpus_id, score)


@dataclass(slots=True)
class _RetrySpec:
    max_retries: int
    delay_seconds: float

    def iter_attempts(self) -> Iterable[int]:  # yields attempt indices (0-based)
        return range(self.max_retries + 1)


class LLMReranker:
    """
    Gemini-based LLM reranker with optional full-document and image inlining.

    Args:
        model_name: Gemini model to use (e.g. "gemini-2.0-flash").
        api_key: Explicit API key (falls back to GOOGLE_API_KEY / GEMINI_API_KEY).
        temperature: Sampling temperature (0.0 for deterministic ranking).
        request_timeout: Per-call timeout (seconds) for the LLM request.
        corpus_dir: Optional directory containing source documents (filenames match metadata 'name').
        max_chars_per_doc: Max characters of text to include per document when inlining.
        max_retries: Number of retry attempts for Gemini API calls (0 disables retries).
        retry_delay_seconds: Fixed delay (seconds) between retry attempts.
        content_mode: 'full' to inline file contents/images, 'description' to use metadata descriptions only.
        full_mode_limit: Hard cap on docs passed to LLM in full mode.
        description_mode_limit: Hard cap on docs passed in description mode.
        concurrent_requests: Max in-flight rerank calls (async concurrency semaphore).
        max_total_prompt_chars: Guard rail limiting total textual prompt size (chars).
        prompt_truncation_notice: Append a sentinel marker if truncation occurred.

    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        request_timeout: float | None = 65.0,
        corpus_dir: str | Path | None = None,
        max_chars_per_doc: int = 100000,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        content_mode: Literal["full", "description"] = "full",
        # When True, large text documents will be uploaded once to Gemini's File API
        # and referenced via file_data parts instead of fully inlined, saving prompt tokens.
        use_file_references: bool = False,
        file_reference_min_chars: int = 5000,
        full_mode_limit: int = 20,
        description_mode_limit: int = 100,
        concurrent_requests: int = 4,
        max_total_prompt_chars: int = 100000,
        prompt_truncation_notice: bool = True,
    ) -> None:
        # Core config ---------------------------------------------------
        self.model_name = model_name
        self.api_key = (
            api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            msg = "LLMReranker requires a Google Gemini API key (set GOOGLE_API_KEY or pass api_key)."
            raise ValueError(msg)
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.corpus_dir = Path(corpus_dir) if corpus_dir else None
        self.max_chars_per_doc = max_chars_per_doc

        # Retry config as a dataclass for clarity
        self.retry = _RetrySpec(max(0, max_retries), max(0.0, retry_delay_seconds))

        if content_mode not in {"full", "description"}:
            msg = "content_mode must be 'full' or 'description'"
            raise ValueError(msg)
        self.content_mode = content_mode
        self.use_file_references = bool(use_file_references and self.corpus_dir)
        self.file_reference_min_chars = max(1000, file_reference_min_chars)
        self.full_mode_limit = max(1, full_mode_limit)
        self.description_mode_limit = max(1, description_mode_limit)
        self._max_description_chars = 5000  # safety clamp
        self.concurrent_requests = max(1, concurrent_requests)
        self.max_total_prompt_chars = max(2000, max_total_prompt_chars)
        self.prompt_truncation_notice = prompt_truncation_notice

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

        # Simple in-memory cache for file contents (text & encoded images)
        self._doc_cache: dict[str, Any] = {}
        # Cache for uploaded Gemini file references (path -> {file_uri, mime_type})
        self._file_ref_cache: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # ID / key helpers
    # ---------------------------------------------------------------------
    def _resolve_doc_id(self, meta: dict[str, Any], fallback_idx: int) -> str:
        """
        Return a stable identifier for a candidate document.

        We accept multiple possible incoming keys (corpus-id, corpus_id, id, name).
        If none exist, we create a synthetic one (doc_<n>_<shortuuid>) to keep
        consistency. The value returned is what we instruct the LLM to echo back
        verbatim under corpus_id so merging succeeds.
        """
        cid = (
            meta.get("corpus-id")
            or meta.get("corpus_id")
            or meta.get("id")
            or meta.get("name")
        )
        if isinstance(cid, str) and cid.strip():
            return cid.strip()
        # create synthetic but reproducible within this run
        synthetic = meta.setdefault(
            "_synthetic_id",
            f"doc_{fallback_idx}_{uuid4().hex[:8]}",
        )
        return synthetic

    @staticmethod
    def _normalize_id(value: str) -> str:
        return value.strip().lower()

    async def rerank(
        self,
        queries: str | list[str],
        retrieved_candidates: list[list[Candidate]],
        top_k: int | None = None,
    ) -> list[list[Candidate]]:
        """Rerank candidate documents for each query using the LLM."""
        query_list = [queries] if isinstance(queries, str) else queries
        if len(query_list) != len(retrieved_candidates):
            msg = f"Query count ({len(query_list)}) != candidate list count ({len(retrieved_candidates)})"
            raise ValueError(msg)

        # Concurrency: schedule tasks with bounded semaphore
        semaphore = asyncio.Semaphore(self.concurrent_requests)

        async def _process(q: str, candidates: list[Candidate]):
            if not candidates:
                return []
            hard_cap = self._mode_limit
            slice_candidates = candidates[:hard_cap]
            try:
                async with semaphore:
                    new_order = await self._rank_single_query(q, slice_candidates)
                merged = self._merge_new_order(new_order, slice_candidates)
            except Exception:  # fail open on any exception (broader than before)
                logger.exception("LLM rerank failed; returning original order")
                merged = slice_candidates
            # Apply top_k if provided (after rerank but before hard cap shrink)
            limit = min(hard_cap, top_k) if top_k else hard_cap
            return merged[:limit]

        tasks = [
            asyncio.create_task(_process(q, c))
            for q, c in zip(query_list, retrieved_candidates, strict=True)
        ]
        return await asyncio.gather(*tasks)

    def _merge_new_order(
        self,
        new_order: LLMOrder,
        slice_candidates: list[Candidate],
    ) -> list[Candidate]:
        """
        Reconstruct ordered candidate list using model scores.

        Any IDs not returned by the LLM (should not happen if it followed instructions)
        are appended in original order with their original retrieval scores as a fallback.
        """
        id_to_meta: dict[str, Candidate] = {}
        for idx, (meta, orig_score) in enumerate(slice_candidates, start=1):
            rid = self._resolve_doc_id(meta, idx)
            id_to_meta[self._normalize_id(rid)] = (meta, orig_score)

        ordered: list[Candidate] = []
        used: set[str] = set()
        for cid, score in new_order:
            norm = self._normalize_id(str(cid))
            meta = id_to_meta.get(norm)
            if meta:
                ordered.append((meta[0], score))
                used.add(norm)

        # Append items the model missed to preserve completeness
        for meta, score in slice_candidates:
            norm = self._normalize_id(self._resolve_doc_id(meta, 0))
            if norm not in used:
                ordered.append((meta, score))
        return ordered

    async def _rank_single_query(
        self,
        query: str,
        candidates: list[Candidate],
    ) -> LLMOrder:
        """Build prompt, issue Gemini call (with retries), parse result."""
        prompt = self._build_prompt(query, candidates)
        loop = asyncio.get_running_loop()
        raw = await asyncio.wait_for(
            loop.run_in_executor(None, self._call_gemini, prompt),
            timeout=self.request_timeout,
        )
        return self._parse_llm_output(raw)

    # ------------------------------------------------------------------
    # Gemini API invocation (sync portion executed in threadpool)
    # ------------------------------------------------------------------
    def _call_gemini(
        self,
        prompt: list[dict[str, Any]],
    ) -> str:  # pragma: no cover - mostly I/O
        last_exc: Exception | None = None
        for attempt in self.retry.iter_attempts():
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
                # Preferred direct attribute
                if getattr(response, "text", None):
                    return response.text  # type: ignore[return-value]
                # Fallback: stitch candidate parts
                try:
                    joined = "\n".join(
                        p.text
                        for c in getattr(response, "candidates", [])
                        for p in getattr(c, "content", []).parts
                        if getattr(p, "text", None)
                    )
                    if joined:
                        return joined
                except AttributeError:
                    pass
                return str(response)
            except Exception as exc:
                last_exc = exc
                if attempt >= self.retry.max_retries:
                    logger.exception(
                        "Gemini generate_content failed after %s attempts",
                        attempt + 1,
                    )
                    raise
                if self.retry.delay_seconds > 0:
                    logger.warning(
                        "Gemini call failed (attempt %d/%d): %s. Retrying in %.2fs",
                        attempt + 1,
                        self.retry.max_retries + 1,
                        exc,
                        self.retry.delay_seconds,
                    )
                    time.sleep(self.retry.delay_seconds)
        if last_exc:
            raise last_exc
        msg = "Gemini call failed unexpectedly without response"
        raise RuntimeError(msg)

    def _build_prompt(
        self,
        query: str,
        candidates: list[Candidate],
    ) -> list[dict[str, Any]]:
        """Assemble Gemini "contents" payload with size guard rails."""
        system_instruction = (
            "You are an expert reranker. Given a query and a list of documents, output ONLY valid JSON matching the schema. "
            "Schema: { query: string, results: [ { corpus_id: string, score: number (0-1), reason?: string } ] }. "
            "Return each input document exactly once (no omissions, no extras). Use provided document IDs verbatim (case-sensitive). "
            "Rank by descending relevance (index 0 = most relevant). Do not invent IDs or wrap JSON in prose."
        )
        parts: list[Any] = [
            {"text": f"System: {system_instruction}"},
            {"text": f"Query: {query}"},
            *self._build_doc_parts(candidates),
        ]
        # Budget enforcement
        total_chars = 0
        truncated = False
        filtered: list[Any] = []
        for item in parts:
            if "text" in item:
                text_val: str = item["text"]
                length = len(text_val)
                if total_chars + length > self.max_total_prompt_chars:
                    remaining = self.max_total_prompt_chars - total_chars
                    if remaining > 0:
                        filtered.append({"text": text_val[:remaining]})
                        total_chars += remaining
                    truncated = True
                    break
                filtered.append(item)
                total_chars += length
            else:
                filtered.append(item)
        if truncated and self.prompt_truncation_notice:
            filtered.append({"text": "[TRUNCATED_PROMPT_CONTENT_DUE_TO_SIZE_LIMIT]"})
        return [{"role": "user", "parts": filtered}]

    # --- helpers -----------------------------------------------------------------
    def _build_doc_parts(self, candidates: list[Candidate]) -> list[Any]:
        if self.content_mode == "full":
            return self._build_full_mode_parts(candidates)
        return self._build_description_mode_parts(candidates)

    # --- mode specific -------------------------------------------------
    def _build_full_mode_parts(self, candidates: list[Candidate]) -> list[Any]:
        output: list[Any] = []
        for idx, (meta, _) in enumerate(candidates, start=1):
            cid = self._resolve_doc_id(meta, idx)
            name = meta.get("name", "unknown")
            header = f"### DOC_START id={cid}\nNAME: {name}"
            added = False
            if self.corpus_dir and name:
                file_path = self.corpus_dir / name
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    if suffix in {".txt", ".md"}:
                        text_content = self._get_cached_text(file_path)
                        if text_content is not None:
                            # Decide whether to upload & reference instead of inline
                            if (
                                self.use_file_references
                                and len(text_content) >= self.file_reference_min_chars
                            ):
                                ref = self._get_or_upload_file(
                                    file_path,
                                    "text/markdown"
                                    if suffix == ".md"
                                    else "text/plain",
                                )
                                if ref:
                                    output.append({"text": header})
                                    output.append({"file_data": ref})
                                    output.append({"text": "### DOC_END"})
                                    added = True
                            if not added:  # fallback or small file
                                output.append(
                                    {"text": f"{header}\n{text_content}\n### DOC_END"},
                                )
                                added = True
                    elif suffix in {".png", ".jpg", ".jpeg"}:
                        img = self._get_cached_image(file_path)
                        if img is not None:
                            output.append({"text": header + "\n(IMAGE)"})
                            output.append({"inline_data": img})
                            output.append({"text": "### DOC_END"})
                            added = True
            if not added:
                output.append(
                    {
                        "text": f"{header}\nCONTENT: (NOT_AVAILABLE_OR_MODE_DESCRIPTION)\n### DOC_END",
                    },
                )
        return output

    def _build_description_mode_parts(self, candidates: list[Candidate]) -> list[Any]:
        output: list[Any] = []
        for idx, (meta, _) in enumerate(candidates, start=1):
            cid = self._resolve_doc_id(meta, idx)
            name = meta.get("name", "unknown")
            desc = str(meta.get("description", "")).strip() or "(NO_DESCRIPTION)"
            if len(desc) > self._max_description_chars:
                desc = desc[: self._max_description_chars] + "…"
            output.append(
                {
                    "text": f"### DOC_START id={cid}\nNAME: {name}\nDESCRIPTION: {desc}\n### DOC_END",
                },
            )
        return output

    # --- caching helpers ------------------------------------------------
    def _get_cached_text(self, file_path: Path) -> str | None:
        key = str(file_path)
        cached = self._doc_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        try:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover
            return None
        text_content = raw[: self.max_chars_per_doc]
        self._doc_cache[key] = text_content
        return text_content

    def _get_cached_image(self, file_path: Path) -> dict[str, str] | None:
        key = f"{file_path}::image"
        cached = self._doc_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        try:
            with file_path.open("rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        except OSError:  # pragma: no cover
            return None
        mime = "image/png" if file_path.suffix.lower() == ".png" else "image/jpeg"
        data = {"mime_type": mime, "data": b64}
        self._doc_cache[key] = data
        return data

    # --- file reference helpers --------------------------------------
    def _get_or_upload_file(
        self,
        file_path: Path,
        mime_type: str,
    ) -> dict[str, str] | None:  # pragma: no cover - network I/O
        """
        Upload file once to Gemini File API and cache reference.

        Returns a dict suitable for a Gemini part: {"file_uri": ..., "mime_type": ...}.
        Falls back to None on failure so caller can inline.
        """
        key = str(file_path)
        existing = self._file_ref_cache.get(key)
        if existing:
            return existing
        try:
            uploaded = self.client.files.upload(
                file=str(file_path),
                mime_type=mime_type,
            )
            file_obj = getattr(uploaded, "file", uploaded)
            file_uri = getattr(file_obj, "uri", None) or getattr(
                file_obj,
                "file_uri",
                None,
            )
            if not file_uri:
                return None
            ref = {"file_uri": file_uri, "mime_type": mime_type}
            self._file_ref_cache[key] = ref
            return ref
        except Exception:
            logger.exception("Failed to upload file for reference: %s", file_path)
            return None

    # --- derived properties --------------------------------------------
    @property
    def _mode_limit(self) -> int:
        return (
            self.full_mode_limit
            if self.content_mode == "full"
            else self.description_mode_limit
        )

    def _parse_llm_output(self, text: str) -> LLMOrder:
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

    def _collect_scores(self, items: list[Any]) -> LLMOrder:
        parsed: LLMOrder = []
        for item in items:
            if not isinstance(item, dict):
                continue
            cid = item.get("corpus_id") or item.get("corpus-id") or item.get("id")
            score = item.get("score")
            if cid is None or not isinstance(score, (int, float)):
                continue
            score_f = float(max(0.0, min(1.0, float(score))))
            parsed.append((str(cid).strip(), score_f))
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
