"""
FastAPI application exposing RAG operations.

Endpoints:
  POST /rag/select     -> Load & set active RAG (unloads previous)
  POST /index          -> Extract & index documents for a (config, knowledge_base)
  POST /retrieve       -> Retrieve top-k results from active RAG
  POST /answer         -> Retrieve + generate answer from active RAG
  GET  /health         -> Liveness & current state

Design notes:
  * Only one RAG instance kept in memory at a time to preserve GPU/CPU memory.
  * Loading a new RAG disposes of the previous (dereference + memory cleanup).
  * Dataset preparation (extraction + indexing) can optionally set the new RAG active.
  * All heavy RAG methods are async already; endpoints await them directly.
  * Documents are passed as server-accessible file paths (future: multipart upload).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from .pipeline.rags.multimodal import MultiModalRAG
from .pipeline.rags.multirag import MultiRAG
from .pipeline.rags.traditional import TraditionalRAG
from .utils.device import cleanup_memory

logger = logging.getLogger(__name__)


CONFIG_DIR = Path(__file__).parent / "configs"
DATA_DIR = Path(__file__).parent / "data" / "rags"


class SelectRAGRequest(BaseModel):
    config: str = Field(
        ...,
        description="Config filename located in configs directory, e.g. traditional_arxiv.json",
    )
    knowledge_base: str = Field(
        ...,
        description="Knowledge base / index name (logical dataset identifier)",
    )


class IndexRequest(SelectRAGRequest):
    documents: list[str] | None = Field(
        None,
        description="Optional list of individual document file paths accessible to the server",
    )
    folder_path: str | None = Field(
        None,
        description="Optional folder path whose contents will be indexed recursively",
    )
    set_current: bool = Field(
        True,
        description="Whether to set this (config, kb) as the active RAG after indexing",
    )


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = Field(
        10,
        ge=1,
        le=100,
        description="Number of results (default 10 as per API spec unless overridden)",
    )


class RetrieveResult(BaseModel):
    metadata: dict[str, Any]
    score: float


class RetrieveResponse(BaseModel):
    rag: str
    knowledge_base: str
    top_k: int
    results: list[RetrieveResult]


class AnswerResponse(RetrieveResponse):
    answer: str


class RAGManager:
    """Manage a single in-memory RAG instance with dynamic loading/unloading."""

    def __init__(self):
        self._rag: Any | None = None
        self._config_name: str | None = None
        self._knowledge_base: str | None = None
        self._lock = asyncio.Lock()

    @property
    def active(self) -> Any | None:  # explicit return type for clarity
        return self._rag

    @property
    def state(self) -> dict:
        return {
            "config": self._config_name,
            "knowledge_base": self._knowledge_base,
            "loaded": self._rag is not None,
        }

    async def load(self, config_name: str, knowledge_base: str) -> None:
        async with self._lock:
            # Short-circuit if already loaded with same params
            if (
                self._rag is not None
                and self._config_name == config_name
                and self._knowledge_base == knowledge_base
            ):
                return

            # Unload previous
            if self._rag is not None:
                logger.info("Unloading existing RAG: %s", self._config_name)
                self._rag = None
                cleanup_memory()

            config_path = CONFIG_DIR / config_name
            if not config_path.exists():
                msg = f"Config file not found: {config_name}"
                raise FileNotFoundError(msg)

            try:
                import json

                with config_path.open() as f:
                    configs = json.load(f)
            except Exception as exc:  # pragma: no cover - defensive
                msg = f"Failed loading config {config_name}: {exc}"
                raise RuntimeError(msg) from exc

            # Override knowledge base so data directory segregates per dataset
            configs["knowledge_base"] = knowledge_base

            print("Setting up system with configs:")
            print(configs)
            print()

            rag_type = self._infer_rag_type(config_name, configs)

            logger.info(
                "Loading RAG (type=%s, config=%s, kb=%s)",
                rag_type,
                config_name,
                knowledge_base,
            )

            if rag_type == "traditional":
                self._rag = TraditionalRAG(
                    name=configs["name"],
                    data_dir=DATA_DIR,
                    configs=configs["configs"],
                )
            elif rag_type == "multimodal":
                self._rag = MultiModalRAG(
                    name=configs["name"],
                    data_dir=DATA_DIR,
                    configs=configs["configs"],
                )
            elif rag_type == "multi":
                self._rag = MultiRAG(
                    name=configs["name"],
                    data_dir=DATA_DIR,
                    configs=configs["configs"],
                )
            else:  # fallback
                msg = f"Unsupported rag type: {rag_type}"
                raise ValueError(msg)

            self._config_name = config_name
            self._knowledge_base = knowledge_base
            cleanup_memory()

    def _infer_rag_type(self, config_name: str, configs: dict) -> str:
        # Priority: explicit field in config; else filename prefix
        if "type" in configs:
            return str(configs["type"]).lower()
        if config_name.startswith("traditional_"):
            return "traditional"
        if config_name.startswith("multimodal_"):
            return "multimodal"
        # Heuristic fallback
        return "traditional"

    async def ensure_active(self):
        if self._rag is None:
            msg = "No active RAG loaded"
            raise RuntimeError(msg)
        return self._rag


rag_manager = RAGManager()

# ---------------- Index task tracking (for async indexing progress) ---------------- #
_index_tasks: dict[str, dict[str, Any]] = {}
_index_task_listeners: dict[str, set[asyncio.Queue]] = {}
_index_tasks_lock = asyncio.Lock()
TASK_RETENTION_SECONDS = 3600  # 1 hour retention for finished tasks


def _now_iso() -> str:
    # Using explicit UTC to unify timestamp format
    return datetime.now(UTC).isoformat()


app = FastAPI(title="RAG Backend Service", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok", "state": rag_manager.state}


@app.post("/rag/select")
async def select_rag(req: SelectRAGRequest):
    try:
        await rag_manager.load(req.config, req.knowledge_base)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    else:
        return {"message": "RAG loaded", **rag_manager.state}


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"}


def _collect_documents(
    folder_path: str | None,
    documents: list[str] | None,
) -> list[Path]:
    collected: list[Path] = []
    if folder_path:
        base = Path(folder_path)
        if not base.exists() or not base.is_dir():
            msg = f"Folder not found or not a directory: {folder_path}"
            raise FileNotFoundError(msg)
        for file in base.rglob("*"):
            if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES:
                collected.append(file)
    if documents:
        for d in documents:
            p = Path(d)
            if p.exists() and p.is_file():
                collected.append(p)
            else:
                logger.warning("Document not found or not a file, skipping: %s", d)
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in collected:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


async def _run_index(
    config: str,
    knowledge_base: str,
    documents: list[str] | None,
    folder_path: str | None,
    task_id: str | None = None,
):
    async def _update(status: str, **extra):
        if task_id is None:
            return
        async with _index_tasks_lock:
            task = _index_tasks.get(task_id)
            if task:
                task.update(status=status, **extra)
                snapshot = dict(task)
                for q in _index_task_listeners.get(task_id, set()):
                    if not q.full():
                        q.put_nowait(snapshot)

    await _update("running", started_at=_now_iso())
    try:
        await rag_manager.load(config, knowledge_base)
        rag = await rag_manager.ensure_active()
        paths = _collect_documents(folder_path, documents)
        if not paths:
            raise RuntimeError("No valid documents to process")
        await rag.extract(paths)
        await rag.index()
        cleanup_memory()
        await _update(
            "done",
            finished_at=_now_iso(),
            documents_indexed=len(paths),
        )
    except Exception as exc:  # pragma: no cover
        await _update(
            "error",
            finished_at=_now_iso(),
            error=str(exc),
        )
        raise


def _clone_config_with_new_kb(original_config: str, new_kb: str) -> str:
    """Clone existing config adjusting knowledge base & name; reuse if exists."""
    src_path = CONFIG_DIR / original_config
    if not src_path.exists():  # pragma: no cover
        msg = f"Base config not found: {original_config}"
        raise FileNotFoundError(msg)
    try:
        data = json.loads(src_path.read_text())
    except Exception as exc:  # pragma: no cover
        msg = f"Failed reading base config {original_config}: {exc}"
        raise RuntimeError(msg) from exc

    # Update kg
    if isinstance(data.get("configs"), dict):
        data["configs"]["knowledge_base"] = new_kb

    base_name = str(data.get("name", Path(original_config).stem))
    if not base_name.endswith(f"_{new_kb}"):
        data["name"] = f"{base_name}_{new_kb}"
    new_filename = f"{Path(original_config).stem}_{new_kb}.json"
    dest_path = CONFIG_DIR / new_filename
    if not dest_path.exists():
        dest_path.write_text(json.dumps(data, indent=2))
        logger.info("Created cloned config: %s", new_filename)
    return new_filename


def _resolve_effective_index_params(req: IndexRequest) -> tuple[str, str, str]:
    """Return (config_filename, knowledge_base, folder_path) after staging/cloning."""
    if not (req.documents or req.folder_path):
        raise ValueError("No documents or folder path provided")

    if req.folder_path:
        return _resolve_from_folder(req)
    return _resolve_from_documents(req)


def _resolve_from_folder(req: IndexRequest) -> tuple[str, str, str]:
    effective_kb = req.knowledge_base
    folder = Path(req.folder_path).resolve()  # type: ignore[arg-type]
    folder_name = folder.name
    parent_name = folder.parent.name
    base_kb = req.knowledge_base.split("/")
    if len(base_kb) > 1 and parent_name == base_kb[1]:
        effective_kb = req.knowledge_base
    elif folder_name != req.knowledge_base:
        effective_kb = folder_name

    print(f"Knowledge base: {req.knowledge_base}")
    print(f"Folder name: {folder_name}")
    print(f"Parent name: {parent_name}")
    print(f"Effective knowledge base: {effective_kb}")

    effective_config = (
        _clone_config_with_new_kb(req.config, effective_kb)
        if effective_kb != req.knowledge_base
        else req.config
    )
    return effective_config, effective_kb, req.folder_path or ""


def _resolve_from_documents(req: IndexRequest) -> tuple[str, str, str]:
    effective_kb = req.knowledge_base
    staging_root = DATA_DIR / req.config["name"] / effective_kb
    staging_root.mkdir(parents=True, exist_ok=True)
    for d in req.documents or []:
        src = Path(d)
        if not (src.exists() and src.is_file()):
            logger.warning("Skipping missing document during staging: %s", d)
            continue
        dest = staging_root / src.name
        if not dest.exists():
            try:
                shutil.copy2(src, dest)
            except OSError as exc:  # pragma: no cover
                logger.warning("Failed copying %s: %s", src, exc)
    effective_config = _clone_config_with_new_kb(req.config, effective_kb)
    return effective_config, effective_kb, str(staging_root)


@app.post("/index")
async def create_index(req: IndexRequest, background: BackgroundTasks):
    if not (req.documents or req.folder_path):
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of documents or folder_path",
        )
    try:
        effective_config, effective_kb, effective_folder = (
            _resolve_effective_index_params(req)
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    task_id = uuid.uuid4().hex
    async with _index_tasks_lock:
        _index_tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "config": effective_config,
            "knowledge_base": effective_kb,
            "folder": effective_folder,
            "documents": req.documents,
            "set_current": req.set_current,
            "created_at": _now_iso(),
        }
        _index_task_listeners.setdefault(task_id, set())

    background.add_task(
        _run_index,
        effective_config,
        effective_kb,
        req.documents,
        effective_folder,
        task_id,
    )
    if req.set_current:
        await rag_manager.load(effective_config, effective_kb)
    try:
        count = len(_collect_documents(effective_folder, req.documents))
    except (FileNotFoundError, RuntimeError):  # pragma: no cover
        count = None
    return {
        "message": "Indexing started",
        "task_id": task_id,
        "status_url": f"/index/status/{task_id}",
        "ws_url": f"/index/ws/{task_id}",
        "active": rag_manager.state,
        "documents_count": count,
        "effective_config": effective_config,
        "effective_knowledge_base": effective_kb,
        "staging_folder": effective_folder,
    }


@app.get("/index/status/{task_id}")
async def index_status(task_id: str):
    async with _index_tasks_lock:
        task = _index_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/index/status")
async def list_index_tasks():
    async with _index_tasks_lock:
        tasks = list(_index_tasks.values())
    return {"tasks": tasks}


@app.websocket("/index/ws/{task_id}")
async def index_status_ws(websocket: WebSocket, task_id: str):
    await websocket.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    async with _index_tasks_lock:
        if task_id not in _index_tasks:
            await websocket.send_json({"error": "Task not found"})
            await websocket.close(code=1008)
            return
        _index_task_listeners.setdefault(task_id, set()).add(q)
        await websocket.send_json(_index_tasks[task_id])  # initial snapshot
    try:
        while True:
            update = await q.get()
            await websocket.send_json(update)
            if update.get("status") in {"done", "error"}:
                break
    except WebSocketDisconnect:  # pragma: no cover
        pass
    finally:
        async with _index_tasks_lock:
            listeners = _index_task_listeners.get(task_id)
            if listeners and q in listeners:
                listeners.remove(q)
            if listeners and len(listeners) == 0:
                _index_task_listeners.pop(task_id, None)


async def _cleanup_index_tasks_loop():  # pragma: no cover
    while True:
        await asyncio.sleep(600)
        cutoff = datetime.now(UTC).timestamp() - TASK_RETENTION_SECONDS
        async with _index_tasks_lock:
            to_remove: list[str] = []
            for tid, data in _index_tasks.items():
                status = data.get("status")
                finished_at = data.get("finished_at")
                if status in {"done", "error"} and finished_at:
                    try:
                        ts = datetime.fromisoformat(str(finished_at)).timestamp()
                        if ts < cutoff:
                            to_remove.append(tid)
                    except ValueError:
                        continue
            for tid in to_remove:
                _index_tasks.pop(tid, None)
                _index_task_listeners.pop(tid, None)


_background_tasks: set[asyncio.Task] = set()


@app.on_event("startup")
async def _startup_index_task_cleanup():  # pragma: no cover
    task = asyncio.create_task(_cleanup_index_tasks_loop())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: QueryRequest):
    try:
        rag = await rag_manager.ensure_active()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    requested_top_k = req.top_k or 10
    try:
        results_nested = await rag.retrieve(req.query, top_k=requested_top_k)
        results = results_nested[0] if results_nested else []
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    model_results = [RetrieveResult(metadata=m, score=float(s)) for m, s in results]

    return RetrieveResponse(
        rag=rag_manager.state["config"] or "unknown",
        knowledge_base=rag_manager.state["knowledge_base"] or "unknown",
        top_k=requested_top_k,
        results=model_results,
    )


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: QueryRequest):
    try:
        rag = await rag_manager.ensure_active()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    requested_top_k = req.top_k or 10
    try:
        answer_text, retrieved = await rag.answer(
            req.query,
            top_k=requested_top_k,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {exc}",
        ) from exc

    model_results = [RetrieveResult(metadata=m, score=float(s)) for m, s in retrieved]

    return AnswerResponse(
        rag=rag_manager.state["config"] or "unknown",
        knowledge_base=rag_manager.state["knowledge_base"] or "unknown",
        top_k=requested_top_k,
        results=model_results,
        answer=answer_text,
    )
