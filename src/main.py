"""
Application entrypoint for launching FastAPI RAG backend.

Usage (development):
        python -m src.main --host 0.0.0.0 --port 8000

Environment variables to preload a RAG on startup:
        DEFAULT_RAG_CONFIG=<config json filename>
        DEFAULT_RAG_KB=<knowledge base name>

Example:
        DEFAULT_RAG_CONFIG=traditional_arxiv.json DEFAULT_RAG_KB=arxiv python -m src.main

"""

from __future__ import annotations

import argparse
import asyncio
import os

import uvicorn

from src.app import app, rag_manager  # noqa: F401 (app is imported for uvicorn)


async def _preload_if_requested() -> None:
    config = os.getenv("DEFAULT_RAG_CONFIG")
    kb = os.getenv("DEFAULT_RAG_KB")
    if config and kb:
        try:
            await rag_manager.load(config, kb)
        except Exception as exc:  # pragma: no cover
            print(f"[startup] Failed to preload RAG: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG backend server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    return parser.parse_args()


def main():
    args = parse_args()
    # Preload if env vars present
    asyncio.run(_preload_if_requested())

    uvicorn.run(
        "src.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
