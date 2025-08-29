import asyncio
from pathlib import Path

import nest_asyncio
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.hf import hf_embed
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

nest_asyncio.apply()

WORKING_DIR = Path(__file__).parent.parent.parent.parent / "data/rags/graph"

if not WORKING_DIR.exists():
    WORKING_DIR.mkdir(parents=True)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5vl:7b",
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=5000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(
                    "bert-base-uncased",
                ),
                embed_model=AutoModel.from_pretrained(
                    "nomic-ai/nomic-embed-text-v1.5",
                    trust_remote_code=True,
                    safe_serialization=True,
                ),
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    rag = await initialize_rag()

    corpuses = WORKING_DIR / "corpuses"

    for corpus in corpuses.glob("*.txt"):
        with corpus.open(encoding="utf-8") as f:
            await rag.ainsert(f.read())

    query_test = "Quelles étaient les principales composantes du bénéfice net par action d'une entreprise pour les années 2016 et 2017?"
    # Perform naive search
    print(
        await rag.query(
            query_test,
            param=QueryParam(
                mode="naive",
                only_need_context=True,
                top_k=10,
            ),
        ),
    )

    # Perform local search
    print(
        await rag.query(
            query_test,
            param=QueryParam(
                mode="local",
                only_need_context=True,
                top_k=10,
            ),
        ),
    )

    # Perform global search
    print(
        await rag.query(
            query_test,
            param=QueryParam(
                mode="global",
                only_need_context=True,
                top_k=10,
            ),
        ),
    )

    # Perform hybrid search
    print(
        await rag.query(
            query_test,
            param=QueryParam(
                mode="hybrid",
                only_need_context=True,
                top_k=10,
            ),
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
