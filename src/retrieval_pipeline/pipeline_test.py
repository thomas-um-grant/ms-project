import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path

import pymupdf
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available


def get_torch_device(device: str = "auto") -> str:
    """
    Get the appropriate torch device based on the input string.

    Args:
        device (str): Device type, can be 'auto', 'cuda', 'mps', or 'cpu'.

    Returns:
        str: The selected device type.

    """
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


# --- Corpus class for storing embeddings and metadata ---
class Corpus:
    def __init__(self, index=None):
        # index: list of dicts, each with keys: id, embedding_files (dict), metadata
        self.index = index or []

    def add(self, corpus_id, embedding_type=None, embedding_file=None, metadata=None):
        # If entry exists, update embedding_files; else, create new entry
        matches = [item for item in self.index if item["id"] == corpus_id]
        if len(matches) > 1:
            print(
                f"[Corpus] Warning: Duplicate entry for id '{corpus_id}' detected! Corpus may be inconsistent.",
            )

        entry = matches[0] if matches else None
        if entry is None:
            entry = {
                "id": corpus_id,
                "embedding_files": {},
                "metadata": metadata or {},
            }
            self.index.append(entry)
        if embedding_type and embedding_file:
            entry["embedding_files"][embedding_type] = embedding_file
        if metadata:
            entry["metadata"].update(metadata)

    def save(self, file_path):
        with file_path.open("w") as f:
            json.dump(self.index, f)

    @classmethod
    def load(cls, file_path):
        with file_path.open() as f:
            index = json.load(f)
        return cls(index)

    def get_ids(self):
        return [item["id"] for item in self.index]

    def get_metadata(self):
        return [item["metadata"] for item in self.index]

    def get_embedding_file(self, idx, embedding_type):
        return self.index[idx]["embedding_files"].get(embedding_type)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.index[idx]

    def get_entry_by_id(self, corpus_id):
        return next((item for item in self.index if item["id"] == corpus_id), None)


# --- PreProcessor class for PDF chunking ---
class PreProcessor:
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = chunk_dir
        self.chunk_dir.mkdir(exist_ok=True)

    def process_pdf(
        self,
        pdf_path: Path,
        extract_text: bool = True,
        extract_images: bool = False,
        corpus: Corpus | None = None,
    ) -> list[dict]:
        doc = pymupdf.open(str(pdf_path))
        docname = pdf_path.name
        chunk_entries = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_id = f"{docname}_page{page_num + 1}"
            meta = {
                "corpus_id": pdf_path.name,
                "doc_id": page_num + 1,
            }
            # Extract page as image (render whole page)
            if extract_images:
                img_path = self.chunk_dir / f"{page_id}.png"
                if img_path.exists():
                    # Already extracted
                    entry = {
                        "id": page_id,
                        "embedding_files": {},
                        "metadata": {
                            **meta,
                            "chunk_type": "image",
                            "chunk_file": str(img_path),
                        },
                    }
                else:
                    pix = page.get_pixmap()
                    mode = "RGB" if pix.alpha == 0 else "RGBA"
                    pil_img = Image.frombytes(
                        mode,
                        (pix.width, pix.height),
                        pix.samples,
                    )
                    pil_img.save(str(img_path))
                    entry = {
                        "id": page_id,
                        "embedding_files": {},
                        "metadata": {
                            **meta,
                            "chunk_type": "image",
                            "chunk_file": str(img_path),
                        },
                    }
                if corpus is not None:
                    corpus.add(
                        page_id,
                        embedding_type=None,
                        embedding_file=None,
                        metadata=entry["metadata"],
                    )
                chunk_entries.append(entry)
            # Optionally, extract text as before
            if extract_text:
                text_path = self.chunk_dir / f"{page_id}_text.txt"
                if text_path.exists():
                    entry = {
                        "id": f"{page_id}_text",
                        "embedding_files": {},
                        "metadata": {
                            **meta,
                            "chunk_type": "text",
                            "chunk_file": str(text_path),
                        },
                    }
                else:
                    text = page.get_text()
                    with text_path.open("w") as f:
                        f.write(text)
                    entry = {
                        "id": f"{page_id}_text",
                        "embedding_files": {},
                        "metadata": {
                            **meta,
                            "chunk_type": "text",
                            "chunk_file": str(text_path),
                        },
                    }
                if corpus is not None:
                    corpus.add(
                        entry["id"],
                        embedding_type=None,
                        embedding_file=None,
                        metadata=entry["metadata"],
                    )
                chunk_entries.append(entry)

        print(
            f"Processed {len(doc)} pages for {pdf_path.name}. Created {len(chunk_entries)} chunks.",
        )
        return chunk_entries


# --- Model Abstraction ---
class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed_images(self, images: list):
        """Embed a list of images asynchronously and in parallel."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]):
        """Embed a list of texts asynchronously and in parallel."""


# Local ColQwen2 Model
class ColQwen2Model(BaseEmbeddingModel):
    def __init__(self, model_name, device):
        self.model = (
            ColQwen2.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
                if is_flash_attn_2_available()
                else None,
            )
            .to(device)
            .eval()
        )
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        self.device = device

    async def embed_images(self, images: list[Image.Image]):
        loop = asyncio.get_event_loop()

        def _embed():
            batch = self.processor.process_images(images).to(self.device)
            with torch.no_grad():
                img_emb = self.model(**batch)
            return img_emb

        return await loop.run_in_executor(None, _embed)

    async def embed_texts(self, texts: list[str]):
        loop = asyncio.get_event_loop()

        def _embed():
            batch = self.processor.process_queries(texts).to(self.device)
            with torch.no_grad():
                q_emb = self.model(**batch)
            return q_emb

        return await loop.run_in_executor(None, _embed)


# Nomic Model via Ollama API (text embedding only)
class NomicOllamaModel(BaseEmbeddingModel):
    def __init__(
        self,
        ollama_url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    ):
        self.ollama_url = ollama_url
        self.model_name = model_name

    async def embed_images(self, images: list):
        msg = "NomicOllamaModel does not support image embedding."
        raise NotImplementedError(msg)

    async def embed_texts(self, texts: list[str]):
        import aiohttp

        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model_name,
                "prompt": texts if isinstance(texts, list) else [texts],
            }
            async with session.post(self.ollama_url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
        return torch.tensor(data["embeddings"], dtype=torch.float32)


class Embedder:
    def __init__(
        self,
        model: BaseEmbeddingModel,
        configs=None,
        device=None,
    ):
        device = device or get_torch_device("auto")
        self.model = model
        self.device = device

    async def embed_images(self, images: list):
        return await self.model.embed_images(images)

    async def embed_texts(self, texts: list[str]):
        return await self.model.embed_texts(texts)


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        corpus: Corpus,
        retrieval_strategy: str = "image",
    ):
        self.embedder = embedder
        self.corpus = corpus
        self.retrieval_strategy = retrieval_strategy  # 'image' or 'text'

    def get_entries_and_embeddings(self):
        entries = []
        embeddings = []
        names = []
        device = self.embedder.device if hasattr(self.embedder, "device") else "cpu"
        for entry in self.corpus.index:
            if entry["metadata"].get("chunk_type") == self.retrieval_strategy:
                emb_file = entry["embedding_files"].get(self.retrieval_strategy)
                if emb_file:
                    try:
                        emb = torch.load(emb_file, map_location=device)
                        embeddings.append(emb)
                        entries.append(entry)
                        names.append(entry["id"])
                    except (FileNotFoundError, RuntimeError) as exc:
                        print(f"Failed to load embedding for {entry['id']}: {exc}")
        print(
            f"[Retriever] Found {len(embeddings)} valid '{self.retrieval_strategy}' embeddings for retrieval.",
        )
        return entries, embeddings, names

    async def retrieve(self, query: str, top_k: int = 5):
        # Embed the query (as a batch of one)
        query_emb = await self.embedder.embed_texts(
            [query],
        )  # shape: [1, Q, D] or [Q, D]
        if query_emb.ndim == 2:
            query_emb = query_emb.unsqueeze(0)  # [1, Q, D]
        entries, embeddings, names = self.get_entries_and_embeddings()
        if not embeddings:
            return []
        # Compute MaxSim score for each embedding individually to avoid stacking
        scores = []
        for emb in embeddings:
            # emb: [P, D] or [1, P, D]
            if emb.ndim == 2:
                emb = emb.unsqueeze(0)  # [1, P, D]
            elif emb.ndim == 3 and emb.shape[0] != 1:
                emb = emb[:1]  # ensure [1, P, D]
            score = self.maxsim_score(query_emb, emb)  # [1, 1]
            scores.append(score[0, 0].item())
            del emb, score
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        sorted_results = sorted(
            zip(scores, names, strict=False),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]
        return sorted_results

    @staticmethod
    def maxsim_score(query_emb, corpus_embs):
        """
        Compute MaxSim (ColBERT-like) score between a query embedding and a list of passage/image embeddings.

        Args:
            query_emb: [1, Q, D] (Q = #query tokens)
            corpus_embs: [N, P, D] (N = #images, P = #patches/tokens)
        Returns: [1, N] tensor of scores

        """
        # Normalize
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        corpus_embs = torch.nn.functional.normalize(corpus_embs, dim=-1)
        if query_emb.ndim == 2:
            query_emb = query_emb.unsqueeze(0)  # [1, Q, D]
        if corpus_embs.ndim == 2:
            corpus_embs = corpus_embs.unsqueeze(0)  # [1, P, D]
        if corpus_embs.ndim == 3 and corpus_embs.shape[0] != 1:
            corpus_embs = corpus_embs[:1]  # [1, P, D]
        sim = torch.einsum("iqd,npd->inqp", query_emb, corpus_embs)  # [1, 1, Q, P]
        max_sim, _ = sim.max(dim=-1)  # [1, 1, Q]
        score = max_sim.sum(dim=-1)  # [1, 1]
        return score


async def main(folder: Path):
    # 1. Process a PDF, extract images, and embed them in batches
    model_name = "vidore/colqwen2-v1.0"
    device = get_torch_device()
    embedder = Embedder(model=ColQwen2Model(model_name, device), device=device)

    pdfs_folder = folder / "pdfs_folder"
    chunk_dir = folder / "pdf_chunks"
    emb_index_file = folder / "image_corpus_index.json"
    emb_dir = folder / "image_embeddings"
    emb_dir.mkdir(exist_ok=True)

    print("Processing all PDFs in folder and extracting images...")
    corpus = Corpus()
    preprocessor = PreProcessor(chunk_dir)

    pdf_files = sorted(pdfs_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdfs_folder}")
        return
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        preprocessor.process_pdf(
            pdf_path,
            extract_text=False,
            extract_images=True,
            corpus=corpus,
        )

    # Gather all image chunk files from the corpus (for embedding)
    image_entries = [
        entry
        for entry in corpus.index
        if entry["metadata"].get("chunk_type") == "image"
    ]
    image_paths = [Path(entry["metadata"]["chunk_file"]) for entry in image_entries]
    image_ids = [entry["id"] for entry in image_entries]

    print("Embedding extracted images in batch and saving embeddings...")
    batch_size = 10
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_ids = image_ids[i : i + batch_size]
        batch_images = []
        batch_ids_to_embed = []
        emb_files = []
        for p, chunk_id in zip(batch_paths, batch_ids, strict=False):
            # Extract PDF name from chunk_id (format: <pdfname>.pdf_pageX)
            pdf_name = chunk_id.split("_page")[0]
            pdf_subdir = emb_dir / pdf_name
            pdf_subdir.mkdir(exist_ok=True)
            emb_file = pdf_subdir / f"{chunk_id}.pt"
            if emb_file.exists():
                # Already embedded, skip
                emb_files.append(emb_file)
                continue
            img = Image.open(p)
            batch_images.append(img)
            batch_ids_to_embed.append(chunk_id)
            emb_files.append(emb_file)

        if batch_images:
            image_embs = await embedder.embed_images(batch_images)
            for chunk_id, emb, emb_file in zip(
                batch_ids_to_embed,
                image_embs,
                emb_files,
                strict=False,
            ):
                if not emb_file.exists():
                    torch.save(emb.cpu(), emb_file)
                corpus.add(
                    corpus_id=chunk_id,
                    embedding_type="image",
                    embedding_file=str(emb_file),
                )
            del batch_images, image_embs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            # All embeddings in this batch already exist, just update corpus
            for chunk_id, emb_file in zip(batch_ids, emb_files, strict=False):
                corpus.add(
                    corpus_id=chunk_id,
                    embedding_type="image",
                    embedding_file=str(emb_file),
                )
        print(
            f"Saved corpus index with {len(corpus)} images to {emb_index_file} (up to image {i + batch_size} of {len(image_paths)})",
        )

    # Save the corpus index
    corpus.save(emb_index_file)
    print(f"Final corpus index with metadata saved to {emb_index_file}")

    # Retrieve the top k=5 images for a given query using Retriever
    query = "What are the top macroeconomic concerns identified by investors in BCG's Investor Perspectives Series Pulse Check #21 from October 2022?"
    # Best pages to retrieve are 6, 9, 2 from 1.pdf
    if not emb_index_file.exists():
        msg = f"Corpus index file not found: {emb_index_file}"
        raise FileNotFoundError(msg)
    loaded_corpus = Corpus.load(emb_index_file)

    # Use the Retriever class for scoring
    retriever = Retriever(embedder, loaded_corpus, retrieval_strategy="image")
    top_k = 10
    sorted_results = await retriever.retrieve(query, top_k=top_k)
    print(f"Top {top_k} images for query: '{query}'\n")
    if not sorted_results:
        print("No matching files found for this query.")
    else:
        for rank, (score, img_name) in enumerate(sorted_results, 1):
            print(f"{rank}. Image: {img_name} | Score: {score:.4f}")


if __name__ == "__main__":
    import asyncio
    import sys

    def run_async_main():
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and ("google.colab" in sys.modules or "ipykernel" in sys.modules):
            try:
                task = loop.create_task(
                    main(Path("/content/drive/My Drive/data_test")),
                )
                loop.run_until_complete(task)
            except RuntimeError:
                import nest_asyncio

                nest_asyncio.apply()
                task = loop.create_task(
                    main(Path("/content/drive/My Drive/data_test")),
                )
                loop.run_until_complete(task)
        else:
            asyncio.run(main(Path(__file__).parent))

    run_async_main()
