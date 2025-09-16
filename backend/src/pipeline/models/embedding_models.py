"""Embedding model implementations (text + multimodal) used across RAG pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, ClassVar

import aiohttp
import torch
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from src.pipeline.models.base_embedder import BaseEmbeddingModel
from src.utils.device import DeviceConfig

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""

    def __init__(self, operation: str, details: str):
        self.operation = operation
        self.details = details
        super().__init__(f"{operation} failed: {details}")


# ---------------------------------------------------------------------------
# Nomic (Ollama) text-only embedding model
# ---------------------------------------------------------------------------
class NomicOllamaModel(BaseEmbeddingModel):
    """Nomic embedding model via local Ollama API (text only)."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/embeddings",
        model_name: str = "nomic-embed-text",
        device_config: DeviceConfig | None = None,
    ) -> None:
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.device_config = device_config or DeviceConfig.auto_detect()
        self._load_model()
        logger.info(
            "Initialized NomicOllamaModel with %s at %s",
            model_name,
            ollama_url,
        )

    def _load_model(self) -> None:
        return  # API-based model

    async def embed_images(
        self,
        images: list[Image.Image],
        dtype: torch.dtype | None,
    ) -> list[torch.Tensor]:
        msg = "NomicOllamaModel does not support image embedding."
        raise NotImplementedError(msg)

    async def embed_texts(self, texts: list[str]) -> list[torch.Tensor]:
        if not texts:
            return []

        async def _embed_one(session: aiohttp.ClientSession, text: str, idx: int):
            payload = {"model": self.model_name, "prompt": text}
            async with session.post(self.ollama_url, json=payload) as resp:
                body = await resp.text()
                if resp.status != 200:
                    logger.error(
                        "Ollama embedding failed (status=%s idx=%d): %.160s",
                        resp.status,
                        idx,
                        body,
                    )
                    raise EmbeddingError(
                        "Ollama text embedding",
                        f"status={resp.status} body={body[:120]}",
                    )
                data = await resp.json()
                emb_raw = data.get("embedding")
                if emb_raw is None:
                    raise EmbeddingError(
                        "Ollama text embedding",
                        "Missing 'embedding' in response",
                    )
                return idx, torch.tensor(emb_raw, dtype=torch.float32)

        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(_embed_one(session, t, i))
                for i, t in enumerate(texts)
            ]
            gathered = await asyncio.gather(*tasks, return_exceptions=True)

        first_error = next((g for g in gathered if isinstance(g, Exception)), None)
        if first_error:
            raise first_error
        pairs = [g for g in gathered if isinstance(g, tuple) and len(g) == 2]
        ordered = sorted(pairs, key=lambda x: x[0])
        return [emb for _i, emb in ordered]


# ---------------------------------------------------------------------------
# Nomic HF (nomic-ai/nomic-embed-text-v1) text-only embedding model
# ---------------------------------------------------------------------------
class NomicHFModel(BaseEmbeddingModel):
    """
    Nomic Embed v1 model loaded from Hugging Face via transformers (text only).

    Falls back gracefully across pooling strategies. Normalizes and returns CPU float32 tensors.
    """

    _model_cache: ClassVar[dict[str, tuple[Any, Any]]] = {}

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1",
        *,
        device_config: DeviceConfig | None = None,
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device_config = device_config or DeviceConfig.auto_detect()
        self.batch_size = max(1, batch_size or 64)
        self.normalize = normalize
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._load_model()
        logger.info(
            "Initialized NomicHFModel (%s) on %s (dtype=%s)",
            self.model_name,
            self.device_config.device_str,
            self.device_config.dtype,
        )

    # Properties for parity with other models
    @property
    def device(self) -> str:  # pragma: no cover - trivial
        return self.device_config.device_str

    @property
    def dtype(self) -> torch.dtype:  # pragma: no cover - trivial
        return self.device_config.dtype

    def _load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        cache_key = f"{self.model_name}_{self.device_config.device_str}_{self.device_config.dtype}"
        if cache_key in self._model_cache:
            self.model, self.tokenizer = self._model_cache[cache_key]
            logger.info("Using cached Nomic HF model for %s", cache_key)
            return
        try:
            logger.info(
                "Loading Nomic HF model %s (dtype=%s, device=%s)",
                self.model_name,
                self.device_config.dtype,
                self.device_config.device_str,
            )
            # trust_remote_code since repo may define custom model class
            model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self.device_config.dtype,
                trust_remote_code=True,
            ).eval()
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            try:
                model = model.to(self.device_config.device_str)
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "Could not move Nomic model to %s (%s); using CPU",
                    self.device_config.device_str,
                    e,
                )
            self.model, self.tokenizer = model, tokenizer
            self._model_cache[cache_key] = (self.model, self.tokenizer)
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to load Nomic HF model")
            raise EmbeddingError("Nomic HF load", str(e)) from e

    # Image embedding not supported
    async def embed_images(
        self,
        images: list[Image.Image],
        dtype: torch.dtype | None,
    ) -> list[torch.Tensor]:  # pragma: no cover - interface only
        raise NotImplementedError("NomicHFModel does not support image embeddings")

    def _pool(self, outputs: Any, attention_mask: torch.Tensor | None) -> torch.Tensor:
        # Prefer attribute-based pooling if provided by remote code
        if hasattr(outputs, "embeddings"):
            pooled = outputs.embeddings  # type: ignore[assignment]
            if isinstance(pooled, torch.Tensor):
                return pooled
        hidden = getattr(outputs, "last_hidden_state", None)
        if isinstance(hidden, torch.Tensor):
            if attention_mask is not None and attention_mask.ndim == 2:
                mask = attention_mask.unsqueeze(-1).to(hidden.device)
                # avoid div by zero
                summed = (hidden * mask).sum(1)
                counts = mask.sum(1).clamp_min(1)
                return summed / counts
            # fallback mean over sequence
            return hidden.mean(1)
        # Final fallback: try pooler_output
        po = getattr(outputs, "pooler_output", None)
        if isinstance(po, torch.Tensor):
            return po
        raise EmbeddingError("Nomic HF encode", "Could not pool model outputs")

    async def embed_texts(self, texts: list[str]) -> list[torch.Tensor]:
        if not texts:
            return []
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Nomic HF model not loaded")
        cleaned = [t if (t and t.strip()) else " " for t in texts]
        loop = asyncio.get_event_loop()

        def _run_batch(batch: list[str]) -> list[torch.Tensor]:
            with torch.inference_mode():
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=8192,
                )
                # Move to device (keep integer types as-is)
                for k, v in enc.items():
                    enc[k] = v.to(self.model.device)
                outputs = self.model(**enc)
                pooled = self._pool(outputs, enc.get("attention_mask"))
                if pooled.ndim == 1:
                    pooled = pooled.unsqueeze(0)
                out_list: list[torch.Tensor] = []
                for row in pooled:
                    vec = row.to("cpu", copy=True).float()
                    if self.normalize:
                        vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
                    out_list.append(vec)
                return out_list

        tasks: list[asyncio.Future[list[torch.Tensor]]] = []
        for i in range(0, len(cleaned), self.batch_size):
            batch = cleaned[i : i + self.batch_size]
            tasks.append(loop.run_in_executor(None, _run_batch, batch))
        results_nested = await asyncio.gather(*tasks)
        # Flatten preserving order
        flat: list[torch.Tensor] = [v for sub in results_nested for v in sub]
        if len(flat) != len(texts):  # pragma: no cover
            logger.warning(
                "Nomic HF embedding count mismatch (%d vs %d)",
                len(flat),
                len(texts),
            )
        return flat

    @property
    def embedding_dim(self) -> int:  # pragma: no cover - optional convenience
        if self.model is None:
            return 0
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            for attr in ("hidden_size", "d_model", "dim", "embed_dim"):
                if hasattr(cfg, attr):
                    return int(getattr(cfg, attr))
        return 0


# ---------------------------------------------------------------------------
# Jina v4 embedding model (text + images)
# ---------------------------------------------------------------------------
class JinaV4Model(BaseEmbeddingModel):
    """Jina v4 embedding model supporting text & image embeddings."""

    # Cache loaded model objects keyed by (model_name, device, dtype)
    _model_cache: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        *,
        device_config: DeviceConfig | None = None,
        text_batch_size: int | None = None,
        image_batch_size: int | None = None,
        task_label: str = "retrieval",
        truncate_dim: int | None = None,
        image_max_side: int = 1200,
        force_image_float32: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device_config = device_config or DeviceConfig.auto_detect()
        if self.device_config.device_str != "cuda" and self.device_config.dtype in {
            torch.float16,
            torch.bfloat16,
        }:
            self._override_dtype = torch.float32
            logger.info(
                "Overriding dtype to float32 for stability (device=%s)",
                self.device_config.device_str,
            )
        else:
            self._override_dtype = self.device_config.dtype
        self.text_batch_size = max(1, text_batch_size or 64)
        self.image_batch_size = max(1, image_batch_size or 8)
        self.task_label = task_label
        self.truncate_dim = truncate_dim
        self.image_max_side = image_max_side
        self.force_image_float32 = force_image_float32
        self._model: Any | None = None
        self._supports_images = True
        self._image_dtype_adjusted = False
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._load_model()
        logger.info(
            "Initialized JinaV4Model (%s) on %s (dtype=%s)",
            self.model_name,
            self.device_config.device_str,
            str(self._override_dtype),
        )

    def _load_model(self) -> None:
        if self._model is not None:
            return
        cache_key = f"{self.model_name}_{self.device_config.device_str}_{getattr(self, '_override_dtype', self.device_config.dtype)}"
        if cache_key in self._model_cache:
            self._model = self._model_cache[cache_key]
            logger.info("Using cached Jina model for %s", cache_key)
            return
        load_dtype = getattr(self, "_override_dtype", self.device_config.dtype)
        try:
            logger.info(
                "Loading Jina model %s (dtype=%s, device=%s)",
                self.model_name,
                load_dtype,
                self.device_config.device_str,
            )
            model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=load_dtype,
            )
            try:
                model = model.to(self.device_config.device_str)
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "Could not move Jina model to %s (%s); using CPU",
                    self.device_config.device_str,
                    e,
                )
            self._model = model.eval()
            # Store in cache for reuse
            self._model_cache[cache_key] = self._model
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to load Jina model")
            raise EmbeddingError("Jina model load", str(e)) from e

    def _maybe_truncate(self, emb: torch.Tensor) -> torch.Tensor:
        if self.truncate_dim and self.truncate_dim < emb.shape[-1]:
            return emb[..., : self.truncate_dim]
        return emb

    def _post(self, emb: torch.Tensor) -> torch.Tensor:
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return self._maybe_truncate(emb).to("cpu", copy=True).float()

    def _coerce(self, output: Any) -> list[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            if output.ndim == 1:
                return [output]
            if output.ndim == 2:
                return list(output)
        if (
            isinstance(output, (list, tuple))
            and output
            and all(isinstance(x, torch.Tensor) for x in output)
        ):
            first = output[0]
            if first.ndim == 1:
                return list(output)
            pooled: list[torch.Tensor] = []
            for t in output:
                if t.ndim == 2:
                    pooled.append(t.mean(0))
                elif t.ndim == 1:
                    pooled.append(t)
            return pooled
        raise EmbeddingError("Jina encode", "Unexpected output shape/type")

    async def _encode_texts(
        self,
        texts: list[str],
        *,
        prompt_name: str,
    ) -> list[torch.Tensor]:
        if not texts:
            return []
        if self._model is None:
            raise RuntimeError("Jina model not loaded")
        cleaned = [t if (t and t.strip()) else " " for t in texts]
        loop = asyncio.get_event_loop()

        def _run():
            with torch.inference_mode():
                dev = (
                    self.device_config.device_str
                    if self.device_config.device_str in {"cuda", "cpu"}
                    else "cpu"
                )
                ac = torch.autocast(device_type=dev, enabled=False)
                with ac:  # type: ignore[arg-type]
                    return self._model.encode_text(  # type: ignore[attr-defined]
                        texts=cleaned,
                        task=self.task_label,
                        prompt_name=prompt_name,
                        batch_size=self.text_batch_size,
                    )

        raw = await loop.run_in_executor(None, _run)
        return [self._post(v) for v in self._coerce(raw)]

    async def embed_texts(self, texts: list[str]) -> list[torch.Tensor]:
        return await self._encode_texts(texts, prompt_name="passage")

    async def embed_queries(self, queries: list[str]) -> list[torch.Tensor]:
        return await self._encode_texts(queries, prompt_name="query")

    async def embed_images(
        self,
        images: list[Image.Image | str],
        dtype: torch.dtype | None = None,  # noqa: ARG002
    ) -> list[torch.Tensor]:
        if not images:
            return []
        if self._model is None:
            raise RuntimeError("Jina model not loaded")
        if not self._supports_images:
            return await self.embed_texts(["image"] * len(images))

        def _prepare(img_in: Image.Image | str) -> Image.Image:
            im = Image.open(img_in) if isinstance(img_in, str) else img_in
            if im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size
            ms = max(w, h)
            if ms > self.image_max_side:
                scale = self.image_max_side / ms
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            return im

        processed: list[Image.Image] = []
        for img in images:
            try:
                processed.append(_prepare(img))
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to load/prepare image (%s); using blank", e)
                processed.append(Image.new("RGB", (64, 64), color="white"))

        loop = asyncio.get_event_loop()
        outputs: list[torch.Tensor] = []
        for i in range(0, len(processed), self.image_batch_size):
            batch_imgs = processed[i : i + self.image_batch_size]

            def _encode_batch(batch=batch_imgs):
                with torch.inference_mode():
                    dev = (
                        self.device_config.device_str
                        if self.device_config.device_str in {"cuda", "cpu"}
                        else "cpu"
                    )
                    ac = torch.autocast(device_type=dev, enabled=False)
                    with ac:  # type: ignore[arg-type]
                        result = self._model.encode_image(  # type: ignore[attr-defined]
                            images=batch,
                            task=self.task_label,
                            batch_size=min(len(batch), self.image_batch_size),
                        )
                        # Clear intermediate tensors from GPU memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return result

            try:
                raw = await loop.run_in_executor(None, _encode_batch)
                vecs = [self._post(v) for v in self._coerce(raw)]
                if len(vecs) != len(batch_imgs):
                    logger.warning(
                        "Jina image batch mismatch (%d vectors for %d imgs)",
                        len(vecs),
                        len(batch_imgs),
                    )
                outputs.extend(vecs[: len(batch_imgs)])
            except Exception as e:  # pragma: no cover
                if (
                    self.force_image_float32
                    and not self._image_dtype_adjusted
                    and ("autocast" in str(e).lower() or "scalartype" in str(e).lower())
                ):
                    try:
                        logger.warning(
                            "Image batch failed due to dtype (%s); upcasting model to float32 and retrying once",
                            e,
                        )
                        self._model.float()  # type: ignore[call-arg]
                        self._image_dtype_adjusted = True

                        def _encode_batch_retry(batch=batch_imgs):
                            with torch.inference_mode():
                                dev_r = (
                                    self.device_config.device_str
                                    if self.device_config.device_str in {"cuda", "cpu"}
                                    else "cpu"
                                )
                                ac_r = torch.autocast(device_type=dev_r, enabled=False)
                                with ac_r:  # type: ignore[arg-type]
                                    return self._model.encode_image(  # type: ignore[attr-defined]
                                        images=batch,
                                        task=self.task_label,
                                        batch_size=min(
                                            len(batch),
                                            self.image_batch_size,
                                        ),
                                    )

                        raw_retry = await loop.run_in_executor(
                            None,
                            _encode_batch_retry,
                        )
                        vecs_retry = [self._post(v) for v in self._coerce(raw_retry)]
                        outputs.extend(vecs_retry[: len(batch_imgs)])
                        continue
                    except Exception as up_e:  # pragma: no cover
                        logger.warning("Retry after upcasting failed: %s", up_e)
                logger.warning(
                    "Image batch (%d) failed (%s); retrying individually",
                    len(batch_imgs),
                    e,
                )
                for single in batch_imgs:
                    try:

                        def _encode_single(img=single):
                            with torch.inference_mode():
                                dev_s = (
                                    self.device_config.device_str
                                    if self.device_config.device_str in {"cuda", "cpu"}
                                    else "cpu"
                                )
                                ac_s = torch.autocast(device_type=dev_s, enabled=False)
                                with ac_s:  # type: ignore[arg-type]
                                    return self._model.encode_image(  # type: ignore[attr-defined]
                                        images=[img],
                                        task=self.task_label,
                                        batch_size=1,
                                    )

                        raw_one = await loop.run_in_executor(None, _encode_single)
                        one_vecs = [self._post(v) for v in self._coerce(raw_one)]
                        if one_vecs:
                            outputs.append(one_vecs[0])
                    except Exception as se:  # pragma: no cover
                        logger.warning(
                            "Single image failed (%s); using placeholder",
                            se,
                        )
                        placeholder = (await self.embed_texts(["image"]))[0]
                        outputs.append(placeholder)

        if not outputs:
            self._supports_images = False
            return await self.embed_texts(["image"] * len(images))
        return outputs

    @property
    def embedding_dim(self) -> int:
        if self.truncate_dim:
            return self.truncate_dim
        cfg = getattr(self._model, "config", None)
        if cfg is not None:
            return getattr(cfg, "hidden_size", 0)
        return 0

    @property
    def processor(self):
        # For compatibility with MultiModalRAG, return self as processor
        return self

    def score(
        self,
        query_vectors: list[torch.Tensor],
        doc_vectors: list[torch.Tensor],
    ) -> torch.Tensor:
        """Calculates cosine similarity scores between a query and document vectors."""
        if not query_vectors or not doc_vectors:
            # Return an empty tensor if inputs are empty
            return torch.tensor([])

        # Ensure the query vector is on the same device as the document vectors
        device = doc_vectors[0].device
        q = query_vectors[0].to(device).float().unsqueeze(0)  # (1, D)
        q = torch.nn.functional.normalize(q, p=2, dim=-1)

        # Stack and normalize document vectors efficiently
        docs = torch.stack([d.to(device).float() for d in doc_vectors])  # (N, D)
        docs = torch.nn.functional.normalize(docs, p=2, dim=-1)

        # Calculate scores using matrix multiplication (dot product for normalized vectors)
        scores = torch.matmul(q, docs.T).squeeze(0)  # (N,)

        return scores


# ---------------------------------------------------------------------------
# ColQwen2 multimodal model
# ---------------------------------------------------------------------------
class ColQwen2Model(BaseEmbeddingModel):
    """ColQwen2 model (text + image embeddings)."""

    _model_cache: ClassVar[dict[str, tuple[ColQwen2, ColQwen2Processor]]] = {}

    def __init__(self, device_config: DeviceConfig):
        self.model_name = "vidore/colqwen2-v1.0"
        self.device_config = device_config
        self.model: ColQwen2 | None = None
        self.processor: ColQwen2Processor | None = None
        self._load_model()
        logger.info("Initialized ColQwen2Model on %s", device_config.device_str)

    @property
    def device(self) -> str:
        return self.device_config.device_str

    @property
    def dtype(self) -> torch.dtype:
        return self.device_config.dtype

    def _process_single_image(
        self,
        image: Image.Image,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        assert self.processor is not None and self.model is not None, "Model not loaded"
        try:
            processed_image = self.processor.process_images([image])
            processed_image = self._move_to_device_with_dtype(processed_image, dtype)
            with torch.inference_mode():
                embedding = self.model(**processed_image)
            return self._check_and_clean_tensor(embedding[0].to("cpu"), "image")
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to process image")
            raise EmbeddingError("Image embedding", str(e)) from e

    def _process_single_text(self, text: str) -> torch.Tensor:
        assert self.processor is not None and self.model is not None, "Model not loaded"
        try:
            processed_query = self.processor.process_queries([text])
            processed_query = self._move_to_device_without_dtype(processed_query)
            with torch.inference_mode():
                embedding = self.model(**processed_query)
            context = f"text: '{text[:50]}...'"
            return self._check_and_clean_tensor(embedding[0].to("cpu"), context)
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to process text '%s'", text[:50])
            raise EmbeddingError("Text embedding", str(e)) from e

    def _move_to_device_with_dtype(self, inputs: dict, dtype: torch.dtype) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        try:
            return {
                k: v.to(self.model.device).to(
                    self.model.dtype if v.dtype.is_floating_point else v.dtype,
                )
                for k, v in inputs.items()
            }
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "Could not move inputs to %s, using CPU: %s",
                self.model.device,
                e,
            )
            return {
                k: v.to("cpu").to(dtype if v.dtype.is_floating_point else v.dtype)
                for k, v in inputs.items()
            }

    def _move_to_device_without_dtype(self, inputs: dict) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        try:
            return {k: v.to(self.model.device) for k, v in inputs.items()}
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "Could not move text inputs to %s, falling back to CPU: %s",
                self.model.device,
                e,
            )
            return {k: v.to("cpu") for k, v in inputs.items()}

    def _check_and_clean_tensor(
        self,
        tensor: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        if torch.isnan(tensor).any():
            logger.warning(
                "NaN detected in embedding for %s, replacing with zeros",
                context,
            )
            return torch.zeros_like(tensor)
        return tensor

    async def embed_images(
        self,
        images: list[Image.Image],
        dtype: torch.dtype | None = None,
    ) -> list[torch.Tensor]:
        if not images:
            return []
        if dtype is None:
            dtype = self.device_config.dtype
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self._process_single_image, img, dtype)
            for img in images
        ]
        embeddings = await asyncio.gather(*tasks)
        return embeddings

    async def embed_texts(self, texts: list[str]) -> list[torch.Tensor]:
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self._process_single_text, text)
            for text in texts
        ]
        embeddings = await asyncio.gather(*tasks)
        return embeddings

    def _load_model(self) -> None:
        cache_key = f"{self.model_name}_{self.device_config.device_str}_{self.device_config.dtype}"
        if cache_key in self._model_cache:
            logger.info("Using cached ColQwen2 model for %s", cache_key)
            self.model, self.processor = self._model_cache[cache_key]
            return
        try:
            logger.info("Loading ColQwen2 model (%s)", self.model_name)
            self.model = ColQwen2.from_pretrained(
                self.model_name,
                torch_dtype=self.device_config.dtype,
            ).eval()
            try:
                self.model = self.model.to(self.device_config.device_str)
            except RuntimeError as e:  # pragma: no cover
                if "offloaded" not in str(e).lower():
                    logger.warning(
                        "Could not move ColQwen2 model to %s: %s",
                        self.device_config.device_str,
                        e,
                    )
            self.processor = ColQwen2Processor.from_pretrained(self.model_name)
            self._model_cache[cache_key] = (self.model, self.processor)
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to load ColQwen2 model")
            raise EmbeddingError("ColQwen2 load", str(e)) from e


class ColPaliModel(BaseEmbeddingModel):
    """ColPali multimodal embedding model with fallback versions and masking."""

    _model_cache: ClassVar[dict[str, tuple[ColPali, ColPaliProcessor]]] = {}

    def __init__(self, device_config: DeviceConfig):
        self.model_name = "vidore/colpali-v1.2"
        self.device_config = device_config
        self.model: ColPali | None = None
        self.processor: ColPaliProcessor | None = None
        self._load_model()
        logger.info("Initialized ColPaliModel on %s", device_config.device_str)

    @property
    def device(self) -> str:
        return self.device_config.device_str

    @property
    def dtype(self) -> torch.dtype:
        return self.device_config.dtype

    def _process_single_image(
        self,
        image: Image.Image,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.processor is None or self.model is None:
            raise RuntimeError("ColPaliModel not loaded before image processing")
        try:
            processed_image = self.processor.process_images([image])
            processed_image = self._move_to_device_with_dtype(processed_image, dtype)
            with torch.inference_mode():
                embedding = self.model(**processed_image)
            emb = self._check_and_clean_tensor(embedding[0].to("cpu"), "image")
            return emb.to(torch.float32)
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to process image")
            raise EmbeddingError("Image embedding", str(e)) from e

    def _process_single_text(self, text: str) -> torch.Tensor:
        if self.processor is None or self.model is None:
            raise RuntimeError("ColPaliModel not loaded before text processing")
        try:
            processed_query = self.processor.process_queries([text])
            processed_query = self._move_to_device_without_dtype(processed_query)
            with torch.inference_mode():
                embedding = self.model(**processed_query)
            emb = self._check_and_clean_tensor(
                embedding[0].to("cpu"),
                f"text: '{text[:50]}...'",
            )
            return emb.to(torch.float32)
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to process text '%s'", text[:50])
            raise EmbeddingError("Text embedding", str(e)) from e

    def _move_to_device_with_dtype(self, inputs: dict, dtype: torch.dtype) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        try:
            return {
                k: v.to(self.model.device).to(
                    self.model.dtype if v.dtype.is_floating_point else v.dtype,
                )
                for k, v in inputs.items()
            }
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "Could not move inputs to %s, using CPU: %s",
                self.model.device,
                e,
            )
            return {
                k: v.to("cpu").to(dtype if v.dtype.is_floating_point else v.dtype)
                for k, v in inputs.items()
            }

    def _move_to_device_without_dtype(self, inputs: dict) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        try:
            return {k: v.to(self.model.device) for k, v in inputs.items()}
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "Could not move text inputs to %s, falling back to CPU: %s",
                self.model.device,
                e,
            )
            return {k: v.to("cpu") for k, v in inputs.items()}

    def _check_and_clean_tensor(
        self,
        tensor: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        if torch.isnan(tensor).any():
            logger.warning(
                "NaN detected in embedding for %s, replacing with zeros",
                context,
            )
            return torch.zeros_like(tensor)
        return tensor

    async def embed_images(
        self,
        images: list[Image.Image],
        dtype: torch.dtype | None = None,
    ) -> list[torch.Tensor]:
        if not images:
            return []
        if dtype is None:
            dtype = self.device_config.dtype
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self._process_single_image, img, dtype)
            for img in images
        ]
        embeddings = await asyncio.gather(*tasks)
        return embeddings

    async def embed_texts(self, texts: list[str]) -> list[torch.Tensor]:
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self._process_single_text, text)
            for text in texts
        ]
        embeddings = await asyncio.gather(*tasks)
        return embeddings

    def _load_model(self) -> None:
        cache_key = f"{self.model_name}_{self.device_config.device_str}_{self.device_config.dtype}"
        if cache_key in self._model_cache:
            logger.info("Using cached ColPali model for %s", cache_key)
            self.model, self.processor = self._model_cache[cache_key]
            return
        logger.info("Loading ColPali model (with fallbacks)")
        models_to_try = [
            "vidore/colpali-v1.3",
            "vidore/colpali-v1.2",
            "vidore/colpali-v1.0",
            "vidore/colpali-v0.3",
        ]
        model_loaded = False
        for name in models_to_try:
            try:
                logger.info(
                    "Attempting to load %s (mask_non_image_embeddings=True)",
                    name,
                )
                self.model = ColPali.from_pretrained(
                    name,
                    torch_dtype=self.device_config.dtype,
                    mask_non_image_embeddings=True,
                ).eval()
                self.model_name = name
                model_loaded = True
                break
            except (OSError, ValueError, RuntimeError) as e:  # pragma: no cover
                logger.warning("Failed to load %s: %s", name, str(e)[:120])
                continue
        if not model_loaded:
            raise RuntimeError("Failed to load any ColPali model version")
        try:
            try:
                self.model = self.model.to(self.device_config.device_str)
            except RuntimeError as e:  # pragma: no cover
                if "offloaded" not in str(e).lower():
                    logger.warning(
                        "Failed to move ColPali model to %s: %s",
                        self.device_config.device_str,
                        e,
                    )
            self.processor = ColPaliProcessor.from_pretrained(self.model_name)
            self._model_cache[cache_key] = (self.model, self.processor)
            logger.info("ColPali model loaded and cached (%s)", self.model_name)
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to finalize ColPali model load")
            raise EmbeddingError("ColPali load", str(e)) from e


def setup_embedding_model(
    model_name: str,
    device_config: DeviceConfig,
) -> BaseEmbeddingModel:
    if model_name == "nomic_ollama_embed":
        return NomicOllamaModel(
            ollama_url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text",
        )
    if model_name == "nomic_hf_embed":
        return NomicHFModel(device_config=device_config)
    if model_name == "colqwen2_embed":
        return ColQwen2Model(device_config=device_config)
    if model_name == "colpali_embed":
        return ColPaliModel(device_config=device_config)
    if model_name == "jina_embed":
        return JinaV4Model(device_config=device_config)
    supported = [
        "nomic_ollama_embed",
        "nomic_hf_embed",
        "colqwen2_embed",
        "colpali_embed",
        "jina_embed",
    ]
    raise ValueError(
        f"Unsupported model name: {model_name}. Supported models: {supported}",
    )
