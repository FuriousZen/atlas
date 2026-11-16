from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any

from app.config import get_settings
from app.store import fake_embedding

logger = logging.getLogger(__name__)


class Embedder:
    """Lazy embedding adapter with a deterministic fallback.

    Real model loading is intentionally lazy. A missing provider or missing HF
    model must not prevent the API from starting or tests from running.
    """

    def __init__(self, model_id: str | None = None, use_local_model: bool = False) -> None:
        self.model_id = model_id or get_settings().atlas_embed_model
        self.use_local_model = use_local_model or get_settings().atlas_use_local_models
        self._model: Any = None
        self._lock = threading.Lock()
        self._fake_cache: dict[str, list[float]] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.use_local_model:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.warning("local embedding model requested but sentence-transformers is not installed; using fake embeddings")
                return
            try:
                self._model = SentenceTransformer(self.model_id, device="cpu")
            except Exception:
                logger.warning("local embedding model failed to load; using fake embeddings", exc_info=True)
                self._model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        if self._model is None:
            from app.metrics import EMBEDDING_CACHE_HITS

            output: list[list[float]] = []
            for text in texts:
                key = embedding_key(text, self.model_id)
                cached = self._fake_cache.get(key)
                if cached is None:
                    cached = fake_embedding(text)
                    self._fake_cache[key] = cached
                    try:
                        EMBEDDING_CACHE_HITS.labels("miss").inc()
                    except Exception:
                        logger.debug("embedding miss metric failed", exc_info=True)
                else:
                    try:
                        EMBEDDING_CACHE_HITS.labels("memory").inc()
                    except Exception:
                        logger.debug("embedding hit metric failed", exc_info=True)
                output.append(cached)
            return output
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=16)
        return vectors.tolist()

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]

    @property
    def dimensions(self) -> int:
        return int(self._model.get_sentence_embedding_dimension()) if self._model is not None else 128


def embedding_key(text: str, model_id: str | None = None) -> str:
    model = model_id or get_settings().atlas_embed_model
    return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()
