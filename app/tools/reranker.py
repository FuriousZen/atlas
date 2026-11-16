from __future__ import annotations

from typing import Any

from app.models import SearchHit


class Reranker:
    """Cross-encoder adapter; lexical scoring remains safe when HF is absent."""

    def __init__(self, model_id: str | None = None, use_local_model: bool = False) -> None:
        from app.config import get_settings

        self.model_id = model_id or get_settings().atlas_rerank_model
        self.use_local_model = use_local_model
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None or not self.use_local_model:
            return
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_id, max_length=512, device="cpu")

    def rerank(self, query: str, hits: list[SearchHit], limit: int = 5) -> list[SearchHit]:
        self._load()
        if self._model is None:
            return sorted(hits, key=lambda hit: (hit.score + hit.lexical_score * 0.1, hit.paper.citations), reverse=True)[:limit]
        scores = self._model.predict([(query, hit.chunk.text) for hit in hits], batch_size=8)
        ranked = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
        for hit, score in ranked:
            hit.score = float(score)
        return [hit for hit, _ in ranked[:limit]]
