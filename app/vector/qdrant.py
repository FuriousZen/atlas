from __future__ import annotations

import logging
from typing import Any

from app.models import Chunk, Paper, SearchHit
from app.store import InMemoryStore, get_store
from app.tools.embedder import Embedder

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant adapter with the same local fallback used by tests and the API.

    The client is imported lazily so a missing Qdrant package or service does
    not block the provider-free development path.
    """

    def __init__(self, store: InMemoryStore | None = None, embedder: Embedder | None = None) -> None:
        self.local = store or get_store()
        self.embedder = embedder or Embedder()
        self.client: Any = None
        self.collection: str | None = None
        self.dimensions: int | None = None

    def connect(self, url: str, collection: str, api_key: str | None = None) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            return
        client = QdrantClient(url=url, api_key=api_key)
        dimension = self.embedder.dimensions
        collections = {item.name for item in client.get_collections().collections}
        if collection not in collections:
            client.create_collection(collection_name=collection, vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))
            self.dimensions = dimension
            self._ensure_payload_indexes(client, collection)
        else:
            try:
                info = client.get_collection(collection_name=collection)
                params = getattr(getattr(info, "config", None), "params", None)
                vectors = getattr(params, "vectors", None)
                existing = getattr(vectors, "size", None) if vectors is not None else None
            except Exception:
                existing = None
            if isinstance(existing, int):
                self.dimensions = existing
                if existing != dimension:
                    raise ValueError(f"qdrant collection {collection!r} has dimension {existing}, Embedder reports {dimension}; refusing to mix dimensions")
            else:
                self.dimensions = dimension
        self.client, self.collection = client, collection

    @staticmethod
    def _ensure_payload_indexes(client: Any, collection: str) -> None:
        try:
            from qdrant_client.models import PayloadSchemaType
        except ImportError:
            return
        for key, schema in (("topic", "keyword"), ("citations", "integer"), ("ingested_at", "datetime")):
            try:
                field_type = getattr(PayloadSchemaType, schema.upper(), schema)
                client.create_payload_index(collection_name=collection, field_name=key, field_schema=field_type)
            except Exception:
                logger.debug("qdrant payload index %s unavailable; continuing", key, exc_info=True)
                continue

    def upsert(self, paper: Paper, chunks: list[Chunk]) -> None:
        if self.client is None or self.collection is None:
            self.local.upsert(paper, chunks)
            return
        if self.dimensions is not None and self.embedder.dimensions != self.dimensions:
            raise ValueError(f"Embedder dimension {self.embedder.dimensions} disagrees with collection dimension {self.dimensions}; refusing upsert")
        from qdrant_client.models import PointStruct

        points = []
        for chunk in chunks:
            vector = chunk.vector or self.embedder.encode_one(chunk.text)
            chunk.vector = vector
            points.append(PointStruct(id=chunk.id, vector=vector, payload={"paper_id": paper.id, "chunk_id": chunk.id, "section": chunk.section, "text": chunk.text, "citations": paper.citations, "topic": paper.topic, "title": paper.title, "url": paper.url}))
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        self.local.upsert(paper, chunks)

    def search(self, query: str, topic: str | None = None, min_citations: int | None = None, limit: int = 12) -> list[SearchHit]:
        if self.client is None or self.collection is None:
            return self.local.search(query, topic, min_citations, limit=limit)
        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

        conditions = []
        if topic:
            conditions.append(FieldCondition(key="topic", match=MatchValue(value=topic)))
        if min_citations is not None:
            conditions.append(FieldCondition(key="citations", range=Range(gte=min_citations)))
        result = self.client.query_points(collection_name=self.collection, query=self.embedder.encode_one(query), query_filter=Filter(must=conditions) if conditions else None, limit=limit, with_payload=True).points
        output: list[SearchHit] = []
        for point in result:
            payload = point.payload or {}
            paper = self.local.get_paper(str(payload.get("paper_id", "")))
            if paper is None:
                continue
            chunk = Chunk(id=str(payload.get("chunk_id", point.id)), paper_id=paper.id, n=0, section=str(payload.get("section", "unknown")), text=str(payload.get("text", "")))
            output.append(SearchHit(chunk=chunk, paper=paper, score=float(point.score)))
        return output

    def delete(self, paper_ids: list[str]) -> None:
        # NOTE (single-owner rule): InMemoryStore.enforce_capacity owns eviction.
        # This delete is a stand-in pass-through, not a second eviction policy.
        if self.client is not None and self.collection is not None:
            from qdrant_client.models import FieldCondition, Filter, MatchAny

            self.client.delete(collection_name=self.collection, points_selector=Filter(must=[FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))]), wait=True)
        for paper_id in paper_ids:
            self.local.papers.pop(paper_id, None)
            for chunk_id in [key for key, chunk in self.local.chunks.items() if chunk.paper_id == paper_id]:
                self.local.chunks.pop(chunk_id, None)
