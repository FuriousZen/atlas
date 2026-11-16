from __future__ import annotations

import hashlib
import math
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Iterable

from app.config import Settings, get_settings
from app.models import Chunk, Paper, SearchHit


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+-]*")


class CapacityBlocked(Exception):
    """Raised when capacity overflow cannot be relieved without evicting young papers.

    429 contract: callers that admit new content (run_ingestion, POST /ingest/run)
    must NOT evict papers with ingested_at > now - MIN_KEEP_DAYS. Instead they
    surface this exception so the HTTP layer can answer 429 with a Retry-After
    hint. Background ingest runs record it in the run registry
    (status "capacity_blocked", error "capacity_blocked", retry_after seconds)
    instead of a generic error; direct run_ingestion() callers see the exception.
    """

    def __init__(self, needed: int, retry_after: int) -> None:
        super().__init__(f"storage bound would evict {needed} papers within grace period")
        self.needed = needed
        self.retry_after = retry_after


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def fake_embedding(text: str, dimensions: int = 128) -> list[float]:
    """Stable local embedding used when the Hugging Face model is unavailable."""
    values = [0.0] * dimensions
    words = sorted(tokens(text)) or ["empty"]
    for word in words:
        digest = hashlib.sha256(word.encode()).digest()
        for offset in range(0, min(len(digest), dimensions), 4):
            index = int.from_bytes(digest[offset : offset + 2], "big") % dimensions
            values[index] += (digest[offset + 2] / 255.0) - 0.5
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return sum(left[i] * right[i] for i in range(size))


class InMemoryStore:
    """Deterministic store for local development and tests.

    Production adapters can use the same interface with Postgres and Qdrant.
    The in-memory mode is intentionally useful: it allows the agent API to run
    before a provider, database, or container runtime has been configured.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.papers: dict[str, Paper] = {}
        self.chunks: dict[str, Chunk] = {}
        self._lock = threading.RLock()
        self.evictions: list[dict[str, str | int]] = []

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {"papers": len(self.papers), "chunks": len(self.chunks), "bytes": sum(len(c.text.encode()) for c in self.chunks.values())}

    def upsert(self, paper: Paper, chunks: Iterable[Chunk]) -> None:
        with self._lock:
            self.papers[paper.id] = paper
            for chunk in chunks:
                if chunk.vector is None:
                    chunk.vector = fake_embedding(chunk.text)
                self.chunks[chunk.id] = chunk

    def get_paper(self, paper_id: str) -> Paper | None:
        return self.papers.get(paper_id)

    def list_papers(self, topic: str | None = None, since: datetime | None = None, min_citations: int | None = None, limit: int = 100, offset: int = 0, sort: str = "citations") -> list[Paper]:
        rows = list(self.papers.values())
        if topic:
            rows = [row for row in rows if row.topic == topic]
        if since:
            rows = [row for row in rows if row.ingested_at >= since]
        if min_citations is not None:
            rows = [row for row in rows if row.citations >= min_citations]
        if sort == "recent":
            rows.sort(key=lambda row: (row.ingested_at, row.citations), reverse=True)
        else:
            rows.sort(key=lambda row: (row.citations, row.ingested_at), reverse=True)
        return rows[offset : offset + limit]

    def search(self, query: str, topic: str | None = None, min_citations: int | None = None, since: datetime | None = None, section: list[str] | None = None, limit: int = 12, sort: str = "citations") -> list[SearchHit]:
        query_vector = fake_embedding(query)
        query_tokens = tokens(query)
        hits: list[SearchHit] = []
        for chunk in self.chunks.values():
            paper = self.papers.get(chunk.paper_id)
            if paper is None or (topic and paper.topic != topic) or (min_citations is not None and paper.citations < min_citations) or (since and paper.ingested_at < since) or (section and chunk.section not in section):
                continue
            lexical = len(query_tokens & tokens(f"{paper.title} {paper.abstract} {chunk.text}")) / max(len(query_tokens), 1)
            dense = cosine(query_vector, chunk.vector or fake_embedding(chunk.text))
            score = max(0.0, 0.75 * dense + 0.25 * lexical)
            hits.append(SearchHit(chunk=chunk, paper=paper, score=score, lexical_score=lexical))
        if sort == "recent":
            hits.sort(key=lambda hit: (hit.paper.ingested_at, hit.score), reverse=True)
        else:
            hits.sort(key=lambda hit: (hit.score, hit.paper.citations), reverse=True)
        return hits[:limit]

    def enforce_capacity(self, incoming_papers: int = 0, incoming_chunks: int = 0, incoming_bytes: int = 0) -> list[str]:
        """Evict old eligible papers before indexing new content.

        Victims are computed per dimension (papers, chunks, bytes) and converted
        to a paper count via per-paper averages, then the max wins. Only papers
        older than the grace period are eligible; when victims are needed but
        none are eligible, CapacityBlocked is raised (HTTP 429) instead of
        evicting young papers.

        NOTE (single-owner rule): InMemoryStore owns capacity/eviction. The
        QdrantStore/MCP delete paths are stand-ins and must not grow their own
        eviction math; route any future wired store through this one unit.
        """
        with self._lock:
            current_papers = len(self.papers)
            current_chunks = len(self.chunks)
            current_bytes = sum(len(c.text.encode()) for c in self.chunks.values())
            max_papers = self.settings.atlas_max_papers
            max_chunks = self.settings.atlas_max_chunks
            max_bytes = self.settings.max_bytes

            papers_needed = max(0, current_papers + incoming_papers - max_papers)

            chunk_overflow = max(0, current_chunks + incoming_chunks - max_chunks)
            if chunk_overflow > 0:
                if current_papers > 0:
                    per_paper_chunks: float = current_chunks / current_papers
                elif incoming_papers > 0:
                    per_paper_chunks = incoming_chunks / incoming_papers
                else:
                    per_paper_chunks = 1.0
                per_paper_chunks = max(1.0, per_paper_chunks)
                chunks_needed = math.ceil(chunk_overflow / per_paper_chunks)
            else:
                chunks_needed = 0

            byte_overflow = max(0, current_bytes + incoming_bytes - max_bytes)
            if byte_overflow > 0:
                if current_papers > 0:
                    per_paper_bytes: float = current_bytes / current_papers
                elif incoming_papers > 0:
                    per_paper_bytes = incoming_bytes / incoming_papers
                else:
                    per_paper_bytes = 1.0
                per_paper_bytes = max(1.0, per_paper_bytes)
                bytes_needed = math.ceil(byte_overflow / per_paper_bytes)
            else:
                bytes_needed = 0

            victims_needed = max(papers_needed, chunks_needed, bytes_needed)
            if victims_needed <= 0:
                over = (
                    current_papers + incoming_papers > max_papers
                    or current_chunks + incoming_chunks > max_chunks
                    or current_bytes + incoming_bytes > max_bytes
                )
                if not over:
                    return []
                victims_needed = 1
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(days=self.settings.atlas_min_keep_days)
            eligible = [paper for paper in self.papers.values() if paper.ingested_at <= cutoff]
            if self.settings.atlas_eviction == "citation_agnostic":
                eligible.sort(key=lambda paper: paper.ingested_at)
            else:
                eligible.sort(key=lambda paper: paper.ingested_at.timestamp() + min(7 * 86400, math.log1p(max(0, paper.citations)) * 12 * 3600))
            if len(eligible) < victims_needed:
                young = [paper for paper in self.papers.values() if paper.ingested_at > cutoff]
                if young:
                    oldest_young = min(paper.ingested_at for paper in young)
                    retry_after = int((oldest_young + timedelta(days=self.settings.atlas_min_keep_days) - now).total_seconds())
                    retry_after = max(1, retry_after)
                else:
                    retry_after = self.settings.atlas_min_keep_days * 86400
                raise CapacityBlocked(needed=victims_needed, retry_after=retry_after)
            victims = eligible[:victims_needed]
            victim_ids = [paper.id for paper in victims]
            for paper_id in victim_ids:
                self.papers.pop(paper_id, None)
                for chunk_id in [cid for cid, chunk in self.chunks.items() if chunk.paper_id == paper_id]:
                    self.chunks.pop(chunk_id, None)
                self.evictions.append({"paper_id": paper_id, "evicted_at": datetime.now(timezone.utc).isoformat()})
            return victim_ids


_store: InMemoryStore | None = None


def get_store() -> InMemoryStore:
    global _store
    if _store is None:
        _store = InMemoryStore()
    return _store


def reset_store() -> InMemoryStore:
    global _store
    _store = InMemoryStore()
    return _store
