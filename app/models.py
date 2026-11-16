from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Paper:
    id: str
    title: str
    authors: list[str]
    venue: str | None
    year: int | None
    citations: int
    topic: str
    url: str
    abstract: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_url: str | None = None
    ingested_at: datetime = field(default_factory=utcnow)
    summarized_at: datetime | None = None
    why_kept: str = ""
    extraction: str = "abstract_only"
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("ingested_at", "summarized_at"):
            if value[key] is not None:
                value[key] = value[key].isoformat()
        return value


@dataclass
class Chunk:
    id: str
    paper_id: str
    n: int
    section: str
    text: str
    page: int | None = None
    vector: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("vector", None)
        return value


@dataclass
class SearchHit:
    chunk: Chunk
    paper: Paper
    score: float
    lexical_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper.id,
            "chunk_id": self.chunk.id,
            "section": self.chunk.section,
            "page": self.chunk.page,
            "text": self.chunk.text,
            "score": round(self.score, 6),
            "citations": self.paper.citations,
            "title": self.paper.title,
            "url": self.paper.url,
            "topic": self.paper.topic,
        }


@dataclass
class Summary:
    tldr: str
    contributions: list[str]
    method: str
    results: dict[str, Any]
    limits: str
    why_kept: str
    extraction: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
