from datetime import datetime, timedelta, timezone

import pytest

from app.config import Settings
from app.models import Chunk, Paper
from app.store import CapacityBlocked, InMemoryStore


def paper(identifier: str, days_old: int, citations: int = 100) -> Paper:
    return Paper(id=identifier, title=f"Research {identifier}", authors=["A"], venue="Journal", year=2025, citations=citations, topic="retrieval", url=f"https://example.invalid/{identifier}", abstract="retrieval evidence", ingested_at=datetime.now(timezone.utc) - timedelta(days=days_old))


def test_search_returns_semantic_and_lexical_hit():
    store = InMemoryStore(Settings(_env_file=None))
    item = paper("p1", 1)
    store.upsert(item, [Chunk("p1#0", "p1", 0, "results", "neural reranker improves retrieval")])
    hits = store.search("retrieval reranker")
    assert hits and hits[0].paper.id == "p1"


def test_oldest_paper_is_evicted_at_capacity():
    settings = Settings(_env_file=None, atlas_max_papers=2, atlas_max_chunks=100)
    store = InMemoryStore(settings)
    for identifier, age in [("old", 20), ("new", 10)]:
        item = paper(identifier, age)
        store.upsert(item, [Chunk(f"{identifier}#0", identifier, 0, "abstract", identifier)])
    victims = store.enforce_capacity(incoming_papers=1, incoming_chunks=1)
    assert victims == ["old"]
    assert store.get_paper("old") is None
    assert store.get_paper("new") is not None


def test_grace_blocked_raises_capacity_blocked():
    settings = Settings(_env_file=None, atlas_max_papers=1, atlas_max_chunks=100, atlas_min_keep_days=7)
    store = InMemoryStore(settings)
    item = paper("young", 1)
    store.upsert(item, [Chunk("young#0", "young", 0, "abstract", "young")])
    with pytest.raises(CapacityBlocked) as exc_info:
        store.enforce_capacity(incoming_papers=1, incoming_chunks=1)
    assert exc_info.value.needed == 1
    assert exc_info.value.retry_after > 0
    assert store.get_paper("young") is not None


def test_max_papers_three_seed_five_ingest_two_evicts_oldest():
    settings = Settings(_env_file=None, atlas_max_papers=3, atlas_max_chunks=100000, atlas_max_bytes_mb=2048)
    store = InMemoryStore(settings)
    for index in range(5):
        identifier = f"p{index}"
        item = paper(identifier, 20 + index)
        store.upsert(item, [Chunk(f"{identifier}#0", identifier, 0, "abstract", identifier)])
    victims = store.enforce_capacity(incoming_papers=2, incoming_chunks=2)
    assert victims == ["p4", "p3", "p2", "p1"]
    assert store.get_paper("p0") is not None
    for identifier in ("p1", "p2", "p3", "p4"):
        assert store.get_paper(identifier) is None


def test_byte_pressure_only_evicts_multiple_papers():
    settings = Settings(_env_file=None, atlas_max_papers=100, atlas_max_chunks=100000, atlas_max_bytes_mb=1)
    store = InMemoryStore(settings)
    payload = "x" * 200_000
    for index in range(5):
        identifier = f"b{index}"
        item = paper(identifier, 20 + index)
        store.upsert(item, [Chunk(f"{identifier}#0", identifier, 0, "abstract", payload)])
    victims = store.enforce_capacity(incoming_papers=0, incoming_chunks=0, incoming_bytes=500_000)
    assert len(victims) == 3
    assert victims == ["b4", "b3", "b2"]


def test_low_citation_paper_evicts_before_high_citation_peer():
    settings = Settings(_env_file=None, atlas_max_papers=2, atlas_max_chunks=100)
    store = InMemoryStore(settings)
    ingested_at = datetime.now(timezone.utc) - timedelta(days=20)
    low = paper("low", 20, citations=0)
    low.ingested_at = ingested_at
    high = paper("high", 20, citations=10000)
    high.ingested_at = ingested_at
    store.upsert(low, [Chunk("low#0", "low", 0, "abstract", "low")])
    store.upsert(high, [Chunk("high#0", "high", 0, "abstract", "high")])
    victims = store.enforce_capacity(incoming_papers=1, incoming_chunks=1)
    assert victims == ["low"]
    assert store.get_paper("low") is None
    assert store.get_paper("high") is not None
