from app.agent.graphs import run_ingestion, run_query
from app.ingest.sources import CandidatePaper


def test_query_graph_handles_natural_prompt():
    run_ingestion([CandidatePaper(id="p1", title="New neural retrieval findings", authors=["A"], venue="Journal", year=2025, citations=100, topic="retrieval", url="https://example.invalid/p1", abstract="New science about retrieval with negative findings.")], "retrieval", 50)
    result = run_query("most damning scientific discoveries published?", {"topic": "retrieval"})
    assert result["model"] == "atlas-stub-heuristic"
    assert result["citations"]
    assert result["filters"]["section"] == ["limits", "results"]


def test_run_query_respects_since_filter():
    from datetime import datetime, timezone

    from app.models import Chunk, Paper
    from app.store import get_store

    store = get_store()
    old = Paper(id="old1", title="Retrieval study old", authors=["A"], venue="Journal", year=2020, citations=100, topic="retrieval", url="https://example.invalid/old1", abstract="retrieval reranker evidence", ingested_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    new = Paper(id="new1", title="Retrieval study new", authors=["A"], venue="Journal", year=2025, citations=100, topic="retrieval", url="https://example.invalid/new1", abstract="retrieval reranker evidence", ingested_at=datetime.now(timezone.utc))
    store.upsert(old, [Chunk("old1#0", "old1", 0, "results", "retrieval reranker improves search quality")])
    store.upsert(new, [Chunk("new1#0", "new1", 0, "results", "retrieval reranker improves search quality")])
    result = run_query("retrieval reranker", {"topic": "retrieval", "since": "2024-01-01"})
    ids = {item["paper_id"] for item in result["citations"]}
    assert "new1" in ids
    assert "old1" not in ids
