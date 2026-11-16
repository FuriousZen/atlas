from app.agent.graphs import run_ingestion
from app.ingest.sources import CandidatePaper
from app.mcp.server_atlas import ask, library_status, most_damning_discoveries, new_science_today


def test_mcp_exposes_natural_language_shortcuts():
    run_ingestion([CandidatePaper(id="mcp:p1", title="A retrieval discovery", authors=["A"], venue="Journal", year=2025, citations=100, topic="retrieval", url="https://example.invalid/p1", abstract="A result about retrieval.")], "retrieval", 50)
    assert "model" in ask("new science today?", topic="retrieval")
    assert "model" in new_science_today(topic="retrieval")
    assert "model" in most_damning_discoveries(topic="retrieval")
    assert library_status()["store"]["papers"] == 1
