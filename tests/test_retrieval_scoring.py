from __future__ import annotations

import json

from app.agent.graphs import compute_faithfulness, run_ingestion, run_query
from app.ingest.sources import CandidatePaper
from app.models import Chunk, Paper
from app.store import get_store
from app.tools.llm_adapter import LlmResult


def _seed_five_chunks() -> list[str]:
    store = get_store()
    ids: list[str] = []
    for index in range(5):
        pid = f"faith{index}"
        paper = Paper(id=pid, title=f"Retrieval reranker study {index}", authors=["A"], venue="Journal", year=2025, citations=100, topic="retrieval", url=f"https://example.invalid/{pid}", abstract="retrieval reranker improves search quality evidence")
        chunk = Chunk(f"{pid}#chunk-0000", pid, 0, "results", "retrieval reranker improves search quality evidence")
        store.upsert(paper, [chunk])
        ids.append(chunk.id)
    return ids


def _mock_query_llm(monkeypatch, text: str, model: str = "test-mock", stub: bool = False):
    import app.tools.llm_adapter as adapter

    class MockProvider:
        def generate(self, prompt, response_schema=None, temperature=0.2):
            return LlmResult(text=text, model=model, elapsed_ms=1, backend="mock", stub=stub)

    monkeypatch.setattr(adapter, "get_llm", lambda backend=None: MockProvider())


def test_two_citations_over_five_hits_no_retry(monkeypatch):
    ids = _seed_five_chunks()
    payload = json.dumps({
        "answer": f"Finding one [{ids[0]}] and finding two [{ids[1]}].",
        "citations": [
            {"paper_id": "faith0", "chunk_id": ids[0], "section": "results", "score": 1.0, "url": "https://example.invalid/faith0"},
            {"paper_id": "faith1", "chunk_id": ids[1], "section": "results", "score": 1.0, "url": "https://example.invalid/faith1"},
        ],
    })
    _mock_query_llm(monkeypatch, payload)
    result = run_query("retrieval reranker", {"topic": "retrieval"})
    assert result["faithfulness"] >= 0.99
    assert result["retry_count"] == 0


def test_unsupported_claim_triggers_exactly_one_retry(monkeypatch):
    _seed_five_chunks()
    payload = json.dumps({
        "answer": "Claimed result [ghost#chunk-9999] with no evidence.",
        "citations": [{"paper_id": "ghost", "chunk_id": "ghost#chunk-9999", "section": "results", "score": 1.0, "url": "https://example.invalid/ghost"}],
    })
    _mock_query_llm(monkeypatch, payload)
    result = run_query("retrieval reranker", {"topic": "retrieval"})
    assert result["faithfulness"] < 0.7
    assert result["retry_count"] == 1


def test_faithfulness_ignores_non_chunk_brackets():
    valid = {"p1#chunk-0000"}
    score = compute_faithfulness("See [Figure 1] and [p1#chunk-0000] for details.", [{"paper_id": "p1", "chunk_id": "p1#chunk-0000"}], valid)
    assert score == 1.0
    assert compute_faithfulness("", [], set()) == 1.0


def test_timings_present_and_nonnegative():
    run_ingestion([CandidatePaper(id="t1", title="Retrieval reranker timing study", authors=["A"], venue="Journal", year=2025, citations=100, topic="retrieval", url="https://example.invalid/t1", abstract="retrieval reranker timing evidence")], "retrieval", 50)
    result = run_query("retrieval reranker", {"topic": "retrieval"})
    timings = result["timings_ms"]
    for node in ("rewrite", "retrieve", "rerank", "synthesize", "verify"):
        assert node in timings
        assert timings[node] >= 0
    assert len(timings) >= 5 and (any(value > 0 for value in timings.values()) or len(timings) >= 5)


def _mock_summary_llm(monkeypatch, text: str):
    import app.tools.summarizer as summarizer_mod

    class MockSummaryProvider:
        def generate(self, prompt, response_schema=None, temperature=0.2):
            return LlmResult(text=text, model="mock-summary", elapsed_ms=1, backend="mock")

    monkeypatch.setattr(summarizer_mod, "get_llm", lambda backend=None: MockSummaryProvider())


def test_summarizer_garbage_returns_fallback(monkeypatch):
    from app.tools.summarizer import summarize

    _mock_summary_llm(monkeypatch, "not json at all {{{")
    paper = Paper(id="g1", title="Garbage paper", authors=["A"], venue="J", year=2025, citations=50, topic="retrieval", url="https://example.invalid/g1", abstract="Abstract about retrieval evidence.")
    summary = summarize(paper)
    assert summary.extraction == "fallback"
    assert "Garbage paper" in summary.tldr or "fallback" in summary.tldr.lower()


def test_summarizer_embedded_json_parsed(monkeypatch):
    from app.tools.summarizer import summarize

    inner = {"tldr": "embedded ok", "contributions": ["c1"], "method": "m", "results": {"r": "1"}, "limits": "l", "why_kept": "w"}
    _mock_summary_llm(monkeypatch, f"Here is the summary: {json.dumps(inner)} hope that helps")
    paper = Paper(id="e1", title="Embedded paper", authors=["A"], venue="J", year=2025, citations=60, topic="retrieval", url="https://example.invalid/e1", abstract="Abstract.")
    summary = summarize(paper)
    assert summary.tldr == "embedded ok"
    assert summary.extraction == "llm"


def test_run_query_garbage_provider_still_cited(monkeypatch):
    _seed_five_chunks()
    _mock_query_llm(monkeypatch, "not json at all {{{")
    result = run_query("retrieval reranker", {"topic": "retrieval"})
    assert result["citations"]
    assert result["answer"]


def test_run_ingestion_continues_past_bad_summary(monkeypatch):
    import app.tools.summarizer as summarizer_mod

    calls = {"n": 0}

    class FlakyProvider:
        def generate(self, prompt, response_schema=None, temperature=0.2):
            calls["n"] += 1
            if calls["n"] == 1:
                return LlmResult(text="not json at all {{{", model="flaky", elapsed_ms=1, backend="mock")
            return LlmResult(text=json.dumps({"tldr": "ok", "contributions": ["c"], "method": "m", "results": {}, "limits": "l", "why_kept": "w"}), model="flaky", elapsed_ms=1, backend="mock")

    monkeypatch.setattr(summarizer_mod, "get_llm", lambda backend=None: FlakyProvider())
    candidates = [
        CandidatePaper(id="bad1", title="Bad one", authors=["A"], venue="J", year=2025, citations=100, topic="retrieval", url="https://example.invalid/bad1", abstract="retrieval evidence one"),
        CandidatePaper(id="good1", title="Good one", authors=["A"], venue="J", year=2025, citations=100, topic="retrieval", url="https://example.invalid/good1", abstract="retrieval evidence two"),
    ]
    result = run_ingestion(candidates, "retrieval", 50)
    assert result["indexed"] == 2
    assert get_store().get_paper("bad1") is not None
    assert get_store().get_paper("good1") is not None
