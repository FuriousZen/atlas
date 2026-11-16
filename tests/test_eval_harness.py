import pytest

import app.eval.harness as harness
from app.eval.harness import hit_at_k, percentile, run, summarize_metrics


def test_hit_at_k_uses_citation_order():
    assert hit_at_k(["p1", "p2"], {"p1"}, 1) is True
    assert hit_at_k(["p9", "p1"], {"p1"}, 1) is False
    assert hit_at_k(["p9", "p1"], {"p1"}, 5) is True
    assert hit_at_k([], {"p1"}, 10) is False


def test_percentile_edge_cases():
    assert percentile([], 95) == 0.0
    assert percentile([7.0], 50) == 7.0
    assert percentile([10.0, 20.0, 30.0], 50) == 20.0
    assert percentile([10.0, 20.0, 30.0], 95) == pytest.approx(29.0)


def test_summarize_metrics_hit_rates_latency_faithfulness():
    items = [
        {"id": "a", "ranked": ["p1", "p2"], "expected": ["p1"], "elapsed_ms": 10.0, "faithfulness": 1.0, "model": "m"},
        {"id": "b", "ranked": ["p9", "p2", "p3", "p4", "p5", "p6", "p7"], "expected": ["p7"], "elapsed_ms": 20.0, "faithfulness": 0.5, "model": "m"},
        {"id": "c", "ranked": ["p9"], "expected": ["p1"], "elapsed_ms": 30.0, "faithfulness": 0.0, "model": "m"},
    ]
    out = summarize_metrics(items)
    assert out["questions"] == 3
    assert out["hit_at_1"] == pytest.approx(1 / 3)
    assert out["hit_at_5"] == pytest.approx(1 / 3)
    assert out["hit_at_10"] == pytest.approx(2 / 3)
    assert out["p50_ms"] == pytest.approx(20.0)
    assert out["p95_ms"] == pytest.approx(29.0)
    assert out["faithfulness_mean"] == pytest.approx(0.5)
    assert out["provider"] == "m"


def _fake_run_query(question, filters=None):
    return {
        "answer": "extractive answer [seed:reranking:0#0]",
        "citations": [{"paper_id": "seed:reranking:0", "chunk_id": "seed:reranking:0#0"}],
        "faithfulness": 1.0,
        "model": "unit-test-model",
        "timings_ms": {"synthesize": 1},
    }


def test_run_provider_comes_from_query_response(monkeypatch):
    monkeypatch.setattr(harness, "run_query", _fake_run_query)
    dataset = [{"id": "q", "question": "anything", "filters": {}, "gold_paper_ids": ["seed:reranking:0"]}]
    out = run(dataset)
    assert out["provider"] == "unit-test-model"
    assert out["hit_at_1"] == 1.0
    assert out["rows"][0]["model"] == "unit-test-model"


def test_run_concurrent_matches_sequential(monkeypatch):
    monkeypatch.setattr(harness, "run_query", _fake_run_query)
    dataset = [
        {"id": f"q{i}", "question": "anything", "filters": {}, "gold_paper_ids": ["seed:reranking:0"]} for i in range(3)
    ]
    out = run(dataset, concurrency=2)
    assert out["questions"] == 3
    assert out["provider"] == "unit-test-model"
    assert out["hit_at_5"] == 1.0
    assert len(out["per_query_ms"]) == 3
