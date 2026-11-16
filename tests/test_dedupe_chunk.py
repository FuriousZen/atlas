import sys

from app.config import Settings
from app.ingest.chunk import _sentence_splitter, chunk_paper
from app.ingest.dedupe import base_arxiv_id, deduplicate
from app.ingest.sources import CandidatePaper
from app.models import Paper
from app.tools.embedder import Embedder, embedding_key


def candidate(identifier: str, title: str, citations: int = 100, arxiv_id: str | None = None) -> CandidatePaper:
    return CandidatePaper(id=identifier, title=title, authors=["A"], venue="Journal", year=2025, citations=citations, topic="retrieval", url=f"https://example.invalid/{identifier}", abstract="retrieval evidence", arxiv_id=arxiv_id)


def test_version_strip_keeps_trailing_v2_but_not_inner_v():
    assert base_arxiv_id("2101.12345v2") == "2101.12345"
    assert base_arxiv_id("2101.12345") == "2101.12345"
    assert base_arxiv_id("evolving-systems") == "evolving-systems"


def test_arxiv_versions_merge_but_inner_v_titles_do_not():
    kept, duplicates = deduplicate([candidate("a", "First paper on reranking methods", arxiv_id="2101.12345v1"), candidate("b", "First paper on reranking methods", arxiv_id="2101.12345v2")])
    assert [paper.id for paper in kept] == ["a"]
    assert duplicates[0]["reason"] == "doi_or_arxiv_version"
    kept, duplicates = deduplicate([candidate("a", "Evolving-systems survey of retrieval methods"), candidate("b", "Quantum annealing review of optimization")])
    assert [paper.id for paper in kept] == ["a", "b"]
    assert duplicates == []


def test_near_dupe_band_keeps_both_and_records_pair():
    title_a = "deep neural reranking for dense passage retrieval with hard negatives"
    title_b = "deep neural reranking for dense passage retrieval with hard mining"
    kept, duplicates = deduplicate([candidate("a", title_a), candidate("b", title_b)])
    assert [paper.id for paper in kept] == ["a", "b"]
    assert len(duplicates) == 1
    assert duplicates[0]["reason"] == "possibleDuplicateOf"
    assert duplicates[0]["kept_id"] == "a"
    assert duplicates[0]["duplicate_id"] == "b"


def test_chunk_falls_back_without_llama_index(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_index", None)
    monkeypatch.setitem(sys.modules, "llama_index.core", None)
    monkeypatch.setitem(sys.modules, "llama_index.core.node_parser", None)
    assert _sentence_splitter() is None
    paper = Paper(id="p1", title="Chunk fallback", authors=["A"], venue=None, year=2025, citations=10, topic="retrieval", url="https://example.invalid/p1", abstract="word " * 600)
    chunks = chunk_paper(paper)
    assert len(chunks) > 1
    assert chunks[0].id == "p1#chunk-0000"
    assert [chunk.n for chunk in chunks] == list(range(len(chunks)))


def test_embedder_caches_by_embedding_key():
    embedder = Embedder()
    first = embedder.encode(["hello world"])
    second = embedder.encode(["hello world"])
    assert first == second
    assert first[0] is second[0]
    assert embedding_key("hello world", embedder.model_id) in embedder._fake_cache


def test_local_models_opt_in_defaults_off():
    assert Settings(_env_file=None).atlas_use_local_models is False
    assert Embedder().use_local_model is False


def test_run_ingestion_survives_unreachable_postgres(monkeypatch):
    from app.agent.graphs import run_ingestion
    from app.config import get_settings
    from app.store import InMemoryStore

    monkeypatch.setenv("DATABASE_URL", "postgresql://atlas:atlas@127.0.0.1:1/atlas")
    get_settings.cache_clear()
    try:
        store = InMemoryStore(Settings(_env_file=None))
        result = run_ingestion([candidate("high", "A paper about retrieval and evidence", citations=100)], "retrieval", 50, store=store)
        assert result["indexed"] == 1
        assert store.get_paper("high") is not None
    finally:
        get_settings.cache_clear()
