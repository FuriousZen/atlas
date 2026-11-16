from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.ingest.chunk import chunk_paper
from app.ingest.dedupe import deduplicate
from app.ingest.sources import CandidatePaper
from app.models import Paper
from app.store import InMemoryStore, get_store
from app.tools.embedder import Embedder
from app.tools.reranker import Reranker
from app.tools.summarizer import fallback_summary, summarize


_CHUNK_CLAIM_RE = re.compile(r"\[([^\[\]]*#[^\[\]]+)\]")


def _observe_node_latency(node: str, seconds: float) -> None:
    try:
        from app.metrics import NODE_LATENCY
    except Exception:
        return
    try:
        NODE_LATENCY.labels(node).observe(seconds)
    except Exception:
        return


def _observe_llm_elapsed(backend: str, elapsed_ms: int) -> None:
    try:
        from app.metrics import LLM_ELAPSED_MS
    except Exception:
        return
    try:
        LLM_ELAPSED_MS.labels(backend).observe(max(0, elapsed_ms))
    except Exception:
        return


def _observe_ingest_drops(topic: str, dropped_threshold: int, dedupe_drops: int) -> None:
    try:
        from app.metrics import DROPPED_DEDUPE, DROPPED_THRESHOLD
    except Exception:
        return
    try:
        if dropped_threshold > 0:
            DROPPED_THRESHOLD.labels(topic).inc(dropped_threshold)
        if dedupe_drops > 0:
            DROPPED_DEDUPE.labels(topic).inc(dedupe_drops)
    except Exception:
        return


def _observe_eviction_ages(ages_days: list[float]) -> None:
    try:
        from app.metrics import EVICTION_AGE_DAYS
    except Exception:
        return
    try:
        for age in ages_days:
            EVICTION_AGE_DAYS.observe(max(0.0, age))
    except Exception:
        return


def _claim_ids(answer: str, citations: list[Any]) -> set[str]:
    cited = {str(item.get("chunk_id", "")).strip() for item in citations if isinstance(item, dict) and str(item.get("chunk_id", "")).strip()}
    cited.discard("")
    bracketed = {match.strip() for match in _CHUNK_CLAIM_RE.findall(answer or "") if match.strip()}
    return cited | bracketed


def compute_faithfulness(answer: str, citations: list[Any], valid_ids: set[str]) -> float:
    claimed = _claim_ids(answer, citations)
    if not claimed:
        return 1.0 if not (answer or "").strip() else 0.0
    supported = claimed & set(valid_ids)
    return min(1.0, len(supported) / max(1, len(claimed)))


class QueryState(TypedDict, total=False):
    question: str
    filters: dict[str, Any]
    rewritten: list[str]
    hits: list[Any]
    answer: str
    citations: list[dict[str, Any]]
    faithfulness: float
    timings_ms: dict[str, int]
    retry_count: int
    model: str
    stub: bool


def _parse_since(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def rewrite_question(question: str, filters: dict[str, Any]) -> list[str]:
    """Turn natural prompts into retrieval intent without requiring an LLM."""
    lower = question.lower().strip()
    variants = [question]
    if "today" in lower or "new science" in lower or "recent" in lower:
        variants.extend(["recently published high-citation scientific research", "new research published today"])
        filters.setdefault("since", datetime.now(timezone.utc).date().isoformat())
    if "damning" in lower or "negative" in lower or "controvers" in lower or "failed" in lower:
        variants.extend(["scientific papers with negative results limitations failures or retractions", "high-impact disputed scientific findings"])
        filters.setdefault("section", ["limits", "results"])
    return list(dict.fromkeys(variants))[:4]


def _prompt(state: QueryState) -> str:
    evidence = "\n".join(
        f"[{hit.chunk.id}] {hit.paper.title} ({hit.paper.citations} citations): {hit.chunk.text[:1200]}"
        for hit in state.get("hits", [])
    )
    return (
        "Answer only from the evidence below. Every factual claim must cite a chunk id in square brackets. "
        "If evidence is insufficient, say that explicitly.\n"
        f"Question: {state['question']}\nEvidence:\n{evidence}"
    )


def _parse_answer(text: str, hits: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    try:
        data = json.loads(text)
        answer = str(data.get("answer", ""))
        citations = data.get("citations", [])
        if answer:
            return answer, citations if isinstance(citations, list) else []
    except (TypeError, ValueError):
        pass
    citations = [
        {"paper_id": hit.paper.id, "chunk_id": hit.chunk.id, "section": hit.chunk.section, "score": hit.score, "url": hit.paper.url}
        for hit in hits
    ]
    return text.strip(), citations


def build_query_graph(store: InMemoryStore | None = None, embedder: Embedder | None = None, reranker: Reranker | None = None):
    store = store or get_store()
    embedder = embedder or Embedder()
    reranker = reranker or Reranker()

    def rewrite(state: QueryState) -> dict[str, Any]:
        t0 = time.perf_counter()
        filters = dict(state.get("filters", {}))
        out: dict[str, Any] = {"filters": filters, "rewritten": rewrite_question(state["question"], filters)}
        elapsed = time.perf_counter() - t0
        out["timings_ms"] = {"rewrite": int(elapsed * 1000)}
        _observe_node_latency("rewrite", elapsed)
        return out

    def retrieve(state: QueryState) -> dict[str, Any]:
        t0 = time.perf_counter()
        filters = state.get("filters", {})
        since = _parse_since(filters.get("since"))
        sort = str(filters.get("sort", "citations"))
        if sort not in ("citations", "recent"):
            sort = "citations"
        hits: list[Any] = []
        for query in state.get("rewritten", [state["question"]]):
            hits.extend(store.search(query, topic=filters.get("topic"), min_citations=filters.get("min_citations"), since=since, section=filters.get("section"), limit=12, sort=sort))
        unique = {hit.chunk.id: hit for hit in hits}
        elapsed = time.perf_counter() - t0
        _observe_node_latency("retrieve", elapsed)
        return {"hits": list(unique.values()), "timings_ms": {**state.get("timings_ms", {}), "retrieve": int(elapsed * 1000)}}

    def rerank(state: QueryState) -> dict[str, Any]:
        t0 = time.perf_counter()
        ranked = reranker.rerank(state["question"], state.get("hits", []), limit=5)
        elapsed = time.perf_counter() - t0
        _observe_node_latency("rerank", elapsed)
        return {"hits": ranked, "timings_ms": {**state.get("timings_ms", {}), "rerank": int(elapsed * 1000)}}

    def synthesize(state: QueryState) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            from app.tools.llm_adapter import get_llm

            result = get_llm().generate(_prompt(state), {"type": "object", "properties": {"answer": {"type": "string"}, "citations": {"type": "array"}}}, temperature=0.2)
            _observe_llm_elapsed(result.backend, result.elapsed_ms)
            answer, citations = _parse_answer(result.text, state.get("hits", []))
            if result.stub and not citations:
                citations = [
                    {"paper_id": hit.paper.id, "chunk_id": hit.chunk.id, "section": hit.chunk.section, "score": hit.score, "url": hit.paper.url}
                    for hit in state.get("hits", [])
                ]
            out = {"answer": answer, "citations": citations, "model": result.model, "stub": result.stub}
        except Exception:
            logger.exception("synthesize provider failed; returning degraded cited answer")
            hits = state.get("hits", [])
            _observe_llm_elapsed("fallback", int((time.perf_counter() - t0) * 1000))
            top = f" [{hits[0].chunk.id}]" if hits else ""
            out = {
                "answer": f"Degraded answer for '{state.get('question', '')[:200]}': provider failed, showing strongest retained evidence.{top}",
                "citations": [
                    {"paper_id": hit.paper.id, "chunk_id": hit.chunk.id, "section": hit.chunk.section, "score": hit.score, "url": hit.paper.url}
                    for hit in hits
                ],
                "model": "fallback-degraded",
                "stub": True,
            }
        elapsed = time.perf_counter() - t0
        _observe_node_latency("synthesize", elapsed)
        return {**out, "timings_ms": {**state.get("timings_ms", {}), "synthesize": int(elapsed * 1000)}}

    def verify(state: QueryState) -> dict[str, Any]:
        t0 = time.perf_counter()
        answer = state.get("answer", "")
        valid_ids = {hit.chunk.id for hit in state.get("hits", [])}
        faithfulness = compute_faithfulness(answer, state.get("citations", []), valid_ids)
        elapsed = time.perf_counter() - t0
        _observe_node_latency("verify", elapsed)
        return {"faithfulness": faithfulness, "timings_ms": {**state.get("timings_ms", {}), "verify": int(elapsed * 1000)}}

    def route(state: QueryState) -> str:
        if state.get("faithfulness", 0.0) < 0.7 and state.get("retry_count", 0) < 1:
            return "retry"
        return "done"

    def retry(state: QueryState) -> dict[str, Any]:
        return {"retry_count": state.get("retry_count", 0) + 1, "rewritten": state.get("rewritten", []) + [state["question"] + " evidence limitations results"]}

    graph = StateGraph(QueryState)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("synthesize", synthesize)
    graph.add_node("verify", verify)
    graph.add_node("retry", retry)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "synthesize")
    graph.add_edge("synthesize", "verify")
    graph.add_conditional_edges("verify", route, {"retry": "retry", "done": END})
    graph.add_edge("retry", "retrieve")
    return graph.compile()


def run_query(question: str, filters: dict[str, Any] | None = None, store: InMemoryStore | None = None, sort: str = "citations", explain: bool = False) -> dict[str, Any]:
    merged = dict(filters or {})
    if sort != "citations":
        merged["sort"] = sort
    else:
        merged.setdefault("sort", "citations")
    if merged.get("sort") not in ("citations", "recent"):
        merged["sort"] = "citations"
    result = build_query_graph(store).invoke({"question": question, "filters": merged, "retry_count": 0})
    if explain:
        return {key: value for key, value in result.items() if key != "hits"}
    return {key: value for key, value in result.items() if key not in {"hits", "rewritten"}}


logger = logging.getLogger(__name__)


def _persist_ingest_state_best_effort(topic: str, min_citations: int, duplicates: list[dict[str, object]]) -> None:
    try:
        from app.db.session import get_conn, load_topic_state, record_duplicates, save_topic_state

        conn = get_conn()
        try:
            load_topic_state(conn)
            save_topic_state(conn, topic, min_citations, None)
            record_duplicates(conn, duplicates)
        finally:
            try:
                conn.close()
            except Exception:
                logger.debug("postgres conn close failed", exc_info=True)
    except Exception:
        logger.warning("postgres unavailable; continuing in-memory", exc_info=True)


class IngestState(TypedDict, total=False):
    topic: str
    min_citations: int
    candidates: list[CandidatePaper]
    kept: list[CandidatePaper]
    duplicates: list[dict[str, object]]
    evicted: list[str]
    indexed: int


def run_ingestion(candidates: list[CandidatePaper], topic: str, min_citations: int, store: InMemoryStore | None = None) -> dict[str, Any]:
    """Run threshold, dedupe, capacity, summarization, and indexing in order.

    CapacityBlocked from enforce_capacity() propagates to the caller (HTTP 429).
    Postgres topic_state/duplicates wiring is best-effort: when DATABASE_URL is
    unreachable a warning is logged and the in-memory path continues.
    """
    store = store or get_store()
    kept_candidates = [paper for paper in candidates if paper.citations >= min_citations]
    dropped_threshold = len(candidates) - len(kept_candidates)
    kept_candidates, duplicates = deduplicate(kept_candidates)
    # "possibleDuplicateOf" pairs keep both papers; only merged rows are drops.
    dedupe_drops = sum(1 for item in duplicates if item.get("reason") != "possibleDuplicateOf")
    _observe_ingest_drops(topic, dropped_threshold, dedupe_drops)
    _persist_ingest_state_best_effort(topic, min_citations, duplicates)
    incoming_chunks = max(1, len(kept_candidates) * 6)
    incoming_bytes = sum(len(f"{paper.title} {paper.abstract}".encode()) for paper in kept_candidates) + incoming_chunks * 256
    try:
        now = datetime.now(timezone.utc)
        ages_by_id = {pid: (now - paper.ingested_at).total_seconds() / 86400.0 for pid, paper in store.papers.items()}
    except Exception:
        ages_by_id = {}
    # Let CapacityBlocked propagate; background tasks record it in the registry.
    evicted = store.enforce_capacity(incoming_papers=len(kept_candidates), incoming_chunks=incoming_chunks, incoming_bytes=incoming_bytes)
    _observe_eviction_ages([ages_by_id[pid] for pid in evicted if pid in ages_by_id])
    indexed = 0
    for candidate in kept_candidates:
        if store.get_paper(candidate.id):
            continue
        paper = Paper(id=candidate.id, title=candidate.title, authors=candidate.authors, venue=candidate.venue, year=candidate.year, citations=candidate.citations, topic=topic, url=candidate.url, abstract=candidate.abstract, doi=candidate.doi, arxiv_id=candidate.arxiv_id, pdf_url=candidate.pdf_url, why_kept=f"{candidate.citations} citations >= {min_citations} threshold", extraction="abstract_only")
        try:
            summary = summarize(paper)
        except Exception:
            logger.exception("summarize failed for %s; using fallback summary", candidate.id)
            summary = fallback_summary(paper)
        paper.summary = summary.to_dict()
        paper.summarized_at = datetime.now(timezone.utc)
        paper.why_kept = summary.why_kept or paper.why_kept
        chunks = chunk_paper(paper)
        store.upsert(paper, chunks)
        indexed += 1
    return {"topic": topic, "polled": len(candidates), "kept": len(kept_candidates), "indexed": indexed, "dropped_threshold": dropped_threshold, "duplicates": duplicates, "evicted": evicted, "counts": store.counts()}
