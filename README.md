# Atlas — citation-filtered research intelligence (bounded RAG)

Atlas polls open research sources, keeps only papers that have *earned* attention (`citationCount >= X` per topic), has agents compress them into a dense, citable store, and serves that compression as a RAG service that answers with receipts.

It is bounded by design: the store holds at most `N` papers / `M` chunks / `B` bytes. When the next poll would overflow, the oldest documents are evicted first — a rolling window, not an ever-growing dump. Grafana makes the bound visible.

This is the third pillar alongside `predexec` (agent/MCP) and `Midwest Solar` (classical ML) — the agentic RAG pillar that finally exercises `LangChain`, `LangGraph`, `LlamaIndex`, `FastMCP`, `Qdrant`, `Redis`, `PyTorch`+`Hugging Face`, `vLLM`/`Ollama`, and `Grafana` end-to-end.

## Quick start

```bash
cp .env.example .env   # all provider keys may remain empty
docker compose up -d --build
# wait for /ready then
curl -s http://127.0.0.1:8000/health | python -m json.tool
curl -s 'http://127.0.0.1:8000/brief?topic=reranking&since=2025-10-01&min_citations=80' | python -m json.tool
curl -s -X POST http://127.0.0.1:8000/ask -H 'content-type: application/json' \
  -d '{"question":"new science today?"}' | python -m json.tool
open http://127.0.0.1:8080   # web
open http://127.0.0.1:3000   # grafana admin/atlas
```

Without Docker (offline, synthetic corpus, no keys):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu || true
pip install -e .[dev]
pytest -q
# seed first: the eval dataset's gold ids are seed:* synthetics, so seed before eval
python scripts/seed_topics.py --topics reranking:80,retrieval:50 --papers 30  # --papers is per topic
python -m app.eval.harness --dataset eval/datasets/atlas_qa.jsonl --out perf/results/eval-$(date +%Y%m%d).json
```

## Architecture

See `PLAN.md` for the full spec (two LangGraph graphs, 7 MCP servers, capacity→evict transaction, two Grafana dashboards).

*   **Ingest graph:** `fetch(S2+arXiv+OpenAlex via FastMCP) → threshold(citations>=X) → dedupe(doi/arxiv_id+fuzzy) → capacity_check → evict(oldest) → summarize(agent, LangChain) → chunk(LlamaIndex 512/64) → embed(bge-m3, PyTorch/HF) → Qdrant+Postgres` — threshold before any LLM spend.
*   **Query graph:** `rewrite(HyDE, 3 paraphrases) → retrieve(Qdrant dense+Postgres FTS, RRF) → rerank(bge-reranker-v2-m3) → synthesize(Gemini/vLLM/Ollama) → verify(faithfulness) → (retry)` — every sentence cited.
*   **MCP servers:** `s2`, `arxiv`, `openalex`, `qdrant`, `postgres`, `redis`, `fs` via `FastMCP`. Adding a source is a new server, not an agent edit.
*   **Bound:** `MAX_PAPERS=1500 / MAX_CHUNKS=15000 / MAX_BYTES=2GiB`, `MIN_KEEP_DAYS=7`, eviction score `ingested_at - log1p(citations)*12h` (bounded to +7d so the window always rolls). See `app/config.py`.

## MCP-deployed RAG — steering prompts

Once the pipeline is up (real or seeded), any agent hooked to the MCP `ask` tool steers naturally:

*   `new science today?` → `rewrite` expands to `recent high-citation research today + since: today + min_citations: topic default` → `brief`-style retrieval over `ingested_at >= today`.
*   `most damning scientific discoveries published?` → `rewrite` extracts `controversial/high-impact, high citations, limits/failures` → `rerank` surfaces papers whose `limits`/`results` chunks discuss negative or retracted claims, still filtered by citations, still cited.

The prompt steering is in the query graph's `rewrite` node (HyDE + paraphrase), not a fat system prompt — see `app/agent/graphs.py:rewrite_question`.

The LLM provider is not part of the agents API. Agent nodes depend on the small
`LlmResult` interface in `app/tools/llm_adapter.py`, and the adapter resolves
`ATLAS_LLM_BACKEND=auto` to the first configured provider. With no key, it uses
the deterministic `atlas-stub-heuristic` provider, so `/ask`, `/brief`, MCP,
and tests remain usable. Add `GEMINI_API_KEY`, `OPENAI_API_KEY`, or
`ANTHROPIC_API_KEY` to `.env` and restart the API. For local serving, explicitly
set `ATLAS_LLM_BACKEND=vllm` or `ollama` and its URL. No graph or MCP code changes.

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | liveness, no deps, echoes `MAX_PAPERS` + `ATLAS_EVICTION` |
| `GET` | `/ready` | pings Qdrant/Postgres/Redis/LLM adapter; 503 if any down |
| `GET` | `/brief?topic=&since=&min_citations=` | polling product: polled/kept/dropped_threshold/evicted + papers with summaries |
| `POST` | `/ask` | `{question, filters?: {topic, min_citations, since}}` → `{answer, citations, timings_ms, model, faithfulness}`; `?explain=true` adds per-node timings. Example: `POST /ask {"question":"new science today?"}` |
| `GET` | `/papers?topic=&min_citations=&since=` | list, paginated |
| `POST` | `/ingest/run` | on-demand poll `{topic, since?, min_citations?}` → `202 {run_id}` |
| `GET` | `/metrics` | Prometheus exposition |
| `GET` | `/docs` | OpenAPI |

Every field bounded, `extra="forbid"`, body 4 MiB via ASGI middleware — same hardening as `midwest/src/msa/serve/app.py`.

## Grafana

Two provisioned dashboards (datasource Prometheus + Postgres).

*   **Library health:** `papers polled vs kept vs dropped (threshold/dedupe)` per topic, `evicted`, `store_papers/chunks/bytes` vs `MAX_*` bounds, `eviction_age_days` p95/mean.
*   **Retrieval quality:** `p95` per agent node, HTTP requests by path/status, `llm_elapsed_ms{backend}` p50/p95, embedding cache hits by tier. `hit@k` and `faithfulness` are not Prometheus series; they live in the eval artifact below.

Scrape: `prometheus.yml` → `api:8000/metrics`. Commit artifact: `perf/results/eval-*.json` is the point-in-time truth; Grafana is the live view.

## Configuration

All in `.env` → `app/config.py`. Key knobs: `TOPICS` (e.g., `reranking:80,retrieval:50`), `ATLAS_MAX_PAPERS/CHUNKS/BYTES_MB`, `ATLAS_MIN_KEEP_DAYS`, `ATLAS_EVICTION`, `ATLAS_EMBED_MODEL`, `ATLAS_RERANK_MODEL`, `ATLAS_LLM_BACKEND` (`auto|stub|none|gemini|openai|anthropic|vllm|ollama`), provider keys and model names, `QDRANT_URL`, `DATABASE_URL`, `REDIS_URL`, `S2_API_KEY`.

## Tests

```bash
pytest -q                         # offline, seeded, no keys
pytest --run-live -k test_evict   # hits real Qdrant/Postgres if up
```

## Tech stack

Python, FastAPI, LangChain, LangGraph, LlamaIndex, FastMCP, Qdrant, PostgreSQL, Redis, PyTorch, Hugging Face (`bge-m3` + `bge-reranker-v2-m3`), Gemini / vLLM / Ollama, Docker Compose, Kubernetes, Grafana, Prometheus.

See `PLAN.md` for the full build spec and resume mapping.
