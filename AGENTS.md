# AGENTS.md - Atlas

Atlas is citation-filtered research intelligence: a bounded RAG service over
papers that earned attention. It polls open research sources, keeps only papers
above a per-topic citation threshold, has agents compress keepers into a dense
citable store, and serves that compression with receipts (every answer cites
chunks).

It is bounded by design. The store holds at most N papers / M chunks / B bytes
(defaults 1500 / 15000 / 2 GiB, 7-day grace). When the next poll would overflow,
the oldest documents are evicted first. Grafana makes the bound visible.

## Architecture

Ingest path (implemented as `run_ingestion()` in `app/agent/graphs.py`, not a
separate StateGraph):

```
fetch (S2 + OpenAlex + arXiv) -> threshold (citations >= X, before any LLM spend)
  -> dedupe (doi / arxiv id / fuzzy title) -> capacity_check -> evict (oldest first)
  -> summarize (agent, LangChain prompt + LLM adapter) -> chunk (word-window 256/32)
  -> embed (fake 128-d by default) -> InMemoryStore
```

Query path (LangGraph StateGraph in `app/agent/graphs.py`):

```
rewrite (keyword heuristics) -> retrieve (in-memory dense+lexical blend)
  -> rerank (lexical fallback) -> synthesize (LLM adapter) -> verify (citation overlap)
  -> retry (max 1, expanded k)
```

LLM provider is decoupled. Agent nodes depend only on the `LlmResult`
interface in `app/tools/llm_adapter.py`. `ATLAS_LLM_BACKEND=auto` picks the
first configured provider (gemini -> openai -> anthropic), else the
deterministic `atlas-stub-heuristic` provider. With no keys the API, MCP, and
tests all work on the stub. Setting `vllm` or `ollama` plus its URL selects
local serving. No graph or MCP code changes when switching providers.

## Repo layout

```
app/api/main.py       FastAPI: /health /ready /metrics /papers /brief /ask /ingest/run
app/agent/graphs.py   Both graphs + run_query + run_ingestion (single file, nodes/ is empty)
app/tools/            llm_adapter.py, summarizer.py, embedder.py, reranker.py
app/store.py          InMemoryStore: the actual runtime store (papers/chunks dicts + RLock)
app/models.py         Paper / Chunk / SearchHit / Summary dataclasses
app/ingest/           sources.py (S2/OpenAlex httpx clients), dedupe.py, chunk.py
app/vector/qdrant.py  QdrantStore wrapper (connects, but nothing calls it in the live path)
app/db/schema.sql     Postgres DDL (papers/chunks/caches/topic_state/duplicates); DDL only, unwired
app/mcp/              FastMCP servers: sources, arxiv, openalex shim, qdrant, postgres,
                      redis, fs, store, atlas. qdrant/postgres/redis servers are local-backed
                      stand-ins, not live Qdrant/SQL/Redis calls. compat.py shims FastMCP.
app/eval/harness.py   Minimal harness: JSONL -> run_query -> hit_at_5 + elapsed_ms
scripts/              seed_topics.py (synthetic ingest), backfill.py (live S2 ingest)
web/                  React Ask-only UI (src.jsx) + nginx same-origin /api + Vite, no config file
infra/k8s/atlas.yaml  Namespace + api Deployment (replicas 1) + Service + CronJob poll
grafana/              Provisioning + 2 skeleton dashboards (library, retrieval)
eval/datasets/        atlas_qa.jsonl (2 rows, seed:* synthetics)
perf/                 Empty; intended home of perf/results/eval-*.json (none committed)
tests/                test_api, test_graph, test_ingest, test_store, test_mcp + conftest reset_store
```

## Runtime reality (read before assuming docs)

Executed 2026-09-05: all 16 PLAN.md items (P0-P5) implemented, 45 tests
passing, `ruff check app tests` clean under pinned ruff 0.9.6. User decisions:
bench target deleted, grace-blocked eviction returns 429, minimal Postgres
wire-up, metrics added (not prose trimmed), full K8s objects added. What is
still true below; struck items are fixed.

- Default runtime is in-memory + stub. `Embedder` honors `ATLAS_USE_LOCAL_MODELS`
  (default false, warn-and-fallback); the in-process embedder dict cache is wired
  to `atlas_embedding_cache_hit_total{tier}`. `store.search` still uses
  `fake_embedding` unless local models are enabled.
- Postgres is best-effort wired: `app/db/session.py` (`init_db`,
  `load/save_topic_state`, `record_duplicates`) is called from `run_ingestion`
  and `scripts/backfill.py` with warn-and-continue when unreachable. Qdrant
  payload indexes (`topic/citations/ingested_at`) are created when the client
  supports them; collection `dimensions` are frozen at creation with a mismatch
  error. `server_redis.py` is still a process-local dict.
- `/ready` always returns ready; it does not ping deps despite README wording.
- `POST /ask?explain=true` now exists (adds `rewritten`; `timings_ms` always
  present). `since` is forwarded in the query retrieve path; `sort=citations|recent`
  is explicit on `/ask`, `/papers`, `/brief`, MCP `ask`. `GET /ingest/runs/{run_id}`
  reports pending/ok/error/capacity_blocked (429 + `Retry-After` when blocked);
  background failures are logged and recorded, not swallowed.
- Body-size guard is a byte-counting middleware (covers chunked) plus a guarded
  `content-length` fast path. 4xx errors use `{error, detail, request_id}` with
  `X-Request-ID` echo; 422s keep `{"detail":...}` shape plus the header.
  `Filters.section` items are capped at 64 chars; `Infinity/NaN` rejected.
- Per-node `timings_ms` are real wall-clock and `NODE_LATENCY` is observed per
  node. Ingest threshold/dedupe drops, eviction age, `llm_elapsed_ms{backend}`,
  and embedding-cache tiers are emitted; dashboards cover them (PromQL fixed to
  `sum by (le, node)`). Metric path labels collapse `/ingest/runs/<id>`.
- Faithfulness is claims-over-supported on chunk-id-shaped citations
  (`[id#chunk]`; `[Figure 1]` no longer counts). Empty answer with no claims
  scores 1.0; uncited non-empty answers score 0.0 and retry once.
- Eviction is per-dimension (papers / chunks-per-paper / bytes-per-paper, max
  wins) with citation grace (`ingested_at + min(7d, log1p(cites)*12h)`,
  oldest-first). All-young overflow raises `CapacityBlocked` (HTTP 429), never
  evicts young.
- Dedupe merges at Jaccard >= 0.82 with a 0.72-0.82 `possibleDuplicateOf` band
  (both kept, pair recorded); arxiv version strip uses a trailing-`v\d+` regex.
  Pairs flow to `record_duplicates()` best-effort. Chunking uses LlamaIndex
  `SentenceSplitter(512,64)` when importable, else the 256/32 word window.
- `summarize` never raises: balanced-brace repair (max 5 candidates), then
  extractive fallback (`extraction="fallback"`); ingestion continues past bad
  summaries and `/ask` degrades instead of 500ing.
- Eval is 8 questions with hit@1/5/10, faithfulness mean, p50/p95, and
  `--concurrent N`. Real artifact committed: `perf/results/eval-20260905.json`
  (hit 0.125 on stub - the seed corpus is near-identical abstracts, a corpus
  finding not a harness bug). Run `make seed` before `make eval` (documented).
  CI exists (`.github/workflows/ci.yml`: ruff, pytest, eval-smoke on py3.11).
  Web deps are pinned with `package-lock.json`, `npm ci`, `vite.config.js`
  (react plugin + `/api` proxy), full `index.html`, and Brief/Manage tabs.
  K8s has replicas 2 + HPA + Ingress (regex rewrite to strip `/api`) + web
  Deployment/Service + NetworkPolicies (api egress includes 80/443) + PDB +
  commented Secret template. `.env.example` carries `ATLAS_POSTGRES_PASSWORD`.
- `PLAN.md` is the executed enhancement plan with an execution record header.

## How to run

```bash
cp .env.example .env   # all provider keys may remain empty
docker compose up -d --build
curl -s http://127.0.0.1:8000/health | python -m json.tool
curl -s 'http://127.0.0.1:8000/brief?topic=reranking&since=2025-10-01&min_citations=80' | python -m json.tool
curl -s -X POST http://127.0.0.1:8000/ask -H 'content-type: application/json' \
  -d '{"question":"new science today?"}' | python -m json.tool
```

Offline (synthetic corpus, no keys):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu || true
pip install -e .[dev]
pytest -q
python scripts/seed_topics.py --topics reranking:80,retrieval:50 --papers 30
python -m app.eval.harness --dataset eval/datasets/atlas_qa.jsonl --out perf/results/eval-$(date +%Y%m%d).json
```

Tests: `pytest -q` (offline, seeded, no keys). `Makefile` targets: `test`,
`lint` (`ruff check app tests`), `eval`, `seed`, `backfill`, `serve`, `web`.
`bench` is broken (see above). Env: `OMP_NUM_THREADS=4` for local runs.

## Conventions for agents working here

- Request models use `extra="forbid"` with length/range caps. Keep that on any
  new endpoint or field. Cap per-item list element length too, not just list
  length.
- Threshold before LLM spend is load-bearing and tested. Never summarize
  before filtering on citations.
- `InMemoryStore` is the live path. If you wire Postgres/Qdrant/Redis, route
  all three (plus filesystem PDF cache) through one capacity/evict unit, or do
  not wire any. Three divergent delete paths already exist; do not add a fourth.
- Do not add dependencies, upgrade versions, or reformat whole files. Frontend
  versions need pinning plus a lockfile, but that is a deliberate one-time fix,
  not an upgrade sweep.
- Perf artifacts under `perf/results/` are point-in-time truth when present.
  Do not regenerate with today's date to fake freshness. The stale-spec banner
  and metric prose in docs should be fixed by editing wording, not by inventing
  measurements.
- Verify by running `pytest -q` and `ruff check app tests` before claiming
  completeness. Swift-style toolchains do not apply here; if Docker/K8s or live
  Qdrant/Postgres/LLM keys are unavailable, say the check could not run rather
  than implying it passed.
