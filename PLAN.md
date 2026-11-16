# PLAN.md - Atlas next steps

> **Execution record (2026-09-05): all 16 items implemented in one orchestration
> pass (sequential implementer subagents P0 -> P1 -> P2 -> P3 -> P4 -> P5, final
> review, one fix wave). 45 tests passing, `ruff check app tests` clean under
> pinned ruff 0.9.6. User decisions: bench target deleted (not implemented),
> grace-blocked eviction returns 429, minimal Postgres wire-up (not dep
> demotion), metrics added (not prose trimmed), full K8s objects added (not doc
> narrowing). Final review caught 4 Important issues, all fixed: eviction
> citation-bonus sign, metric path cardinality (`/ingest/runs/:id`), Ingress
> `/api` rewrite (+ `use-regex`), api NetworkPolicy egress 80/443. Real eval
> artifact: `perf/results/eval-20260905.json`. Known follow-ups: CI eval-smoke
> scores 0.0 (seed/harness run in separate processes, in-memory store) -
> plumbing only; no live Grafana/k8s/docker verification in this environment;
> `_runs` dict and `store.evictions` unbounded (minor). Items below are the
> original specs, kept for history.**

This replaces the prior 606-line build spec (git history retains it). Code
exists now; the old "not yet implemented" banner is stale. What follows is the
enhancement plan from a three-way parallel audit (API+agents, storage+ingest,
web+infra) conducted 2026-09-04. Items are ordered by severity, each with
owner files and verification.

Current state in one paragraph: FastAPI + LangGraph + in-memory store works
offline with the stub LLM. Threshold-before-spend, dedupe, grace eviction,
`/ask` with citations, `/brief`, `/papers`, `/metrics`, MCP shims, seed and
backfill scripts, compose topology, and skeleton Grafana all exist. Postgres,
Qdrant, and Redis are provisioned but not on the live path. Eval is minimal,
`make bench` is broken, one Dockerfile line is invalid, and several README
claims overstate the runtime (see AGENTS.md "Runtime reality").

## P0 - boot and build blockers (half day total)

1. `Dockerfile:19` invalid `COPY` shell syntax.
   `COPY alembic.ini ./alembic.ini 2>/dev/null || true` is not valid Dockerfile.
   There is no `alembic.ini` in the repo. Delete the line.
   Verify: `docker compose build api`.
2. Grafana Postgres auth gap.
   `grafana/provisioning/datasources/datasource.yml` needs
   `${ATLAS_POSTGRES_PASSWORD}`, absent from `.env.example`, while compose
   hardcodes `POSTGRES_PASSWORD=atlas`. Add `ATLAS_POSTGRES_PASSWORD=atlas` to
   `.env.example`, thread compose `POSTGRES_PASSWORD:
   ${ATLAS_POSTGRES_PASSWORD:-atlas}`, and keep the datasource var.
   Verify: fresh `cp .env.example .env && docker compose up -d`, Grafana
   Postgres datasource test passes.
3. `make bench` crashes.
   `Makefile:33` passes `--bench --concurrency 8 --requests 200`, which
   `app/eval/harness.py` never defined. Either implement a concurrent sweep in
   the harness (asyncio gather, report p50/p95) or delete the target. Do not
   leave a crashing default target.
   Verify: `make bench` either runs or no longer exists (`make -n bench`).

## P1 - API contract gaps (1-2 days)

4. `since` dropped in query path.
   `app/agent/graphs.py` `retrieve()` never forwards `filters["since"]` to
   `store.search()`; the `sort=citations` hint from `rewrite_question` is also
   unconsumed, and `store.list_papers` always sorts `(citations,
   ingested_at)`. Parse `since` to datetime once at the API boundary, forward
   it, and add an explicit `sort` param instead of hidden hints.
   Verify: test ingesting old + new papers, `run_query` with `since=today`
   returns only new.
5. `POST /ask?explain=true` missing; no ingest status.
   README promises `?explain=true` per-node timings; `POST /ingest/run`
   returns `{run_id}` with no lookup, and `app/api/main.py:_ingest` swallows
   exceptions. Add `?explain` to include `rewritten` + timings, and a small
   in-memory run registry (`GET /ingest/runs/{run_id}` with pending/ok/error)
   that `_ingest` updates instead of `except Exception: return`. Log OpenAlex
   enrichment failures instead of `pass`.
   Verify: tests for explain shape, 202 shape, and failed-ingest visibility.
6. Request hardening to match claims.
   Replace the `content-length`-only guard (`main.py:66-69`) with a
   byte-counting ASGI middleware (covers `Transfer-Encoding: chunked`), guard
   the `int()` parse (malformed header currently 500s), adopt `{error, detail,
   request_id}` + `X-Request-ID` echo, cap `Filters.section` items per-element
   (today only list length is capped), and reject `Infinity/NaN`.
   Verify: add `tests/test_serve_hardening.py` (oversize 413, chunked oversize,
   bad `since` 422, Infinity 422, envelope shape).

## P2 - storage bound correctness (2-3 days)

7. Fix `app/store.py:107-118` eviction math.
   Byte overflow divided by full `max_bytes` yields 0/1 and never wins `max()`;
   chunk overflow evicts `N_chunks` papers (about 6x over-eviction at ~6
   chunks/paper); grace-blocked case evicts young instead of the spec 429.
   Convert bytes to chunk/paper equivalents or run separate victim loops per
   dimension, and either return blocked (429 + `Retry-After`) or document the
   relaxation. One capacity/evict unit must own Postgres + Qdrant + FS + Redis;
   today three delete paths diverge (`InMemoryStore`, `QdrantStore.delete`,
   MCP `delete`) and `run_ingestion` calls only the first.
   Verify: `MAX_PAPERS=3` seed-5-ingest-2 evicts oldest; byte-pressure-only and
   all-young-grace tests.
8. Wire the durable layer or cut dependency weight.
   `psycopg`, `SQLAlchemy`, `alembic`, `redis`, `llama-index` are pinned but
   never imported on the live path; `torch`/`transformers` are lazy opt-in
   only. Cheapest credible step: `app/db/session.py` + `init_db(schema.sql)` +
   `topic_state` cursor persistence + `paper_duplicates` + embedding cache via
   direct psycopg, call `embedding_key()`/`get`/`put` in `Embedder.encode`,
   add Qdrant payload indexes (`topic/citations/ingested_at`). If that is out
   of scope, demote the unused packages to extras so install does not imply
   coverage that does not exist.
   Verify: restart keeps cursors; Qdrant point count matches store after evict.
9. Dedupe, chunk, embed spec deltas.
   Dedupe: fix `arxiv_id.split("v")[0]` (splits any `v`; use version regex),
   add the 0.72-0.82 `possibleDuplicateOf` band, persist pairs to
   `paper_duplicates`. Chunk: use LlamaIndex `SentenceSplitter(512,64)` when
   installed, current 256/32 splitter as fallback. Embed: expose
   `ATLAS_USE_LOCAL_MODELS` so `bge-m3` 1024-d is selectable; fix
   `QdrantStore` dim flip (128 vs 1024) before any live collection is created.
   Verify: near-dupe band test, version-strip test, dim-consistency test.

## P3 - retrieval scoring before real models (1 day)

10. Fix faithfulness + timings before wiring keys.
    `faithfulness = len(supported)/len(hits)` punishes good answers over many
    hits and forces a retry on nearly every correct response; `timings_ms` are
    0 except synthesize and `NODE_LATENCY` is never observed. Change to
    claims-over-supported (or at minimum supported-over-cited), scope the
    citation regex to chunk-id shapes, instrument per-node timing +
    `NODE_LATENCY.observe()`. Otherwise the first real LLM key produces
    perpetual retries and flat dashboards.
    Verify: 2-citation answer over 5 hits scores high and does not retry;
    `/metrics` shows per-node latency after a query.
11. Summarizer JSON fragility.
    `app/tools/summarizer.py` does bare `json.loads`; vLLM/Ollama providers
    return raw text with no schema enforcement, so one non-JSON completion
    500s `/ask` or aborts `run_ingestion` mid-loop. Add repair/retry or a
    constrained-decode path.
    Verify: invalid-JSON provider output test.

## P4 - observability and eval honesty (1-2 days)

12. Metrics or dashboards, pick one per panel.
    Store already knows `dropped_threshold`, `dropped_dedupe`,
    `eviction_age_days`, `llm_elapsed_ms{backend}`, `embedding_cache_hit{tier}`;
    add the counters/histograms, `observe()` latencies, then extend the two
    dashboards to the spec panel list with `MAX_*` red lines. Or trim
    README/Grafana prose to "skeleton". Do not ship dashboards that can never
    fill. Fix the retrieval `histogram_quantile` grouping so the `{{node}}`
    legend resolves.
    Verify: `curl /metrics` shows new series after seed + ask; Grafana renders
    non-empty panels.
13. Eval to minimum useful.
    Dataset is 2 rows (spec 50); harness reports `hit_at_5` + wall clock only
    (no hit@1/10, faithfulness, p50/p95, concurrency, naive-vs-full, CI gate);
    `provider` is the literal string `"reported by query response"`; no
    `perf/results/eval-*.json` committed though README calls it truth; `make
    eval` silently depends on `make seed` ordering. Grow the gold set
    incrementally, add the missing metrics + `--concurrent` sweep, fix the
    provider field, commit one artifact from `make seed && make eval`, document
    the ordering.
    Verify: `make eval` writes a complete JSON; one artifact committed.

## P5 - web, k8s, hygiene (1-2 days)

14. Frontend: pin, lock, config, then features.
    Pin `react/react-dom/vite/@vitejs/plugin-react` from `latest`, commit
    `package-lock.json`, switch `web/Dockerfile` to `npm ci`, add 10-line
    `vite.config.js` (`react()` + dev proxy `/api -> 127.0.0.1:8000`),
    flesh out `web/index.html` (title/meta). Then add Brief (topic pills,
    citation slider, `why_kept`, eviction ghosts), Ask drawer
    (authors/venue/PDF/section), Manage (caps, policy, thresholds, cursors).
    Verify: reproducible `docker compose build web`; dev-mode Ask works.
15. K8s: secret + scope honesty.
    `infra/k8s/atlas.yaml` references `secretRef: atlas-secrets` with no
    `Secret` manifest (pods stall `CreateContainerConfigError`); promises
    replicas 2 + HPA + Ingress + StatefulSets + NetworkPolicy + PDB, ships
    replicas 1 + Service + CronJob only. Add a commented `Secret` template or
    `kubectl create secret` doc, and either add the missing objects or narrow
    the docs to what ships. Confirm `backfill.py` CLI arity matches the CronJob
    args.
    Verify: `kubectl apply --dry-run=client -f infra/k8s/atlas.yaml` plus
    documented secret step.
16. Tests, CI, docs.
    Add the hardening + eviction + filter/pagination + retry tests above (P1-P3
    verification doubles as coverage). Add `.github/workflows/ci.yml` (ruff,
    pytest, eval-smoke, `pip-audit`). Fix `.gitignore` (`node_modules/`,
    `web/dist/`, dedupe `.venv/`). Delete the stale "not yet implemented"
    banner; fix the `llm_adapter` SIGHUP docstring (no handler exists; restart
    is required). `scripts/seed_topics.py --papers N` is per-topic, not total;
    clarify the help/README example.
    Verify: `pytest -q`, `ruff check app tests`, green CI on a test PR.

## Suggested order

P0 first (nothing else matters if the image does not build). Then P1 + P2
together (API truth + bound truth are the portfolio claims). Then P3 before
spending on real model keys. Then P4 (prove it), then P5 (polish). Each bullet
is independently committable; do not bundle P0 fixes with feature work.

## Non-goals

No new dependencies, no version sweeps, no full-file reformats. No live
200-trial-style benchmarking runs on a laptop; `OMP_NUM_THREADS=4` for local
work. No visibility changes, no pushes without owner request; commit locally
only. No regenerating perf artifacts to fake freshness; no inventing
measurements to fill dashboards.
