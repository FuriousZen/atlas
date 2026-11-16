"""Prometheus series for Atlas, owned in one place to avoid import cycles.

app.api.main imports these for /metrics exposition; agent and tool modules
import from here (lazily, as needed) so app.agent.graphs never has to import
app.api.main. Series names are load-bearing for Grafana dashboards and
prometheus.yml — do not rename without updating those.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("atlas_requests_total", "HTTP requests", ["path", "method", "status"])
NODE_LATENCY = Histogram("atlas_node_latency_seconds", "Agent node latency", ["node"])
PAPERS = Gauge("atlas_store_papers", "Papers currently retained")
CHUNKS = Gauge("atlas_store_chunks", "Chunks currently retained")
BYTES = Gauge("atlas_store_bytes", "Chunk text bytes currently retained")
EVICTIONS = Counter("atlas_evictions_total", "Papers evicted", ["reason"])
POLLED = Counter("atlas_papers_polled_total", "Papers returned by source", ["topic"])
KEPT = Counter("atlas_papers_kept_total", "Papers passing threshold", ["topic"])

DROPPED_THRESHOLD = Counter(
    "atlas_papers_dropped_threshold_total",
    "Papers failing the per-topic citation threshold before any LLM spend",
    ["topic"],
)
DROPPED_DEDUPE = Counter(
    "atlas_papers_dropped_dedupe_total",
    "Papers merged away as duplicates after passing threshold",
    ["topic"],
)
EVICTION_AGE_DAYS = Histogram(
    "atlas_eviction_age_days",
    "Age in days of evicted papers at eviction time",
    buckets=(1.0, 3.0, 7.0, 14.0, 30.0, 60.0, 90.0, 180.0, 365.0),
)
LLM_ELAPSED_MS = Histogram(
    "atlas_llm_elapsed_ms",
    "LLM provider elapsed milliseconds per synthesize call",
    ["backend"],
    buckets=(1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0),
)
EMBEDDING_CACHE_HITS = Counter(
    "atlas_embedding_cache_hit_total",
    "In-process fake-embedding cache lookups by tier",
    ["tier"],
)
