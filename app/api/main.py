from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.agent.graphs import run_ingestion, run_query
from app.config import get_settings
from app.ingest.sources import OpenAlexClient, SemanticScholarClient
from app.metrics import BYTES, CHUNKS, EVICTIONS, KEPT, PAPERS, POLLED, REQUESTS
from app.store import CapacityBlocked, get_store
from app.tools.llm_adapter import llm_backend_status


logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 4 * 1024 * 1024

_runs: dict[str, dict[str, Any]] = {}

_RUN_ID_RE = re.compile(r"/ingest/runs/[^/]+")


def _metric_path(path: str) -> str:
    return _RUN_ID_RE.sub("/ingest/runs/:id", path)


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    topic: str | None = Field(default=None, max_length=120)
    min_citations: int | None = Field(default=None, ge=0, le=10_000_000)
    since: str | None = Field(default=None, max_length=32)
    section: list[str] | None = Field(default=None, max_length=10)
    sort: Literal["citations", "recent"] | None = Field(default=None)

    @field_validator("section")
    @classmethod
    def section_items_capped(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        for item in value:
            if len(item) > 64:
                raise ValueError("section items must be at most 64 characters")
        return value


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    question: str = Field(min_length=1, max_length=2000)
    filters: Filters | None = None

    @field_validator("question")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    topic: str = Field(min_length=1, max_length=120)
    since: str | None = Field(default=None, max_length=32)
    min_citations: int | None = Field(default=None, ge=0, le=10_000_000)
    limit: int = Field(default=50, ge=1, le=100)


settings = get_settings()
app = FastAPI(title="Atlas Research Intelligence", version="0.1.0")
origins = [item.strip() for item in settings.cors_origins.split(",") if item.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def request_metrics(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    raw_length = request.headers.get("content-length")
    if raw_length is not None and raw_length != "":
        try:
            declared = int(raw_length)
        except ValueError:
            payload = json.dumps({"error": "bad_request", "detail": "invalid content-length header", "request_id": request_id})
            return Response(content=payload, status_code=400, media_type="application/json", headers={"X-Request-ID": request_id})
        if declared > MAX_BODY_BYTES:
            payload = json.dumps({"error": "payload_too_large", "detail": f"request body exceeds {MAX_BODY_BYTES} bytes", "request_id": request_id})
            return Response(content=payload, status_code=413, media_type="application/json", headers={"X-Request-ID": request_id})
    # Count actual body bytes so Transfer-Encoding: chunked (no content-length) is covered.
    try:
        body = await request.body()
    except Exception:
        body = b""
    if len(body) > MAX_BODY_BYTES:
        payload = json.dumps({"error": "payload_too_large", "detail": f"request body exceeds {MAX_BODY_BYTES} bytes", "request_id": request_id})
        return Response(content=payload, status_code=413, media_type="application/json", headers={"X-Request-ID": request_id})
    # Replay the consumed body downstream (request.body() exhausts the receive channel).

    async def _replay_receive():
        return {"type": "http.request", "body": body, "more_body": False}

    replayed = Request(request.scope, _replay_receive)
    replayed.state.request_id = request_id
    try:
        response = await call_next(replayed)
    except Exception:
        REQUESTS.labels(_metric_path(request.url.path), request.method, "500").inc()
        raise
    REQUESTS.labels(_metric_path(request.url.path), request.method, str(response.status_code)).inc()
    response.headers["X-Request-ID"] = request_id
    return response


def _sanitize_jsonable(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {key: _sanitize_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_jsonable(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or request.headers.get("x-request-id") or uuid.uuid4().hex
    return JSONResponse(status_code=422, content={"detail": _sanitize_jsonable(jsonable_encoder(exc.errors()))}, headers={"X-Request-ID": request_id})


def _refresh_gauges() -> dict[str, int]:
    counts = get_store().counts()
    PAPERS.set(counts["papers"])
    CHUNKS.set(counts["chunks"])
    BYTES.set(counts["bytes"])
    return counts


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "atlas", "llm": llm_backend_status(), "capacity": {"papers": settings.atlas_max_papers, "chunks": settings.atlas_max_chunks, "bytes": settings.max_bytes, "eviction": settings.atlas_eviction, "min_keep_days": settings.atlas_min_keep_days}, "store": _refresh_gauges()}


@app.get("/ready")
def ready() -> dict[str, Any]:
    # Provider readiness is deliberately non-blocking: a stub provider is valid.
    return {"status": "ready", "dependencies": {"store": "in_memory", "llm": llm_backend_status()["effective_backend"]}}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ask")
def ask(body: AskRequest, explain: bool = False, sort: str | None = Query(default=None)) -> dict[str, Any]:
    filters = (body.filters.model_dump(exclude_none=True) if body.filters else {})
    body_sort = filters.pop("sort", None)
    effective_sort = sort or body_sort or "citations"
    if effective_sort not in ("citations", "recent"):
        raise HTTPException(status_code=422, detail="sort must be 'citations' or 'recent'")
    result = run_query(body.question, filters, sort=effective_sort, explain=explain)
    return {"question": body.question, **result, "store": _refresh_gauges()}


@app.get("/papers")
def papers(topic: str | None = Query(default=None, max_length=120), since: str | None = Query(default=None, max_length=32), min_citations: int | None = Query(default=None, ge=0, le=10_000_000), limit: int = Query(default=50, ge=1, le=100), offset: int = Query(default=0, ge=0, le=1_000_000), sort: str = Query(default="citations")) -> dict[str, Any]:
    if sort not in ("citations", "recent"):
        raise HTTPException(status_code=422, detail="sort must be 'citations' or 'recent'")
    parsed_since = _parse_date(since)
    rows = get_store().list_papers(topic=topic, since=parsed_since, min_citations=min_citations, limit=limit, offset=offset, sort=sort)
    return {"papers": [row.to_dict() for row in rows], "limit": limit, "offset": offset, "count": len(rows)}


@app.get("/brief")
def brief(topic: str | None = Query(default=None, max_length=120), since: str | None = Query(default=None, max_length=32), min_citations: int | None = Query(default=None, ge=0, le=10_000_000), limit: int = Query(default=20, ge=1, le=100), sort: str = Query(default="citations")) -> dict[str, Any]:
    if sort not in ("citations", "recent"):
        raise HTTPException(status_code=422, detail="sort must be 'citations' or 'recent'")
    rows = get_store().list_papers(topic=topic, since=_parse_date(since), min_citations=min_citations, limit=limit, sort=sort)
    return {"papers": [row.to_dict() for row in rows], "kept": len(rows), "store": _refresh_gauges(), "llm": llm_backend_status()}


async def _ingest(body: IngestRequest, run_id: str) -> None:
    try:
        threshold = body.min_citations if body.min_citations is not None else settings.topic_threshold(body.topic)
        s2 = SemanticScholarClient(settings.atlas_s2_base, settings.s2_api_key)
        candidates = await asyncio.to_thread(s2.search, body.topic, body.topic, body.since, body.limit)
        # OpenAlex is deliberately optional enrichment; one source failure must not erase S2 results.
        try:
            candidates += await asyncio.to_thread(OpenAlexClient().search, body.topic, body.topic, body.since, threshold, min(body.limit, 50))
        except Exception:
            logger.warning("openalex enrichment failed for run %s", run_id, exc_info=True)
        result = run_ingestion(candidates, body.topic, threshold)
        POLLED.labels(body.topic).inc(result["polled"])
        KEPT.labels(body.topic).inc(result["kept"])
        EVICTIONS.labels("capacity").inc(len(result["evicted"]))
        _refresh_gauges()
        entry = _runs.get(run_id)
        if entry is not None:
            entry["status"] = "ok"
            entry["finished_at"] = datetime.now(timezone.utc).isoformat()
            entry["result"] = {"polled": result["polled"], "kept": result["kept"], "indexed": result["indexed"]}
    except CapacityBlocked as exc:
        logger.warning("ingest run %s blocked by storage bound (needed=%s)", run_id, exc.needed)
        entry = _runs.get(run_id)
        if entry is not None:
            entry["status"] = "capacity_blocked"
            entry["finished_at"] = datetime.now(timezone.utc).isoformat()
            entry["error"] = "capacity_blocked"
            entry["needed"] = exc.needed
            entry["retry_after"] = exc.retry_after
    except Exception as exc:
        logger.exception("ingest run %s failed", run_id)
        entry = _runs.get(run_id)
        if entry is not None:
            entry["status"] = "error"
            entry["finished_at"] = datetime.now(timezone.utc).isoformat()
            entry["error"] = str(exc)


@app.post("/ingest/run", status_code=202)
async def ingest(body: IngestRequest) -> dict[str, str]:
    run_id = str(uuid.uuid4())
    _runs[run_id] = {"status": "pending", "topic": body.topic, "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None}
    asyncio.create_task(_ingest(body, run_id))
    return {"run_id": run_id, "status": "accepted"}


@app.get("/ingest/runs/{run_id}")
def ingest_run(run_id: str) -> dict[str, Any]:
    entry = _runs.get(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown run_id")
    if entry.get("status") == "capacity_blocked":
        raise HTTPException(status_code=429, detail="storage bound reached; retry later", headers={"Retry-After": str(entry.get("retry_after", 0))})
    return {"run_id": run_id, **entry}


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        raise HTTPException(status_code=422, detail="since must be an ISO-8601 date")
