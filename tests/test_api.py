import asyncio

from fastapi.testclient import TestClient

import app.api.main as main_mod
from app.api.main import app
from app.store import CapacityBlocked, reset_store


def test_health_is_available_without_provider():
    reset_store()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["llm"]["stub"] is True


def test_ask_rejects_unknown_fields():
    client = TestClient(app)
    response = client.post("/ask", json={"question": "new science today?", "unexpected": True})
    assert response.status_code == 422


def test_capacity_blocked_run_maps_to_429(monkeypatch):
    class StubS2:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, *args, **kwargs):
            return []

    class StubOA:
        def search(self, *args, **kwargs):
            return []

    def blocked(candidates, topic, threshold):
        raise CapacityBlocked(needed=2, retry_after=60)

    monkeypatch.setattr(main_mod, "SemanticScholarClient", StubS2)
    monkeypatch.setattr(main_mod, "OpenAlexClient", lambda *args, **kwargs: StubOA())
    monkeypatch.setattr(main_mod, "run_ingestion", blocked)
    run_id = "test-blocked-run"
    main_mod._runs[run_id] = {"status": "pending", "topic": "reranking", "started_at": "never", "finished_at": None}
    try:
        asyncio.run(main_mod._ingest(main_mod.IngestRequest(topic="reranking", limit=5), run_id))
        entry = main_mod._runs[run_id]
        assert entry["status"] == "capacity_blocked"
        assert entry["retry_after"] == 60
        response = TestClient(app).get(f"/ingest/runs/{run_id}")
        assert response.status_code == 429
        assert response.headers.get("retry-after") == "60"
    finally:
        main_mod._runs.pop(run_id, None)
