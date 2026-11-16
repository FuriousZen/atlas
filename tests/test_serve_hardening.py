from fastapi.testclient import TestClient

from app.api.main import MAX_BODY_BYTES, app
from app.store import reset_store


def _client() -> TestClient:
    reset_store()
    return TestClient(app, raise_server_exceptions=False)


def test_oversize_body_returns_413_with_envelope():
    client = _client()
    big = "x" * (MAX_BODY_BYTES + 1024)
    response = client.post("/ask", json={"question": big})
    assert response.status_code == 413
    payload = response.json()
    assert payload["error"] == "payload_too_large"
    assert "request_id" in payload
    assert response.headers.get("x-request-id") == payload["request_id"]


def test_malformed_content_length_returns_400_not_500():
    reset_store()
    body = b'{"question": "hi"}'
    scope_headers = [
        (b"host", b"testserver"),
        (b"content-type", b"application/json"),
        (b"content-length", b"not-a-number"),
    ]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/ask",
        "raw_path": b"/ask",
        "query_string": b"",
        "root_path": "",
        "headers": scope_headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    sent = {"first": True}

    async def receive():
        if sent["first"]:
            sent["first"] = False
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    messages: list[dict] = []

    async def send(message: dict) -> None:
        messages.append(message)

    import asyncio

    asyncio.run(app(scope, receive, send))
    statuses = [m.get("status") for m in messages if m.get("type") == "http.response.start"]
    assert statuses and statuses[0] == 400
    bodies = [m.get("body", b"") for m in messages if m.get("type") == "http.response.body"]
    import json as jsonlib

    payload = jsonlib.loads(b"".join(bodies).decode())
    assert payload["request_id"]
    headers = dict(messages[0].get("headers", []))
    assert headers.get(b"x-request-id")


def test_infinity_nan_rejected_with_422():
    client = _client()
    response = client.post(
        "/ask",
        content=b'{"question": "hi", "filters": {"min_citations": Infinity}}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_section_item_too_long_rejected_with_422():
    client = _client()
    response = client.post("/ask", json={"question": "hi", "filters": {"section": ["x" * 65]}})
    assert response.status_code == 422


def test_request_id_echoed():
    client = _client()
    response = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "abc123"
    anonymous = client.get("/health")
    assert anonymous.headers.get("x-request-id")
