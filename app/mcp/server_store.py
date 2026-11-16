from __future__ import annotations

from typing import Any

from app.mcp.compat import FastMCP
from app.store import get_store

mcp = FastMCP("atlas-research-store")


@mcp.tool()
def search(query: str, topic: str | None = None, min_citations: int | None = None, limit: int = 12) -> list[dict[str, Any]]:
    """Search retained summaries and chunks with bounded results."""
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    return [hit.to_dict() for hit in get_store().search(query, topic, min_citations, limit=limit)]


@mcp.tool()
def list_papers(topic: str | None = None, min_citations: int | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """List retained papers by citation count."""
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("invalid pagination")
    return [paper.to_dict() for paper in get_store().list_papers(topic, min_citations=min_citations, limit=limit, offset=offset)]


@mcp.tool()
def capacity() -> dict[str, Any]:
    """Return current bounded-store usage."""
    from app.config import get_settings

    s = get_settings()
    return {"used": get_store().counts(), "limits": {"papers": s.atlas_max_papers, "chunks": s.atlas_max_chunks, "bytes": s.max_bytes}}


if __name__ == "__main__":
    mcp.run(transport="stdio")
