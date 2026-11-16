from __future__ import annotations

from typing import Any

from app.mcp.compat import FastMCP
from app.store import get_store

mcp = FastMCP("atlas-postgres")


@mcp.tool()
def filter_papers(topic: str | None = None, min_citations: int | None = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    """Apply structured metadata filters with bounded pagination."""
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("invalid pagination")
    return [paper.to_dict() for paper in get_store().list_papers(topic, min_citations=min_citations, limit=limit, offset=offset)]


@mcp.tool()
def get_paper(paper_id: str) -> dict[str, Any] | None:
    """Fetch one canonical paper row."""
    paper = get_store().get_paper(paper_id)
    return paper.to_dict() if paper else None


@mcp.tool()
def fts(query: str, topic: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """Use the local lexical fallback for the Postgres full-text contract."""
    return [hit.to_dict() for hit in get_store().search(query, topic=topic, limit=min(max(limit, 1), 20))]


if __name__ == "__main__":
    mcp.run(transport="stdio")
