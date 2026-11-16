"""Qdrant-shaped MCP surface backed by the configured store.

The local implementation keeps the same tool contract while Qdrant is not
available. The production adapter can replace the internals without changing
the agent graph or MCP client.
"""

from __future__ import annotations

from typing import Any

from app.mcp.compat import FastMCP
from app.store import get_store

mcp = FastMCP("atlas-qdrant")


@mcp.tool()
def search(query: str, topic: str | None = None, min_citations: int | None = None, k: int = 12) -> list[dict[str, Any]]:
    """Return bounded dense-search-shaped hits with Qdrant-compatible fields."""
    if not 1 <= k <= 50:
        raise ValueError("k must be between 1 and 50")
    return [hit.to_dict() for hit in get_store().search(query, topic, min_citations, limit=k)]


@mcp.tool()
def delete(paper_ids: list[str]) -> dict[str, int]:
    """Remove all chunks belonging to papers selected by a capacity eviction."""
    if len(paper_ids) > 100:
        raise ValueError("at most 100 paper ids per delete")
    deleted = 0
    store = get_store()
    for paper_id in paper_ids:
        store.papers.pop(paper_id, None)
        for chunk_id in [key for key, chunk in store.chunks.items() if chunk.paper_id == paper_id]:
            store.chunks.pop(chunk_id, None)
            deleted += 1
    return {"deleted_chunks": deleted}


if __name__ == "__main__":
    mcp.run(transport="stdio")
