from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.ingest.sources import OpenAlexClient, SemanticScholarClient
from app.mcp.compat import FastMCP

mcp = FastMCP("atlas-research-sources")


@mcp.tool()
def semantic_scholar_search(query: str, topic: str | None = None, since: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Search Semantic Scholar and return normalized paper candidates."""
    if not query.strip() or len(query) > 500:
        raise ValueError("query must contain 1-500 characters")
    topic = topic or query
    papers = SemanticScholarClient(get_settings().atlas_s2_base, get_settings().s2_api_key).search(query, topic, since, min(limit, 100))
    return [paper.to_dict() for paper in papers]


@mcp.tool()
def openalex_search(query: str, topic: str | None = None, since: str | None = None, min_citations: int = 0, limit: int = 50) -> list[dict[str, Any]]:
    """Search OpenAlex for cross-discipline citation metadata."""
    topic = topic or query
    papers = OpenAlexClient().search(query, topic, since, min_citations, min(limit, 50))
    return [paper.to_dict() for paper in papers]


if __name__ == "__main__":
    mcp.run(transport="stdio")
