from __future__ import annotations

from typing import Any

from app.agent.graphs import run_query
from app.mcp.compat import FastMCP
from app.store import get_store

mcp = FastMCP("atlas-rag")


@mcp.tool()
def ask(question: str, topic: str | None = None, min_citations: int | None = None, since: str | None = None, sort: str = "citations") -> dict[str, Any]:
    """Ask Atlas to retrieve and synthesize cited research evidence.

    Intended for an external agent. Natural prompts such as ``new science
    today?`` and ``most damning scientific discoveries published?`` are
    expanded by the query graph before retrieval.
    """
    filters: dict[str, Any] = {}
    if topic:
        filters["topic"] = topic
    if min_citations is not None:
        filters["min_citations"] = min_citations
    if since:
        filters["since"] = since
    if sort not in ("citations", "recent"):
        sort = "citations"
    return run_query(question, filters, sort=sort)


@mcp.tool()
def new_science_today(topic: str | None = None, min_citations: int | None = None) -> dict[str, Any]:
    """Return recently ingested research, optionally restricted by topic."""
    return ask("new science today?", topic, min_citations)


@mcp.tool()
def most_damning_discoveries(topic: str | None = None, min_citations: int | None = None) -> dict[str, Any]:
    """Retrieve high-impact evidence about negative, disputed, or limited findings."""
    return ask("most damning scientific discoveries published?", topic, min_citations)


@mcp.tool()
def library_status() -> dict[str, Any]:
    """Return retained-store counts and provider status without invoking an LLM."""
    from app.tools.llm_adapter import llm_backend_status

    return {"store": get_store().counts(), "llm": llm_backend_status()}


if __name__ == "__main__":
    mcp.run(transport="stdio")


def main() -> None:
    """Start the Atlas MCP server over stdio for an external agent."""
    mcp.run(transport="stdio")
