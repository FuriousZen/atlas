from app.mcp.server_sources import mcp, openalex_search

__all__ = ["mcp", "openalex_search"]


if __name__ == "__main__":
    mcp.run(transport="stdio")
