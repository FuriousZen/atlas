from __future__ import annotations

from typing import Any, Callable


try:
    from fastmcp import FastMCP  # type: ignore
except Exception:
    try:
        # The MCP SDK ships the same FastMCP server class under this path.
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except Exception:
        class FastMCP:  # pragma: no cover - used only when optional MCP deps are unavailable
            """Small compatibility shell so direct tool functions remain importable."""

            def __init__(self, name: str) -> None:
                self.name = name
                self.tools: dict[str, Callable[..., Any]] = {}

            def tool(self, fn: Callable[..., Any] | None = None, **_: Any):
                def register(func: Callable[..., Any]) -> Callable[..., Any]:
                    self.tools[func.__name__] = func
                    return func
                return register(fn) if fn else register

            def run(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("FastMCP is unavailable; install fastmcp to run an MCP transport")
