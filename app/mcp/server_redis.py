from __future__ import annotations

import time
from typing import Any

from app.mcp.compat import FastMCP

mcp = FastMCP("atlas-redis")
_cache: dict[str, tuple[float, Any]] = {}


@mcp.tool()
def get(key: str) -> Any:
    """Read a cache value; expired values are removed."""
    value = _cache.get(key)
    if value is None:
        return None
    expires, payload = value
    if expires < time.time():
        _cache.pop(key, None)
        return None
    return payload


@mcp.tool()
def put(key: str, value: Any, ttl_seconds: int = 3600) -> bool:
    """Write a bounded cache value with an expiry."""
    if len(key) > 512 or not 1 <= ttl_seconds <= 2_592_000:
        raise ValueError("invalid cache key or TTL")
    _cache[key] = (time.time() + ttl_seconds, value)
    return True


@mcp.tool()
def delete(key: str) -> bool:
    """Invalidate one cache value."""
    return _cache.pop(key, None) is not None


if __name__ == "__main__":
    mcp.run(transport="stdio")
