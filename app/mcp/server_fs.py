from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from app.mcp.compat import FastMCP

mcp = FastMCP("atlas-filesystem")
DATA_ROOT = Path(os.getenv("ATLAS_DATA_ROOT", "./cache")).resolve()
MAX_BYTES = 2 * 1024 * 1024


def _contained(path: str) -> Path:
    target = (DATA_ROOT / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if target != DATA_ROOT and DATA_ROOT not in target.parents:
        raise ValueError("path is outside ATLAS_DATA_ROOT")
    return target


@mcp.tool()
def read(path: str, max_bytes: int = MAX_BYTES) -> dict[str, Any]:
    """Read a bounded cached text extraction under the data root."""
    target = _contained(path)
    if max_bytes < 1 or max_bytes > MAX_BYTES:
        raise ValueError("max_bytes exceeds the filesystem tool limit")
    data = target.read_bytes()[:max_bytes]
    return {"path": str(target.relative_to(DATA_ROOT)), "text": data.decode("utf-8", errors="replace"), "truncated": target.stat().st_size > max_bytes}


@mcp.tool()
def grep(pattern: str, path: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
    """Search extracted text with a bounded, compiled regular expression."""
    if len(pattern) > 8192 or not 1 <= limit <= 1000:
        raise ValueError("invalid grep bounds")
    expression = re.compile(pattern)
    root = _contained(path or ".")
    results: list[dict[str, Any]] = []
    for target in sorted(root.rglob("*")):
        if not target.is_file() or len(results) >= limit:
            continue
        try:
            for line_number, line in enumerate(target.read_text(errors="replace").splitlines(), 1):
                if expression.search(line):
                    results.append({"path": str(target.relative_to(DATA_ROOT)), "line": line_number, "text": line[:2000]})
                    if len(results) >= limit:
                        break
        except OSError:
            continue
    return results


if __name__ == "__main__":
    mcp.run(transport="stdio")
