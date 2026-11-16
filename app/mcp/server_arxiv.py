from __future__ import annotations

import hashlib
import os
from pathlib import Path

import httpx
import xml.etree.ElementTree as ET

from app.mcp.compat import FastMCP

mcp = FastMCP("atlas-arxiv")


@mcp.tool()
def search(query: str, limit: int = 50) -> list[dict[str, str]]:
    """Fetch recent arXiv metadata; arXiv does not provide citation counts."""
    if not query.strip() or not 1 <= limit <= 50:
        raise ValueError("invalid query or limit")
    response = httpx.get("https://export.arxiv.org/api/query", params={"search_query": f"all:{query}", "sortBy": "submittedDate", "max_results": limit}, timeout=20)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", namespace):
        identifier = (entry.findtext("atom:id", "", namespace)).rsplit("/", 1)[-1]
        papers.append({"id": identifier, "title": " ".join((entry.findtext("atom:title", "", namespace)).split()), "abstract": " ".join((entry.findtext("atom:summary", "", namespace)).split()), "published": entry.findtext("atom:published", "", namespace), "pdf_url": f"https://arxiv.org/pdf/{identifier}.pdf", "source": "arxiv"})
    return papers


@mcp.tool()
def fetch_pdf(arxiv_id: str, max_bytes: int = 25 * 1024 * 1024) -> dict[str, object]:
    """Cache one PDF with a per-download bound and return its content hash."""
    if not arxiv_id or len(arxiv_id) > 80 or max_bytes < 1 or max_bytes > 25 * 1024 * 1024:
        raise ValueError("invalid arXiv id or download bound")
    root = Path(os.getenv("ATLAS_DATA_ROOT", "./cache")).resolve()
    destination = root / "pdfs" / f"{arxiv_id.replace('/', '_')}.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        payload = destination.read_bytes()
        return {"path": str(destination), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "cached": True}
    with httpx.stream("GET", f"https://arxiv.org/pdf/{arxiv_id}.pdf", timeout=30) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_bytes(64 * 1024):
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ValueError("PDF exceeds per-download bound")
    destination.write_bytes(data)
    return {"path": str(destination), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "cached": False}


if __name__ == "__main__":
    mcp.run(transport="stdio")
