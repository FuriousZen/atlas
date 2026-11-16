from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import httpx


@dataclass
class CandidatePaper:
    id: str
    title: str
    authors: list[str]
    venue: str | None
    year: int | None
    citations: int
    topic: str
    url: str
    abstract: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_url: str | None = None
    source: str = "unknown"
    published_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.published_at:
            result["published_at"] = self.published_at.isoformat()
        return result


def _date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class SemanticScholarClient:
    def __init__(self, base_url: str = "https://api.semanticscholar.org/graph/v1", api_key: str | None = None, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=20.0)

    def search(self, query: str, topic: str, since: str | None = None, limit: int = 100) -> list[CandidatePaper]:
        params: dict[str, Any] = {"query": query, "limit": min(max(limit, 1), 100), "fields": "title,authors,venue,year,citationCount,externalIds,url,abstract,publicationDate"}
        if since:
            params["year"] = f"{since[:4]}-"
        headers = {"X-API-KEY": self.api_key} if self.api_key else {}
        for attempt in range(3):
            response = self.client.get(f"{self.base_url}/paper/search", params=params, headers=headers)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = float(response.headers.get("Retry-After", "0.5"))
                time.sleep(min(8.0, retry_after * (2**attempt)))
                continue
            response.raise_for_status()
            data = response.json()
            return [self._paper(row, topic) for row in data.get("data", []) if self._valid(row)]
        raise RuntimeError("Semantic Scholar remained unavailable after bounded retries")

    @staticmethod
    def _valid(row: dict[str, Any]) -> bool:
        return bool(row.get("paperId") and row.get("title"))

    @staticmethod
    def _paper(row: dict[str, Any], topic: str) -> CandidatePaper:
        external = row.get("externalIds") or {}
        doi = external.get("DOI")
        arxiv = external.get("ArXiv")
        paper_id = f"doi:{doi.lower()}" if doi else (f"arxiv:{arxiv}" if arxiv else f"s2:{row['paperId']}")
        return CandidatePaper(id=paper_id, title=_clean_title(row["title"]), authors=[a.get("name", "") for a in row.get("authors", []) if a.get("name")], venue=row.get("venue"), year=row.get("year"), citations=max(0, int(row.get("citationCount") or 0)), topic=topic, url=row.get("url") or (f"https://doi.org/{doi}" if doi else f"https://www.semanticscholar.org/paper/{row['paperId']}"), abstract=row.get("abstract") or "", doi=doi, arxiv_id=arxiv, published_at=_date(row.get("publicationDate")), source="semantic_scholar")


class OpenAlexClient:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(timeout=20.0)

    def search(self, query: str, topic: str, since: str | None = None, min_citations: int = 0, limit: int = 50) -> list[CandidatePaper]:
        filters = [f"title.search:{query}", f"cited_by_count:>{max(0, min_citations)}"]
        if since:
            filters.append(f"from_publication_date:{since}")
        response = self.client.get("https://api.openalex.org/works", params={"filter": ",".join(filters), "per-page": min(max(limit, 1), 50), "mailto": "atlas@example.invalid"})
        response.raise_for_status()
        output: list[CandidatePaper] = []
        for row in response.json().get("results", []):
            doi = (row.get("doi") or "").removeprefix("https://doi.org/") or None
            work_id = row.get("id", "").rsplit("/", 1)[-1]
            output.append(CandidatePaper(id=f"doi:{doi.lower()}" if doi else f"openalex:{work_id}", title=_clean_title(row.get("title") or "Untitled"), authors=[(a.get("author") or {}).get("display_name", "") for a in row.get("authorships", [])], venue=((row.get("primary_location") or {}).get("source") or {}).get("display_name"), year=row.get("publication_year"), citations=max(0, int(row.get("cited_by_count") or 0)), topic=topic, url=row.get("doi") or row.get("id", ""), abstract="", doi=doi, pdf_url=((row.get("open_access") or {}).get("oa_url")), source="openalex", published_at=_date(row.get("publication_date"))))
        return output
