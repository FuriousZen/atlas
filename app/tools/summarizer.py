from __future__ import annotations

import json
from typing import Any

from app.models import Paper, Summary
from app.tools.llm_adapter import get_llm

from langchain_core.prompts import ChatPromptTemplate


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "tldr": {"type": "string"},
        "contributions": {"type": "array", "items": {"type": "string"}},
        "method": {"type": "string"},
        "results": {"type": "object"},
        "limits": {"type": "string"},
        "why_kept": {"type": "string"},
    },
    "required": ["tldr", "contributions", "method", "results", "limits", "why_kept"],
}


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a careful research distiller. Return only JSON matching the requested schema. Do not invent results or numbers. If the paper does not state a result, say so."),
    ("human", "Title: {title}\nAbstract: {abstract}\nExtracted sections: {sections}\nCitation count: {citations}"),
])


def _prompt(paper: Paper, extracted_text: str = "") -> str:
    # LangChain owns prompt construction; the provider adapter owns transport.
    return SUMMARY_PROMPT.invoke({"title": paper.title, "abstract": paper.abstract[:12000], "sections": extracted_text[:20000], "citations": paper.citations}).to_string()


def fallback_summary(paper: Paper, extracted_text: str = "") -> Summary:
    """Extractive fallback built only from caller-supplied paper data; never calls a provider."""
    abstract = (paper.abstract or "").strip()
    sections_text = (extracted_text or "").strip()
    excerpt = abstract[:500] or sections_text[:500] or paper.title[:500]
    return Summary(
        tldr=f"{paper.title[:280]} — fallback extraction (provider returned non-JSON).",
        contributions=[excerpt[:280]] if excerpt else [paper.title[:280]],
        method="Fallback extraction from title/abstract; provider output was not valid JSON.",
        results={"note": "fallback; see abstract", "abstract_excerpt": abstract[:500]},
        limits="Fallback summary; full LLM synthesis unavailable.",
        why_kept=f"{paper.citations} citations >= threshold; fallback summary",
        extraction="fallback",
    )


def _candidate_json_objects(text: str, max_candidates: int = 5) -> list[str]:
    """Collect balanced top-level {...} spans, largest first, capped to avoid unbounded work."""
    spans: list[str] = []
    depth = 0
    start: int | None = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append(text[start : index + 1])
                    start = None
    spans.sort(key=len, reverse=True)
    return spans[:max(0, max_candidates)]


def _coerce_summary(data: dict[str, Any]) -> Summary:
    return Summary(tldr=str(data.get("tldr", "")), contributions=[str(x) for x in data.get("contributions", [])], method=str(data.get("method", "")), results=data.get("results", {}) if isinstance(data.get("results", {}), dict) else {"value": data.get("results")}, limits=str(data.get("limits", "")), why_kept=str(data.get("why_kept", "")), extraction="llm")


def summarize(paper: Paper, extracted_text: str = "", backend: str | None = None) -> Summary:
    try:
        result = get_llm(backend).generate(_prompt(paper, extracted_text), SUMMARY_SCHEMA, temperature=0.2)
        text = result.text or ""
    except Exception:
        return fallback_summary(paper, extracted_text)
    try:
        data: Any = json.loads(text)
    except (TypeError, ValueError):
        data = None
        for candidate in _candidate_json_objects(text):
            try:
                obj = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            if isinstance(obj, dict):
                data = obj
                break
        if data is None:
            return fallback_summary(paper, extracted_text)
    if not isinstance(data, dict):
        return fallback_summary(paper, extracted_text)
    try:
        return _coerce_summary(data)
    except Exception:
        return fallback_summary(paper, extracted_text)
