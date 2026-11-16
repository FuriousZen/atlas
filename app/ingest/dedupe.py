from __future__ import annotations

import re
from app.ingest.sources import CandidatePaper


ARXIV_VERSION_RE = re.compile(r"v\d+$")

MERGE_SIMILARITY = 0.82
POSSIBLE_DUPE_MIN = 0.72


def base_arxiv_id(value: str) -> str:
    """Strip a trailing arXiv version suffix (v\\d+) only, keeping other 'v's."""
    return ARXIV_VERSION_RE.sub("", value)


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def title_similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(candidates: list[CandidatePaper]) -> tuple[list[CandidatePaper], list[dict[str, object]]]:
    """Fold DOI/arXiv versions and close title matches, preserving provenance.

    Title similarity >= 0.82 merges (higher citations retained). Similarity in
    [0.72, 0.82) records a {"reason": "possibleDuplicateOf"} pair but keeps
    both papers. Callers may pass pairs to record_duplicates() best-effort.
    """
    kept: list[CandidatePaper] = []
    duplicates: list[dict[str, object]] = []
    for candidate in candidates:
        match: CandidatePaper | None = None
        reason = ""
        score = 0.0
        for existing in kept:
            if candidate.doi and existing.doi and candidate.doi.lower() == existing.doi.lower():
                match, reason, score = existing, "doi_or_arxiv_version", 1.0
                break
            if candidate.arxiv_id and existing.arxiv_id and base_arxiv_id(candidate.arxiv_id) == base_arxiv_id(existing.arxiv_id):
                match, reason, score = existing, "doi_or_arxiv_version", 1.0
                break
            sim = title_similarity(candidate.title, existing.title)
            if sim >= MERGE_SIMILARITY:
                match, reason, score = existing, "title_similarity", sim
                break
        if match is not None:
            if candidate.citations > match.citations:
                kept[kept.index(match)] = candidate
                retained, discarded = candidate, match
            else:
                retained, discarded = match, candidate
            duplicates.append({"kept_id": retained.id, "duplicate_id": discarded.id, "score": score, "reason": reason})
            continue
        best: CandidatePaper | None = None
        best_sim = 0.0
        for existing in kept:
            sim = title_similarity(candidate.title, existing.title)
            if POSSIBLE_DUPE_MIN <= sim < MERGE_SIMILARITY and sim > best_sim:
                best, best_sim = existing, sim
        if best is not None:
            duplicates.append({"kept_id": best.id, "duplicate_id": candidate.id, "score": best_sim, "reason": "possibleDuplicateOf"})
        kept.append(candidate)
    return kept, duplicates
