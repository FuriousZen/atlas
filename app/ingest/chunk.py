from __future__ import annotations

import re

from app.models import Chunk, Paper


def chunk_paper(paper: Paper, max_words: int = 256, overlap_words: int = 32) -> list[Chunk]:
    """Section-aware splitter; prefers LlamaIndex SentenceSplitter when installed."""
    sections: list[tuple[str, str]] = []
    for key, value in paper.summary.items():
        if isinstance(value, str) and value.strip():
            sections.append((key, value.strip()))
        elif isinstance(value, list):
            sections.append((key, " ".join(str(item) for item in value)))
        elif isinstance(value, dict):
            sections.append((key, " ".join(f"{k}: {v}" for k, v in value.items())))
    if paper.abstract:
        sections.append(("abstract", paper.abstract.strip()))
    if not sections:
        sections = [("abstract", paper.title)]
    splitter = _sentence_splitter()
    output: list[Chunk] = []
    number = 0
    for section, text in sections:
        parts = splitter(text) if splitter is not None else _word_window(text, max_words, overlap_words)
        for part in parts:
            output.append(Chunk(id=f"{paper.id}#chunk-{number:04d}", paper_id=paper.id, n=number, section=section, text=part))
            number += 1
    return output


def _sentence_splitter():
    """Return a split function via LlamaIndex SentenceSplitter(512, 64), or None."""
    try:
        try:
            from llama_index.core.node_parser import SentenceSplitter
        except ImportError:
            from llama_index import SentenceSplitter  # type: ignore[no-redef]
    except ImportError:
        return None
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    return splitter.split_text


def _word_window(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = re.findall(r"\S+", text)
    parts: list[str] = []
    start = 0
    while start < len(words):
        parts.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
        start += max_words - overlap_words
    return parts
