from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM — decoupled: agents API works with no provider; plug one in later via env
    # Backend is "auto" by default → first available key (gemini/openai/anthropic), else "stub" (heuristic).
    # Set ATLAS_LLM_BACKEND to force a backend; agents API never requires a key to start.
    atlas_llm_backend: Literal["auto", "stub", "none", "gemini", "openai", "anthropic", "vllm", "ollama"] = "auto"
    gemini_api_key: str | None = None
    google_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    vllm_url: str = "http://vllm:8000"
    ollama_host: str = "http://ollama:11434"
    atlas_gemini_model: str = "gemini-2.0-flash"
    atlas_openai_model: str = "gpt-4o-mini"
    atlas_anthropic_model: str = "claude-3-5-haiku-latest"
    atlas_ollama_model: str = "qwen2.5:7b"
    atlas_vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Sources
    s2_api_key: str | None = None
    atlas_s2_base: str = "https://api.semanticscholar.org/graph/v1"
    topics: str = "reranking:80,retrieval:50,rag:50,agents:50"

    # Bound — eviction
    atlas_max_papers: int = 1500
    atlas_max_chunks: int = 15000
    atlas_max_bytes_mb: int = 2048
    atlas_min_keep_days: int = 7
    atlas_eviction: Literal["oldest_first", "citation_agnostic"] = "oldest_first"

    # Vector
    atlas_embed_model: str = "BAAI/bge-m3"
    atlas_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    atlas_use_local_models: bool = False
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "atlas_papers"
    qdrant_api_key: str | None = None

    # Structured + cache
    database_url: str = "postgresql+psycopg://atlas:atlas@postgres:5432/atlas"
    redis_url: str = "redis://redis:6379/0"

    # API
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = ""

    @property
    def max_bytes(self) -> int:
        return self.atlas_max_bytes_mb * 1024 * 1024

    @property
    def gemini_key(self) -> str | None:
        return _usable_secret(self.gemini_api_key) or _usable_secret(self.google_api_key) or _usable_secret(os.getenv("GEMINI_API_KEY")) or _usable_secret(os.getenv("GOOGLE_API_KEY"))

    @property
    def openai_key(self) -> str | None:
        return _usable_secret(self.openai_api_key) or _usable_secret(os.getenv("OPENAI_API_KEY"))

    @property
    def anthropic_key(self) -> str | None:
        return _usable_secret(self.anthropic_api_key) or _usable_secret(os.getenv("ANTHROPIC_API_KEY"))

    @property
    def llm_configured(self) -> bool:
        """Any provider key present (or a local URL assumed configured when forced)."""
        if self.atlas_llm_backend in ("gemini",):
            return bool(self.gemini_key)
        if self.atlas_llm_backend in ("openai",):
            return bool(self.openai_key)
        if self.atlas_llm_backend in ("anthropic",):
            return bool(self.anthropic_key)
        if self.atlas_llm_backend in ("vllm", "ollama"):
            return True  # local URLs, assumed reachable; readiness probes separately
        if self.atlas_llm_backend in ("stub", "none"):
            return False
        # auto: configured if any hosted key present or local forced elsewhere
        return bool(self.gemini_key or self.openai_key or self.anthropic_key)

    @property
    def effective_llm_backend(self) -> str:
        """Resolve 'auto' to the first available backend, otherwise 'stub' (no keys → heuristic)."""
        if self.atlas_llm_backend != "auto":
            return self.atlas_llm_backend
        if self.gemini_key:
            return "gemini"
        if self.openai_key:
            return "openai"
        if self.anthropic_key:
            return "anthropic"
        # no hosted keys → stub (decoupled). vllm/ollama are opt-in via explicit ATLAS_LLM_BACKEND
        return "stub"

    def parsed_topics(self) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for part in self.topics.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                topic, thresh = part.rsplit(":", 1)
                try:
                    out.append((topic.strip(), int(thresh.strip())))
                except ValueError:
                    out.append((part, 50))
            else:
                out.append((part, 50))
        return out

    def topic_threshold(self, topic: str) -> int:
        for t, thresh in self.parsed_topics():
            if t == topic:
                return thresh
        return 50


def _usable_secret(value: str | None) -> str | None:
    if not value or value.startswith("${") or value.startswith("YOUR_"):
        return None
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


# Convenience alias
settings = get_settings()
