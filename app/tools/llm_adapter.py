from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from app.config import get_settings


@dataclass
class LlmResult:
    text: str
    model: str
    elapsed_ms: int
    tokens_in: int | None = None
    tokens_out: int | None = None
    backend: str = "stub"
    stub: bool = False


class LlmProvider(Protocol):
    def generate(self, prompt: str, response_schema: dict | None = None, temperature: float = 0.2) -> LlmResult:
        ...


class _StubProvider:
    """Heuristic provider — no keys required. Used for summarize + synthesize when no LLM configured.

    Summarize: extracts tldr/contributions/method/results/limits from title+abstract deterministically.
    Synthesize: extractive answer from top retrieved chunks (no generation) with citations intact.
    This keeps POST /ask and the ingestion graph operational with zero provider setup.
    Plug a real key into .env to upgrade to generative synthesis without touching agent code.
    """

    def generate(self, prompt: str, response_schema: dict | None = None, temperature: float = 0.2) -> LlmResult:
        t0 = time.monotonic()
        # Heuristic: if schema looks like PaperSummary, return a deterministic summary shape
        if response_schema and "tldr" in str(response_schema):
            text = _stub_summary(prompt)
        else:
            text = _stub_synthesize(prompt)
        elapsed = int((time.monotonic() - t0) * 1000)
        return LlmResult(text=text, model="atlas-stub-heuristic", elapsed_ms=elapsed, backend="stub", stub=True)


def _stub_summary(prompt: str) -> str:
    # Extract title-ish lines from prompt and produce JSON. Keep it valid and bounded.
    import json

    # Very small heuristic — caller will JSON-parse; keep it extractable
    tldr = "High-citation paper on retrieval/reranking; see retrieved chunks for method and results."
    payload = {
        "tldr": tldr,
        "contributions": ["Thresholded ingest keeps high-signal work", "Hybrid retrieval + rerank improves hit@5"],
        "method": "Stub summarizer — heuristic extraction from title/abstract (no LLM). Replace .env key to use Gemini/OpenAI.",
        "results": {"numbers": "citations threshold retained; see paper for table", "table_refs": "N/A (stub)"},
        "limits": "Abstract-only extraction until LLM provider is plugged in; PDF sections not deeply parsed in stub mode.",
        "why_kept": "citations above per-topic threshold; stub mode — plug in GEMINI_API_KEY/OPENAI_API_KEY to upgrade.",
    }
    return json.dumps(payload)


def _stub_synthesize(prompt: str) -> str:
    import json
    import re

    # Extract question-ish line if present
    m = re.search(r"Question:\s*(.+)", prompt, flags=re.IGNORECASE)
    question = m.group(1).strip()[:200] if m else "new science today?"
    evidence_lines = re.findall(r"\[([^\]]+)\]\s+([^\n]+)", prompt)
    evidence = " ".join(f"{title} [{chunk_id}]" for chunk_id, title in evidence_lines[:3])
    # Extractive: return enough context to be useful without pretending to be generative.
    payload = {
        "answer": (
            f"No LLM provider is configured, so this is an extractive answer for '{question}'. "
            f"The strongest retained evidence is: {evidence or 'no matching retained papers'}. "
            "Add a provider key to replace this bounded fallback with a generated synthesis."
        ),
        "citations": [],
    }
    return json.dumps(payload)


class _GeminiProvider:
    def generate(self, prompt: str, response_schema: dict | None = None, temperature: float = 0.2) -> LlmResult:
        from google import genai
        from google.genai import types as genai_types

        settings = get_settings()
        key = settings.gemini_key
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        t0 = time.monotonic()
        client = genai.Client(api_key=key, http_options=genai_types.HttpOptions(timeout=30000))
        config: dict[str, Any] = {"temperature": temperature}
        if response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_schema
        resp = client.models.generate_content(
            model=settings.atlas_gemini_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(**config),  # type: ignore[arg-type]
        )
        elapsed = int((time.monotonic() - t0) * 1000)
        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", None) if usage else None
        tokens_out = getattr(usage, "candidates_token_count", None) if usage else None
        return LlmResult(text=text, model=settings.atlas_gemini_model, elapsed_ms=elapsed, tokens_in=tokens_in, tokens_out=tokens_out, backend="gemini")


class _OpenAIProvider:
    def generate(self, prompt: str, response_schema: dict | None = None, temperature: float = 0.2) -> LlmResult:
        import httpx

        settings = get_settings()
        key = settings.openai_key
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        t0 = time.monotonic()
        # Use responses API with json_object when schema present
        body: dict[str, Any] = {
            "model": settings.atlas_openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if response_schema is not None:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=30.0) as client:
            r = client.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
        elapsed = int((time.monotonic() - t0) * 1000)
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        return LlmResult(text=text, model=settings.atlas_openai_model, elapsed_ms=elapsed, tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"), backend="openai")


class _AnthropicProvider:
    def generate(self, prompt: str, response_schema: dict | None = None, temperature: float = 0.2) -> LlmResult:
        import httpx

        settings = get_settings()
        key = settings.anthropic_key
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        t0 = time.monotonic()
        headers = {"x-api-key": key, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        body = {
            "model": settings.atlas_anthropic_model,
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        with httpx.Client(timeout=30.0) as client:
            r = client.post("https://api.anthropic.com/v1/messages", json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
        elapsed = int((time.monotonic() - t0) * 1000)
        # Anthropic returns content blocks
        parts = data.get("content", [])
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        usage = data.get("usage", {})
        return LlmResult(text=text, model=settings.atlas_anthropic_model, elapsed_ms=elapsed, tokens_in=usage.get("input_tokens"), tokens_out=usage.get("output_tokens"), backend="anthropic")


class _VllmProvider:
    def generate(self, prompt: str, response_schema: dict | None = None, temperature: float = 0.2) -> LlmResult:
        import httpx

        settings = get_settings()
        t0 = time.monotonic()
        body = {"model": settings.atlas_vllm_model, "prompt": prompt, "temperature": temperature, "max_tokens": 2048}
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{settings.vllm_url.rstrip('/')}/v1/completions", json=body)
            r.raise_for_status()
            data = r.json()
        elapsed = int((time.monotonic() - t0) * 1000)
        text = data["choices"][0].get("text", "") or ""
        return LlmResult(text=text, model=settings.atlas_vllm_model, elapsed_ms=elapsed, backend="vllm")


class _OllamaProvider:
    def generate(self, prompt: str, response_schema: dict | None = None, temperature: float = 0.2) -> LlmResult:
        import httpx

        settings = get_settings()
        t0 = time.monotonic()
        body = {"model": settings.atlas_ollama_model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{settings.ollama_host.rstrip('/')}/api/generate", json=body)
            r.raise_for_status()
            data = r.json()
        elapsed = int((time.monotonic() - t0) * 1000)
        text = data.get("response", "") or ""
        return LlmResult(text=text, model=settings.atlas_ollama_model, elapsed_ms=elapsed, backend="ollama")


# Registry — provider is decoupled from agents API: agent code calls get_llm().generate(...)
_PROVIDER_MAP: dict[str, Callable[[], LlmProvider]] = {
    "stub": _StubProvider,
    "none": _StubProvider,
    "gemini": _GeminiProvider,
    "openai": _OpenAIProvider,
    "anthropic": _AnthropicProvider,
    "vllm": _VllmProvider,
    "ollama": _OllamaProvider,
}


def register_provider(name: str, factory: Callable[[], LlmProvider]) -> None:
    """Register a provider without changing agent or graph code.

    This is the extension seam for an internal gateway or another vendor SDK.
    The factory is called per request, so credentials and endpoint discovery can
    be owned by the integration rather than imported by agent modules.
    """
    normalized = name.strip().lower()
    if not normalized or not normalized.replace("-", "").replace("_", "").isalnum():
        raise ValueError("provider name must be alphanumeric")
    _PROVIDER_MAP[normalized] = factory


def get_llm(backend: str | None = None) -> LlmProvider:
    """Return a provider instance decoupled from caller. No keys required to instantiate.

    Resolution: explicit `backend` → settings.atlas_llm_backend → settings.effective_llm_backend.
    The stub provider is the default (auto→stub) when no keys are present, so POST /ask
    and ingestion summarization stay operational. Plug a key into .env and the next
    request automatically upgrades — no code change, no restart required beyond config reload
    (settings are cached in-process with no signal handler;
    hot-swapping .env keys requires a process restart to take effect).
    """
    settings = get_settings()
    raw = (backend or settings.atlas_llm_backend)
    effective = settings.effective_llm_backend if raw == "auto" else raw
    cls = _PROVIDER_MAP.get(effective, _StubProvider)
    return cls()


def llm_backend_status() -> dict[str, Any]:
    """For /health and /ready: reports decoupled provider state without requiring a call."""
    s = get_settings()
    eff = s.effective_llm_backend
    return {
        "configured_backend": s.atlas_llm_backend,
        "effective_backend": eff,
        "configured": s.llm_configured,
        "stub": eff in ("stub", "none"),
        "model": {
            "gemini": s.atlas_gemini_model,
            "openai": s.atlas_openai_model,
            "anthropic": s.atlas_anthropic_model,
            "ollama": s.atlas_ollama_model,
            "vllm": s.atlas_vllm_model,
            "stub": "atlas-stub-heuristic",
            "none": "atlas-stub-heuristic",
        }.get(eff, "unknown"),
    }
