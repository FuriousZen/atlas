from app.agent.graphs import run_ingestion
from app.ingest.sources import CandidatePaper
from app.tools.llm_adapter import get_llm


def candidate(identifier: str, citations: int) -> CandidatePaper:
    return CandidatePaper(id=identifier, title=f"Paper {identifier}", authors=["A"], venue="Journal", year=2025, citations=citations, topic="retrieval", url=f"https://example.invalid/{identifier}", abstract="A paper about retrieval and evidence.")


def test_threshold_happens_before_summary():
    result = run_ingestion([candidate("low", 2), candidate("high", 100)], "retrieval", 50)
    assert result["polled"] == 2
    assert result["kept"] == 1
    assert result["dropped_threshold"] == 1
    assert result["indexed"] == 1


def test_provider_defaults_to_stub_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from app.tools.llm_adapter import _StubProvider

    assert isinstance(get_llm("stub"), _StubProvider)


def test_custom_provider_can_be_plugged_in_without_graph_changes():
    from app.tools.llm_adapter import LlmResult, get_llm, register_provider

    class LocalGateway:
        def generate(self, prompt, response_schema=None, temperature=0.2):
            return LlmResult(text='{}', model="local-gateway", elapsed_ms=1, backend="local-gateway")

    register_provider("local-gateway", LocalGateway)
    assert get_llm("local-gateway").generate("test").model == "local-gateway"
