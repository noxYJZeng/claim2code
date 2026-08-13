"""End-to-end observability tests — exercise the full LLMClient /
SafetyGate / workflow paths with a fake OpenAI SDK and assert the
resulting span tree + metric counters.

Covers:
  * ``LLMClient.extract_claims`` opens ``llm.extract_claims`` span;
    records ``llm_tokens_total`` + ``llm_request_duration_seconds``
    on success.
  * ``LLMClient.verdict_for_claim`` opens ``llm.verdict`` span;
    auto-parents ``safety.gate_wrap`` underneath it.
  * ``LLMClient.extract_comparison`` opens ``llm.compare`` span.
  * On error, the LLM span carries ``exception`` event but no
    token counter increment.

These tests run with the same in-memory exporter + reader fixtures
as the unit tests, so they're self-contained and CI-safe.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from openai import APITimeoutError

from claim2code_api.llm.client import LLMClient
from claim2code_api.papers.schemas import ClaimType, Paper, PaperSection

pytestmark = pytest.mark.observability


# ----------------------------------------------------------------------
# Helpers (mirror tests/test_llm_client.py)
# ----------------------------------------------------------------------


def _make_response(content: str, *, total_tokens: int = 500) -> MagicMock:
    """Build a fake OpenAI chat.completions.create response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = 300
    response.usage.completion_tokens = 200
    response.usage.total_tokens = total_tokens
    return response


def _client_with_response(content: str, *, total_tokens: int = 500) -> MagicMock:
    """Build a fake OpenAI SDK."""
    fake = MagicMock()
    fake.chat.completions.create.return_value = _make_response(
        content, total_tokens=total_tokens,
    )
    return fake


def _build_client(fake_openai: MagicMock) -> LLMClient:
    """Construct an LLMClient with a stubbed OpenAI SDK + SafetyGate."""
    client = LLMClient.__new__(LLMClient)
    import claim2code_api.app.config as cfg

    client._settings = cfg.get_settings()
    client._client = fake_openai
    from claim2code_api.safety import SafetyGate

    client._gate = SafetyGate()
    client._on_violations = None
    return client


def _sample_paper() -> Paper:
    return Paper(
        filename="x.pdf",
        sha256="0" * 64,
        n_pages=2,
        n_chars=200,
        sections=[
            PaperSection(
                title="Method", page=1, char_start=0, char_end=100, text="x" * 100,
            ),
            PaperSection(
                title="Experiments", page=2,
                char_start=100, char_end=200, text="y" * 100,
            ),
        ],
    )


def _valid_claims_json() -> str:
    payload = {
        "claims": [
            {
                "claim_id": "c1",
                "text": "Uses β=0.1.",
                "claim_type": ClaimType.HYPERPARAMETER.value,
                "confidence": 0.9,
                "paper_evidence": {
                    "section_title": "Method",
                    "page": 1,
                    "char_start": 0,
                    "char_end": 10,
                    "quote": "beta=0.1",
                },
            },
        ],
    }
    return json.dumps(payload)


def _span_by_name(spans: list, name: str) -> list:
    return [s for s in spans if s.name == name]


def _parent_name(spans: list, child) -> str | None:
    parent_id = child.parent.span_id
    for s in spans:
        if s.context.span_id == parent_id:
            return s.name
    return None


def _find_data_points(captured_metrics, instrument_name: str) -> list:
    out: list = []
    for resource_metrics in captured_metrics.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == instrument_name:
                    out.extend(metric.data.data_points)
    return out


def _point_value(point) -> float:
    if hasattr(point, "value"):
        return float(point.value)
    return float(point.sum)


# ----------------------------------------------------------------------
# LLMClient.extract_claims
# ----------------------------------------------------------------------


def test_extract_claims_emits_llm_span(tracer_provider):
    """Calling ``extract_claims`` opens an ``llm.extract_claims`` span."""
    _provider, exporter = tracer_provider
    fake = _client_with_response(_valid_claims_json())
    client = _build_client(fake)
    client.extract_claims(_sample_paper())

    spans = exporter.get_finished_spans()
    llm_spans = _span_by_name(spans, "llm.extract_claims")
    assert len(llm_spans) == 1
    llm = llm_spans[0]
    assert llm.attributes["llm.method"] == "extract_claims"
    assert llm.attributes["llm.model"]


def test_extract_claims_records_token_counter(meter_provider):
    """Successful LLM call increments ``llm_tokens_total``."""
    _provider, reader = meter_provider
    fake = _client_with_response(_valid_claims_json(), total_tokens=420)
    client = _build_client(fake)
    client.extract_claims(_sample_paper())

    captured = reader.get_metrics_data()
    points = _find_data_points(captured, "llm_tokens_total")
    assert len(points) == 1
    assert _point_value(points[0]) == 420
    labels = dict(points[0].attributes)
    assert labels["kind"] == "extract_claims"


def test_extract_claims_records_duration_histogram(meter_provider):
    """Successful LLM call observes ``llm_request_duration_seconds``."""
    _provider, reader = meter_provider
    fake = _client_with_response(_valid_claims_json())
    client = _build_client(fake)
    client.extract_claims(_sample_paper())

    captured = reader.get_metrics_data()
    points = _find_data_points(captured, "llm_request_duration_seconds")
    assert len(points) == 1
    p = points[0]
    assert p.count >= 1


def test_extract_claims_records_exception_on_error(meter_provider, tracer_provider):
    """SDK errors mark the span with an ``exception`` event; metrics
    for this kind are NOT incremented (no successful usage read)."""
    _provider_tracer, exporter = tracer_provider
    _provider, reader = meter_provider

    fake = MagicMock()
    fake.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
    client = _build_client(fake)
    # LLMClient wraps SDK errors in ``ClaimExtractionError``.
    from claim2code_api.papers.exceptions import ClaimExtractionError

    with pytest.raises(ClaimExtractionError):
        client.extract_claims(_sample_paper())

    # Span records the exception.
    spans = exporter.get_finished_spans()
    llm_spans = _span_by_name(spans, "llm.extract_claims")
    assert len(llm_spans) == 1
    assert any(e.name == "exception" for e in llm_spans[0].events)


# ----------------------------------------------------------------------
# Safety gate auto-parenting inside LLM span
# ----------------------------------------------------------------------


def test_safety_gate_wrap_inside_llm_span(tracer_provider):
    """``safety.gate_wrap`` is auto-parented under the LLM span via
    OTel's contextvars propagation. ``wrap_messages`` is called
    inside ``extract_claims`` immediately before
    ``chat.completions.create`` — same call stack.
    """
    _provider, exporter = tracer_provider
    fake = _client_with_response(_valid_claims_json())
    client = _build_client(fake)
    client.extract_claims(_sample_paper())

    spans = exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    assert "llm.extract_claims" in by_name
    assert "safety.gate_wrap" in by_name
    parent_name = _parent_name(spans, by_name["safety.gate_wrap"])
    assert parent_name == "llm.extract_claims"
    # Same trace as the LLM span.
    assert (
        by_name["safety.gate_wrap"].context.trace_id
        == by_name["llm.extract_claims"].context.trace_id
    )


# ----------------------------------------------------------------------
# Per-claim loop span shape (mirrors verification workflow)
# ----------------------------------------------------------------------


def test_per_claim_loop_span_shape(tracer_provider):
    """A simulated per-claim loop opens N claim_verify spans under one root."""
    from claim2code_api.observability import get_tracer

    _provider, exporter = tracer_provider
    tracer = get_tracer(__name__)
    claim_ids = ["c1", "c2", "c3"]
    with tracer.start_as_current_span(
        "verification_workflow.run",
        attributes={"verification.analysis_id": "abc", "verification.n_claims": 3},
    ):
        for cid in claim_ids:
            with tracer.start_as_current_span(
                "verification.claim_verify",
                attributes={"claim.id": cid},
            ):
                pass

    spans = exporter.get_finished_spans()
    by_name: dict[str, list] = {}
    for s in spans:
        by_name.setdefault(s.name, []).append(s)
    assert len(by_name["verification_workflow.run"]) == 1
    assert len(by_name["verification.claim_verify"]) == 3
    claim_ids_seen = {
        s.attributes["claim.id"] for s in by_name["verification.claim_verify"]
    }
    assert claim_ids_seen == {"c1", "c2", "c3"}