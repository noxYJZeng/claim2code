"""OpenTelemetry tracing + metrics for Claim2Code (Phase 9).

Every production LLM call, retriever call, and workflow call is
instrumented with a span; cost (LLM tokens), latency (LLM +
retriever), and safety-violation counts are emitted as metrics.

Public surface
--------------

  - :func:`setup_tracing` / :func:`setup_metrics` — call once at
    FastAPI startup (``lifespan``). Idempotent (safe to call from
    tests).
  - :func:`get_tracer` / :func:`get_meter` — lazy-init helpers for
    span + metric creation; tests monkeypatch them.
  - The metric instruments live in :mod:`claim2code_api.observability.meter`
    and are exposed by the same name. Tests reset them with
    :func:`reset_metrics_for_tests`.

Exporter routing
----------------

Configuration is env-driven via the 3 pre-staged Settings fields:

  - ``otel_exporter_console=True`` (default) → stdout exporter
  - ``otel_exporter_otlp_endpoint="..."`` (non-empty) → OTLP-HTTP exporter
  - Both can coexist (just two ``SpanProcessor``s on the same
    ``TracerProvider``). Empty endpoint + console=False → no-op
    tracer (CI default; tests use ``InMemorySpanExporter`` instead).

See ``docs/architecture/phase-9-design.md`` for the full design.
"""
from __future__ import annotations

from claim2code_api.observability.meter import (
    get_meter,
    llm_request_duration_seconds,
    llm_tokens_total,
    record_safety_violation,
    reset_metrics_for_tests,
    retriever_request_duration_seconds,
    safety_violations_total,
)
from claim2code_api.observability.setup import setup_metrics, setup_tracing
from claim2code_api.observability.tracer import get_tracer

__all__ = [
    "setup_tracing",
    "setup_metrics",
    "get_tracer",
    "get_meter",
    "llm_tokens_total",
    "llm_request_duration_seconds",
    "retriever_request_duration_seconds",
    "safety_violations_total",
    "record_safety_violation",
    "reset_metrics_for_tests",
]