"""Tracer accessor — thin wrapper over ``opentelemetry.trace``.

Phase 9 design decision: keep ``get_tracer(name)`` as the canonical
accessor so the instrumented code stays the same whether we're
running with a console exporter, an OTLP exporter, an in-memory
exporter (tests), or no exporter at all (CI default).

Pattern (mirrors OTel docs):

    from claim2code_api.observability import get_tracer
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("llm.verdict", attributes={...}) as span:
        ...

``get_tracer`` is a thin call to :func:`opentelemetry.trace.get_tracer`
which itself returns the global tracer configured by
:func:`setup_tracing`. If ``setup_tracing`` was never called the
OTel SDK returns a ``ProxyTracer`` that becomes a real tracer on
first use — no special handling needed.
"""
from __future__ import annotations

from opentelemetry.trace import Tracer
from opentelemetry.trace import get_tracer as _otel_get_tracer

_TRACER_NAME = "claim2code"


def get_tracer(name: str | None = None) -> Tracer:
    """Return a Tracer scoped to ``name`` (or the package default).

    ``name`` is joined to the package's tracer name so spans from
    the same module cluster under one tracer in trace UIs. Tests
    monkeypatch ``opentelemetry.trace._TRACER_PROVIDER`` (or call
    ``opentelemetry.trace.set_tracer_provider``) before invoking
    instrumented code to redirect output to an
    :class:`InMemorySpanExporter`.
    """
    full_name = f"{_TRACER_NAME}.{name}" if name else _TRACER_NAME
    return _otel_get_tracer(full_name)


__all__ = ["get_tracer"]