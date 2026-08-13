"""Metric instruments for Phase 9 observability.

Three metric families (matching the design doc §1):

  1. ``llm_tokens_total`` — counter, unit=tokens; labelled by
     ``model`` + ``kind``. Incremented after every successful
     OpenAI call from ``response.usage.total_tokens``.

  2. ``llm_request_duration_seconds`` — histogram, unit=s;
     labelled by ``model`` + ``kind``. Records the wall time of
     each ``chat.completions.create(...)`` call.

  3. ``retriever_request_duration_seconds`` — histogram, unit=s;
     labelled by ``leg`` (``semantic`` / ``lexical`` /
     ``structural``). One observation per hybrid-ranker sub-call.

  4. ``safety_violations_total`` — counter; labelled by
     ``pattern`` (``openai_key`` / ``ignore_previous`` / ...).
     Incremented for each ``SafetyViolation`` the gate produces.

Instruments are created lazily on first call to
:func:`get_meter`. They share the global ``MeterProvider`` set by
:func:`setup_metrics`. Tests reset state via
:func:`reset_metrics_for_tests`.

We deliberately use ``kind`` as a label (NOT ``claim_id`` /
``paper_id``) — those values are unbounded and would explode
metric cardinality. Per-claim / per-paper context lives on the
**span** attributes, where cardinality is bounded by request size.
"""
from __future__ import annotations

from typing import Any

from opentelemetry.metrics import Counter, Histogram, Meter
from opentelemetry.metrics import get_meter as _otel_get_meter

_METER_NAME = "claim2code"

# Module-level references. Populated by ``_ensure_instruments`` on
# first call. Tests use ``reset_metrics_for_tests`` to force
# recreation after monkeypatching the provider.
_llm_tokens_total: Counter | None = None
_llm_request_duration_seconds: Histogram | None = None
_retriever_request_duration_seconds: Histogram | None = None
_safety_violations_total: Counter | None = None


def get_meter() -> Meter:
    """Return the package's Meter (configured by ``setup_metrics``)."""
    return _otel_get_meter(_METER_NAME)


def _ensure_instruments() -> tuple[Counter, Histogram, Histogram, Counter]:
    """Lazily create all instruments. Idempotent within a single run."""
    global _llm_tokens_total, _llm_request_duration_seconds
    global _retriever_request_duration_seconds, _safety_violations_total
    if _llm_tokens_total is None:
        meter = get_meter()
        _llm_tokens_total = meter.create_counter(
            "llm_tokens_total",
            description="Total LLM tokens consumed",
            unit="tokens",
        )
    if _llm_request_duration_seconds is None:
        meter = get_meter()
        _llm_request_duration_seconds = meter.create_histogram(
            "llm_request_duration_seconds",
            description="LLM SDK call duration",
            unit="s",
        )
    if _retriever_request_duration_seconds is None:
        meter = get_meter()
        _retriever_request_duration_seconds = meter.create_histogram(
            "retriever_request_duration_seconds",
            description="Hybrid retriever leg duration",
            unit="s",
        )
    if _safety_violations_total is None:
        meter = get_meter()
        _safety_violations_total = meter.create_counter(
            "safety_violations_total",
            description="Safety violations by pattern",
        )
    return (
        _llm_tokens_total,
        _llm_request_duration_seconds,
        _retriever_request_duration_seconds,
        _safety_violations_total,
    )


def llm_tokens_total() -> Counter:
    """Counter: total LLM tokens consumed (labelled model + kind)."""
    return _ensure_instruments()[0]


def llm_request_duration_seconds() -> Histogram:
    """Histogram: LLM SDK call duration (seconds, labelled model + kind)."""
    return _ensure_instruments()[1]


def retriever_request_duration_seconds() -> Histogram:
    """Histogram: hybrid retriever leg duration (seconds, labelled leg)."""
    return _ensure_instruments()[2]


def safety_violations_total() -> Counter:
    """Counter: safety violations (labelled pattern)."""
    return _ensure_instruments()[3]


def record_safety_violation(pattern_label: str, span_count: int = 1) -> None:
    """Increment the safety violation counter for one ``SafetyViolation``.

    Convenience helper so call sites can do:

        record_safety_violation(
            pattern_label=v.pattern_label,
            span_count=v.span_count,
        )

    instead of poking the counter directly. Adds ``span_count`` to
    the counter (so one ``SafetyViolation`` covering N matched spans
    increments by N).
    """
    counter = safety_violations_total()
    counter.add(span_count, {"pattern": pattern_label})


def reset_metrics_for_tests() -> None:
    """Drop cached instruments so the next call recreates them.

    Tests call this after monkeypatching the global
    ``MeterProvider`` (e.g. installing an
    :class:`InMemoryMetricReader`). The instruments are bound to a
    provider at creation time, so the old references would record
    into the previous provider — drop them.
    """
    global _llm_tokens_total, _llm_request_duration_seconds
    global _retriever_request_duration_seconds, _safety_violations_total
    _llm_tokens_total = None
    _llm_request_duration_seconds = None
    _retriever_request_duration_seconds = None
    _safety_violations_total = None


__all__ = [
    "get_meter",
    "llm_tokens_total",
    "llm_request_duration_seconds",
    "retriever_request_duration_seconds",
    "safety_violations_total",
    "record_safety_violation",
    "reset_metrics_for_tests",
    # Re-export the Any type so call sites can type-annotate
    # ``attributes: dict[str, Any]`` without an extra import.
    "Any",
]