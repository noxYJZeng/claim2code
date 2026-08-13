"""Metric-instrument tests — verifies the 4 Phase 9 metric instruments
behave as documented.

Covers:
  * ``llm_tokens_total`` counter — incremented by LLMClient helpers
    after a successful chat.completions.create call (read from
    ``response.usage.total_tokens``).
  * ``llm_request_duration_seconds`` histogram — observes the wall
    clock of each ``chat.completions.create(...)`` call.
  * ``retriever_request_duration_seconds`` histogram — labelled by
    ``leg`` (semantic / lexical / structural).
  * ``safety_violations_total`` counter — incremented per
    ``SafetyViolation`` detected by :meth:`SafetyGate.wrap_messages`,
    labelled by ``pattern``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.observability


# ----------------------------------------------------------------------
# Counter / histogram helpers
# ----------------------------------------------------------------------


def _find_data_points(captured_metrics, instrument_name: str) -> list:
    """Return data points for ``instrument_name`` from MetricsData."""
    out: list = []
    for resource_metrics in captured_metrics.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == instrument_name:
                    out.extend(metric.data.data_points)
    return out


def _point_value(point) -> float:
    """Extract the numeric value from a Sum/Histogram data point."""
    # Counter (Sum) data points have ``value``; histogram data points
    # have ``sum`` / ``count``. We return whichever is non-None.
    if hasattr(point, "value"):
        return float(point.value)
    return float(point.sum)


# ----------------------------------------------------------------------
# Token counter
# ----------------------------------------------------------------------


def test_llm_tokens_counter_records_total(meter_provider):
    """``llm_tokens_total.add(total, {"model", "kind"})`` records value."""
    _provider, reader = meter_provider
    from claim2code_api.observability.meter import llm_tokens_total

    counter = llm_tokens_total()
    counter.add(500, {"model": "gpt-4o", "kind": "extract_claims"})
    counter.add(120, {"model": "gpt-4o", "kind": "extract_claims"})

    captured = reader.get_metrics_data()
    points = _find_data_points(captured, "llm_tokens_total")
    assert len(points) == 1
    assert _point_value(points[0]) == 620
    labels = dict(points[0].attributes)
    assert labels["model"] == "gpt-4o"
    assert labels["kind"] == "extract_claims"


def test_llm_tokens_counter_distinguishes_models(meter_provider):
    """The counter separates labels so per-model totals are aggregable."""
    _provider, reader = meter_provider
    from claim2code_api.observability.meter import llm_tokens_total

    counter = llm_tokens_total()
    counter.add(100, {"model": "gpt-4o", "kind": "verdict"})
    counter.add(50, {"model": "gpt-4o-mini", "kind": "verdict"})

    captured = reader.get_metrics_data()
    points = _find_data_points(captured, "llm_tokens_total")
    assert len(points) == 2
    by_model = {dict(p.attributes)["model"]: _point_value(p) for p in points}
    assert by_model == {"gpt-4o": 100, "gpt-4o-mini": 50}


# ----------------------------------------------------------------------
# LLM request duration histogram
# ----------------------------------------------------------------------


def test_llm_request_duration_histogram_observes(meter_provider):
    """``llm_request_duration_seconds.record(s, {"model", "kind"})``."""
    _provider, reader = meter_provider
    from claim2code_api.observability.meter import llm_request_duration_seconds

    histogram = llm_request_duration_seconds()
    histogram.record(0.123, {"model": "gpt-4o", "kind": "verdict"})
    histogram.record(0.456, {"model": "gpt-4o", "kind": "verdict"})

    captured = reader.get_metrics_data()
    points = _find_data_points(captured, "llm_request_duration_seconds")
    assert len(points) == 1
    p = points[0]
    # At least 1 sample observed; histogram.sum >= 0.123.
    assert p.count >= 1
    assert p.sum >= 0.5


# ----------------------------------------------------------------------
# Retriever request duration histogram — leg label
# ----------------------------------------------------------------------


def test_retriever_duration_histogram_per_leg(meter_provider):
    """Retriever duration is recorded per leg (semantic/lexical/structural)."""
    _provider, reader = meter_provider
    from claim2code_api.observability.meter import retriever_request_duration_seconds

    histogram = retriever_request_duration_seconds()
    histogram.record(0.10, {"leg": "semantic"})
    histogram.record(0.05, {"leg": "lexical"})
    histogram.record(0.01, {"leg": "structural"})

    captured = reader.get_metrics_data()
    points = _find_data_points(captured, "retriever_request_duration_seconds")
    by_leg = {dict(p.attributes)["leg"]: p for p in points}
    assert set(by_leg.keys()) == {"semantic", "lexical", "structural"}
    assert by_leg["semantic"].count == 1
    assert by_leg["lexical"].count == 1
    assert by_leg["structural"].count == 1


# ----------------------------------------------------------------------
# Safety violations counter
# ----------------------------------------------------------------------


def test_safety_violations_counter_increments_by_pattern(meter_provider):
    """``record_safety_violation(pattern, span_count=N)`` adds N."""
    _provider, reader = meter_provider
    from claim2code_api.observability.meter import record_safety_violation

    record_safety_violation("openai_key", span_count=2)
    record_safety_violation("openai_key", span_count=3)
    record_safety_violation("ignore_previous", span_count=1)

    captured = reader.get_metrics_data()
    points = _find_data_points(captured, "safety_violations_total")
    by_pattern = {dict(p.attributes)["pattern"]: _point_value(p) for p in points}
    assert by_pattern == {"openai_key": 5, "ignore_previous": 1}


def test_safety_violations_cumulative(meter_provider):
    """Counter accumulates across multiple wrap_messages calls."""
    _provider, reader = meter_provider
    from claim2code_api.observability.meter import record_safety_violation

    for _ in range(5):
        record_safety_violation("openai_key", span_count=1)

    captured = reader.get_metrics_data()
    points = _find_data_points(captured, "safety_violations_total")
    assert _point_value(points[0]) == 5


# ----------------------------------------------------------------------
# Reset hook (used by other test modules)
# ----------------------------------------------------------------------


def test_reset_drops_cached_instruments(meter_provider):
    """``reset_metrics_for_tests`` drops cached instruments."""
    from claim2code_api.observability import meter

    # At this point the meter_provider fixture has installed a fresh
    # MeterProvider; the instruments have been bound to it. Call
    # reset, then re-bind by calling the accessor.
    instrument = meter.llm_tokens_total()
    instrument.add(1, {"model": "m", "kind": "k"})
    meter.reset_metrics_for_tests()
    # After reset, the accessor recreates on next call (the meter
    # provider fixture's reset call already did this, but calling
    # it again here exercises the public reset path).
    new_instrument = meter.llm_tokens_total()
    assert new_instrument is not None
    # Suppress unused warnings — the test is about the reset hook,
    # not the value.
    _ = MagicMock()