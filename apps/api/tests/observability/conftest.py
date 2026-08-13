"""Auto-mark every test under tests/observability/ with the ``observability`` marker.

Phase 9 ships 4 test modules (tracing + metrics + exporters + e2e).
These tests are opt-in via the ``observability`` marker;
``pyproject.toml``'s ``addopts`` skips them from the default suite.
This conftest ensures every test in this directory is marked without
requiring an explicit decorator on each function.

Fixtures
--------

  * ``tracer_provider`` — installs an :class:`InMemorySpanExporter`
    behind a fresh ``TracerProvider`` and resets all module-level
    tracer caches. Yields the provider so tests can pull the
    ``captured_spans`` list directly. Provider is shut down at the
    end of the test to restore the OTel default state.

  * ``meter_provider`` — installs an :class:`InMemoryMetricReader`
    behind a fresh ``MeterProvider`` and resets the metric module
    instruments. Yields the reader so tests can introspect
    ``captured_metrics`` directly.

  * ``captured_spans`` — convenience: list of all finished spans
    recorded by the in-memory exporter.

  * ``captured_metrics`` — convenience: list of
    :class:`data_point.Sequence` / :class:`data_point.Sum` /
    :class:`data_point.Histogram` from the in-memory reader.
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-mark tests under tests/observability/ as ``observability``."""
    for item in items:
        if "tests/observability" in str(item.fspath):
            item.add_marker(pytest.mark.observability)


@pytest.fixture
def tracer_provider():
    """Install a TracerProvider backed by an InMemorySpanExporter.

    Returns the provider; tests can pull
    ``provider._active_span_processor._span_exporter._finished_spans``
    or use the :func:`captured_spans` fixture for the same data.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create({"service.name": "claim2code-otel-test"}),
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Reset the Once guard so ``set_tracer_provider`` actually
    # swaps in this test's provider (otherwise the second call in
    # a session silently no-ops and spans go to the previous test's
    # already-shut-down provider).
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace.set_tracer_provider(provider)
    try:
        yield provider, exporter
    finally:
        provider.shutdown()


@pytest.fixture
def captured_spans(tracer_provider):
    """List of finished spans recorded during the test."""
    _provider, exporter = tracer_provider
    return exporter.get_finished_spans()


@pytest.fixture
def meter_provider():
    """Install a MeterProvider backed by an InMemoryMetricReader.

    Also resets the cached metric instruments in
    ``claim2code_api.observability.meter`` so they bind to this
    fresh provider.
    """
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.resources import Resource

    from claim2code_api.observability.meter import reset_metrics_for_tests

    reset_metrics_for_tests()
    reader = InMemoryMetricReader()
    provider = MeterProvider(
        resource=Resource.create({"service.name": "claim2code-otel-test"}),
        metric_readers=[reader],
    )
    # Bypass the Once guard on ``set_meter_provider``. The Once
    # records "set" so a second call within a session silently
    # no-ops — which is fine for production but useless for tests
    # (each test wants a fresh provider). Resetting ``_done`` lets
    # ``set_meter_provider`` actually swap in the test provider.
    from opentelemetry.metrics import _internal as _otel_metrics_internal
    from opentelemetry.metrics import set_meter_provider

    _otel_metrics_internal._METER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    set_meter_provider(provider)
    try:
        yield provider, reader
    finally:
        provider.shutdown()
        reset_metrics_for_tests()


@pytest.fixture
def captured_metrics(meter_provider):
    """List of data points recorded during the test.

    Pulls metrics via :meth:`InMemoryMetricReader.get_metrics_data()`.
    Each item is a :class:`opentelemetry.sdk.metrics.export.MetricsData`
    containing per-instrument data points.
    """
    _provider, reader = meter_provider
    return reader.get_metrics_data()