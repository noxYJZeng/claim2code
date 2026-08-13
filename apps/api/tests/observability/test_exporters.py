"""Exporter-wiring tests — verifies ``setup_tracing`` / ``setup_metrics``
honour the 3 pre-staged OTel Settings fields.

Covers:
  * Console exporter wired when ``otel_exporter_console=True``
  * OTLP exporter wired when ``otel_exporter_otlp_endpoint`` is set
  * Both exporters can coexist
  * No exporters = idempotent no-op (CI default)
  * Idempotent: a second ``setup_tracing(...)`` call is a no-op
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from claim2code_api.app.config import Settings

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider


pytestmark = pytest.mark.observability


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _providers(trace_provider) -> list:
    """Return the list of span processors on ``trace_provider``."""
    return list(trace_provider._active_span_processor._span_processors)


def _has_console_exporter(trace_provider) -> bool:
    """True iff the provider has at least one ConsoleSpanExporter."""
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter

    for sp in _providers(trace_provider):
        if isinstance(getattr(sp, "span_exporter", None), ConsoleSpanExporter):
            return True
    return False


def _has_otlp_exporter(trace_provider) -> bool:
    """True iff the provider has at least one OTLPSpanExporter."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    for sp in _providers(trace_provider):
        if isinstance(getattr(sp, "span_exporter", None), OTLPSpanExporter):
            return True
    return False


@pytest.fixture
def base_settings() -> Settings:
    """A Settings instance with all 3 OTel fields reset to defaults."""
    from claim2code_api.app.config import get_settings

    settings = get_settings()
    return settings.model_copy(
        update={
            "otel_service_name": "claim2code-test",
            "otel_exporter_console": True,
            "otel_exporter_otlp_endpoint": "",
        },
    )


# ----------------------------------------------------------------------
# Console exporter wiring
# ----------------------------------------------------------------------


def test_console_exporter_wired_when_flag_true(base_settings):
    """``otel_exporter_console=True`` → ConsoleSpanExporter is added."""
    from claim2code_api.observability import setup_tracing

    provider = setup_tracing(base_settings, force=True)
    assert provider is not None
    assert _has_console_exporter(provider)


def test_console_exporter_omitted_when_flag_false(base_settings):
    """``otel_exporter_console=False`` + empty endpoint → no exporters."""
    from claim2code_api.observability import setup_tracing

    settings = base_settings.model_copy(update={"otel_exporter_console": False})
    provider = setup_tracing(settings, force=True)
    assert provider is not None
    assert not _has_console_exporter(provider)
    assert not _has_otlp_exporter(provider)


# ----------------------------------------------------------------------
# OTLP exporter wiring
# ----------------------------------------------------------------------


def test_otlp_exporter_wired_when_endpoint_set(base_settings):
    """Non-empty ``otel_exporter_otlp_endpoint`` → OTLPSpanExporter added."""
    from claim2code_api.observability import setup_tracing

    settings = base_settings.model_copy(
        update={
            "otel_exporter_console": False,
            "otel_exporter_otlp_endpoint": "https://otel.example.com/v1/traces",
        },
    )
    provider = setup_tracing(settings, force=True)
    assert provider is not None
    assert _has_otlp_exporter(provider)


# ----------------------------------------------------------------------
# Both exporters
# ----------------------------------------------------------------------


def test_both_exporters_can_coexist(base_settings):
    """``otel_exporter_console=True`` + endpoint → both exporters added."""
    from claim2code_api.observability import setup_tracing

    settings = base_settings.model_copy(
        update={
            "otel_exporter_console": True,
            "otel_exporter_otlp_endpoint": "https://otel.example.com/v1/traces",
        },
    )
    provider = setup_tracing(settings, force=True)
    assert provider is not None
    assert _has_console_exporter(provider)
    assert _has_otlp_exporter(provider)


# ----------------------------------------------------------------------
# Idempotency
# ----------------------------------------------------------------------


def test_setup_tracing_is_idempotent(base_settings):
    """A second ``setup_tracing(...)`` call returns ``None`` (no-op)."""
    from claim2code_api.observability import setup_tracing

    first = setup_tracing(base_settings, force=True)
    assert first is not None
    # Without ``force``, the second call short-circuits.
    second = setup_tracing(base_settings)
    assert second is None


def test_setup_tracing_force_replaces_provider(base_settings):
    """``force=True`` lets tests swap providers cleanly."""
    from claim2code_api.observability import setup_tracing

    first = setup_tracing(base_settings, force=True)
    second = setup_tracing(base_settings, force=True)
    assert first is not None
    assert second is not None
    assert first is not second


# ----------------------------------------------------------------------
# Service resource
# ----------------------------------------------------------------------


def test_resource_carries_service_name(base_settings):
    """``otel_service_name`` is propagated onto the OTel Resource."""
    from claim2code_api.observability import setup_tracing

    settings = base_settings.model_copy(
        update={"otel_service_name": "claim2code-custom"},
    )
    provider = setup_tracing(settings, force=True)
    assert provider is not None
    name = provider.resource.attributes.get("service.name")
    assert name == "claim2code-custom"


# ----------------------------------------------------------------------
# Metric exporter wiring (mirror for meter_provider)
# ----------------------------------------------------------------------


def test_metric_console_exporter_wired(base_settings):
    """``otel_exporter_console=True`` → MeterProvider has ConsoleMetricReader."""
    from opentelemetry.metrics import get_meter_provider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

    from claim2code_api.observability import setup_metrics

    provider = setup_metrics(base_settings, force=True)
    assert provider is not None
    assert isinstance(get_meter_provider(), MeterProvider)
    # ``_metric_readers`` is the public-private attribute set in
    # the constructor; readers each wrap a single exporter in
    # ``_exporter``.
    assert any(
        isinstance(getattr(r, "_exporter", None), ConsoleMetricExporter)
        for r in provider._metric_readers
    )