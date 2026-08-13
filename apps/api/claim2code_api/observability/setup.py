"""OTel provider setup — call once at FastAPI startup (``lifespan``).

Both :func:`setup_tracing` and :func:`setup_metrics` are idempotent:
calling them more than once is a no-op (the global
``TracerProvider`` / ``MeterProvider`` is set only the first time
unless the caller passes ``force=True``). This matters because:

  - Tests can call ``setup_tracing(force=True)`` after installing an
    :class:`InMemorySpanExporter` to swap providers cleanly.
  - The lifespan is re-entered on hot reloads; we must not stack
    providers on every reload.

Env-driven configuration (read from the 3 pre-staged Settings
fields — see ``app/config.py:126-129``):

  - ``otel_service_name`` → ``Resource`` attribute ``service.name``
  - ``otel_exporter_console=True`` → add ``ConsoleSpanExporter``
  - ``otel_exporter_otlp_endpoint="https://..."`` → add
    ``OTLPSpanExporter`` (HTTP transport)
  - Both can coexist; an empty endpoint + console=False produces a
    no-op tracer (CI default; tests use InMemoryExporter instead).
"""
from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from claim2code_api.app.config import Settings

logger = logging.getLogger(__name__)

# Default export interval for the metric reader. 60s is the OTel
# default but we keep it explicit so production tuning is one
# constant change away.
_METRIC_EXPORT_INTERVAL_SECONDS = 60.0


def _build_resource(settings: Settings) -> Resource:
    """Build the OTel ``Resource`` carrying service identity."""
    return Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": "0.1.0",
        },
    )


def setup_tracing(settings: Settings, *, force: bool = False) -> TracerProvider | None:
    """Wire the global ``TracerProvider`` with console / OTLP exporters.

    :param settings: the process Settings (3 OTel fields used).
    :param force: if True, replace the existing provider even if
        one is already installed. Tests use this after monkeypatching
        the exporter.
    :returns: the installed ``TracerProvider`` (or ``None`` when no
        exporter is configured — a fresh ``TracerProvider`` with no
        processors still emits spans to a no-op destination; we
        return ``None`` so callers can detect "no exporter" state).
    """
    if not force and isinstance(trace.get_tracer_provider(), TracerProvider):
        logger.debug("TracerProvider already installed; skipping setup")
        return None

    resource = _build_resource(settings)
    provider = TracerProvider(resource=resource)

    has_console = settings.otel_exporter_console
    has_otlp = bool(settings.otel_exporter_otlp_endpoint)

    if has_console:
        # SimpleSpanProcessor writes synchronously; convenient for
        # local dev where seeing the span immediately matters.
        provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter()),
        )
        logger.info("OTel: ConsoleSpanExporter wired")

    if has_otlp:
        # Lazy import — keeps the OTLP exporter dep off the import
        # path of the dev cycle that doesn't enable OTLP.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint),
            ),
        )
        logger.info(
            "OTel: OTLP-HTTP exporter wired (endpoint=%s)",
            settings.otel_exporter_otlp_endpoint,
        )

    trace.set_tracer_provider(provider)
    return provider


def setup_metrics(
    settings: Settings, *, force: bool = False,
) -> MeterProvider | None:
    """Wire the global ``MeterProvider`` with console / OTLP exporters.

    Same idempotency / force semantics as :func:`setup_tracing`.
    """
    if not force and _has_meter_provider():
        logger.debug("MeterProvider already installed; skipping setup")
        return None

    resource = _build_resource(settings)
    readers: list[PeriodicExportingMetricReader] = []

    if settings.otel_exporter_console:
        readers.append(
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=int(_METRIC_EXPORT_INTERVAL_SECONDS * 1000),
            ),
        )
        logger.info("OTel: ConsoleMetricExporter wired")

    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )

        readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=settings.otel_exporter_otlp_endpoint),
                export_interval_millis=int(_METRIC_EXPORT_INTERVAL_SECONDS * 1000),
            ),
        )
        logger.info(
            "OTel: OTLP-HTTP metrics exporter wired (endpoint=%s)",
            settings.otel_exporter_otlp_endpoint,
        )

    provider = MeterProvider(resource=resource, metric_readers=readers)
    set_meter_provider(provider)
    return provider


def _has_meter_provider() -> bool:
    """Return True iff a real (non-proxy) MeterProvider is installed."""
    try:
        from opentelemetry.metrics import get_meter_provider
    except ImportError:
        return False
    provider = get_meter_provider()
    # The default ``ProxyMeterProvider`` becomes a real provider on
    # first ``get_meter`` call; until then it has no resource.
    return isinstance(provider, MeterProvider)


__all__ = ["setup_tracing", "setup_metrics"]