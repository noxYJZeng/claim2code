"""FastAPI application factory + health endpoints.

``/health`` is a pure liveness probe (always 200). ``/health/ready`` runs
``SELECT 1`` against the configured database and returns 503 with a JSON
error body when the DB is unreachable. This split is what Kubernetes /
Docker / most uptime probes expect.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from claim2code_api import __version__
from claim2code_api.app.config import get_settings
from claim2code_api.db.session import check_db_ready
from claim2code_api.literature.router import router as related_work_router
from claim2code_api.llm.router import router as llm_router
from claim2code_api.papers.router import router as papers_router
from claim2code_api.reference.router import router as reference_router
from claim2code_api.repos.router import router as repos_router
from claim2code_api.verification.router import router as verification_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan.

    Startup:
      * Wire the global OTel TracerProvider + MeterProvider from
        Settings (console exporter default; OTLP when
        ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set).
      * Install ``FastAPIInstrumentor`` so every HTTP request
        opens a server span (idempotent — safe across hot reloads).

    Teardown:
      * Flush + shut down both providers so any in-flight spans /
        metric batches are exported before the process exits.
    """
    # Imported lazily so module import doesn't require the OTel
    # instrumentation package to be installed in the test default
    # path. ``FastAPIInstrumentor`` is an idempotent no-op when
    # called twice — keep the same FastAPIInstrumentor.instrument_app
    # invocation pattern as the FastAPI docs.
    from claim2code_api.observability import setup_metrics, setup_tracing

    settings = get_settings()
    tracer_provider = setup_tracing(settings)
    meter_provider = setup_metrics(settings)

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)

    try:
        yield
    finally:
        if tracer_provider is not None:
            tracer_provider.shutdown()
        if meter_provider is not None:
            meter_provider.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Claim2Code API",
        version=__version__,
        description=(
            "Paper-to-code verification agent. See "
            "https://github.com/noxYJZeng/claim2code for the full design."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def enforce_read_only(request: Request, call_next: RequestResponseEndpoint) -> Response:
        # A public demo can expose saved results without permitting
        # uploads, clones, inference, provider changes or DB writes.
        if settings.api_read_only and request.method not in {"GET", "HEAD", "OPTIONS"}:
            return JSONResponse({"detail": "This API is read-only"}, status_code=403)
        return await call_next(request)

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/health/ready", tags=["health"])
    def ready() -> JSONResponse:
        ok, err = check_db_ready()
        if ok:
            return JSONResponse({"status": "ok", "db": True}, status_code=200)
        return JSONResponse(
            {"status": "degraded", "db": False, "error": err},
            status_code=503,
        )

    app.include_router(papers_router)
    app.include_router(repos_router)
    app.include_router(verification_router)
    app.include_router(reference_router)
    app.include_router(related_work_router)
    app.include_router(llm_router)
    return app


app = create_app()
