"""FastAPI application factory + health endpoints.

``/health`` is a pure liveness probe (always 200). ``/health/ready`` runs
``SELECT 1`` against the configured database and returns 503 with a JSON
error body when the DB is unreachable. This split is what Kubernetes /
Docker / most uptime probes expect.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from claim2code_api import __version__
from claim2code_api.app.config import get_settings
from claim2code_api.db.session import check_db_ready


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan. Phase 1 has no startup work; the engine is lazy."""
    yield


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

    return app


app = create_app()
