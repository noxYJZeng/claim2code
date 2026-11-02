"""HTTP routes for choosing the chat-model provider.

  - ``GET  /llm/providers`` — presets, which have a key configured,
    the active choice, and the (fixed) embedding model.
  - ``POST /llm/provider/test`` — try a candidate selection without
    switching to it.
  - ``POST /llm/provider`` — switch to a selection; the next LLM call
    uses it.

API keys go in, never out: a key sent here is kept in process memory
(see :mod:`claim2code_api.llm.providers`) and no response echoes it.
"""
from __future__ import annotations

import logging
import time
from hmac import compare_digest
from ipaddress import ip_address

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from claim2code_api.app.config import get_settings
from claim2code_api.llm import providers
from claim2code_api.llm.chat_backends import build_backend
from claim2code_api.llm.client import reset_llm_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])


def require_model_manager(request: Request) -> None:
    """Model management is disabled until an operator explicitly enables it."""
    settings = get_settings()
    if settings.llm_management_token:
        scheme, _, supplied = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() == "bearer" and compare_digest(
            supplied.encode(), settings.llm_management_token.encode(),
        ):
            return
        raise HTTPException(
            status_code=401, detail="Model management requires an administrator token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if settings.llm_management_allow_local and request.client is not None:
        try:
            if ip_address(request.client.host).is_loopback:
                return
        except ValueError:
            pass
    raise HTTPException(status_code=403, detail="Model management is disabled for this client")


class ProviderSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(min_length=1)
    model: str = ""
    base_url: str = ""
    api_key: str = Field(default="", repr=False)


class ProviderTestResult(BaseModel):
    ok: bool
    error: str | None = None
    latency_ms: int | None = None
    model: str
    base_url: str


@router.get("/providers")
def list_providers() -> dict[str, object]:
    return providers.describe(get_settings())


@router.post("/provider/test", response_model=ProviderTestResult, dependencies=[Depends(require_model_manager)])
def test_provider(body: ProviderSelection) -> ProviderTestResult:
    """Check that a selection reaches its server and knows the model.

    OpenAI-compatible servers get a one-token completion; Anthropic a
    model lookup. A failure is a normal 200 response with ``ok=false``.
    """
    settings = get_settings()
    try:
        resolved = providers.resolve(
            settings, preset_id=body.provider_id, model=body.model,
            base_url=body.base_url,
            api_key=body.api_key or providers.session_key(settings, body.provider_id, body.base_url),
        )
    except providers.ProviderError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    start = time.monotonic()
    try:
        build_backend(resolved, timeout=30.0).ping(resolved.model)
    except Exception as exc:  # noqa: BLE001 — reported to the user, not raised
        return ProviderTestResult(
            ok=False, error=f"{exc.__class__.__name__}: {str(exc)[:300]}",
            model=resolved.model, base_url=resolved.base_url,
        )
    return ProviderTestResult(
        ok=True, latency_ms=int(1000 * (time.monotonic() - start)),
        model=resolved.model, base_url=resolved.base_url,
    )


@router.post("/provider", dependencies=[Depends(require_model_manager)])
def set_provider(body: ProviderSelection) -> dict[str, object]:
    settings = get_settings()
    try:
        resolved = providers.select(
            settings, preset_id=body.provider_id, model=body.model,
            base_url=body.base_url, api_key=body.api_key,
        )
    except providers.ProviderError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Rebuild the shared client so the next call uses the new provider.
    reset_llm_client()
    logger.info("chat provider set to %s model=%s", resolved.preset_id, resolved.model)
    return providers.describe(settings)


__all__ = ["router"]
