"""Pytest configuration and shared fixtures.

The fixtures here let unit tests run without a real Postgres — we swap
:func:`claim2code_api.db.session.check_db_ready` for a stub via
``monkeypatch``. Integration tests (``-m integration``) hit a real DB.
"""
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from claim2code_api.app.config import Settings, get_settings
from claim2code_api.app.main import create_app

# Keep unit tests hermetic: assert against the Settings defaults, not
# whatever a developer's local ``.env`` (e.g. a local LLM) configures.
Settings.model_config["env_file"] = None
get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    """A TestClient bound to a fresh FastAPI app instance."""
    return TestClient(create_app())


@pytest.fixture
def monkeypatch_settings(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Helper that clears the lru_cache on ``get_settings`` between tests.

    Tests that mutate environment variables should request this fixture
    so subsequent reads of ``get_settings()`` pick up the new values.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def db_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make :func:`check_db_ready` report success."""

    def _ok() -> tuple[bool, None]:
        return True, None

    monkeypatch.setattr("claim2code_api.app.main.check_db_ready", _ok)


@pytest.fixture
def db_unready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make :func:`check_db_ready` report a realistic failure."""

    def _down() -> tuple[bool, str]:
        return False, "OperationalError: connection refused"

    monkeypatch.setattr("claim2code_api.app.main.check_db_ready", _down)
