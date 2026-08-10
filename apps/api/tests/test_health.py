"""Health endpoint tests.

Three scenarios:
- ``/health`` always returns 200 (liveness).
- ``/health/ready`` returns 200 when the DB probe succeeds.
- ``/health/ready`` returns 503 with a JSON body when the DB probe fails.

The DB probe is stubbed via :func:`monkeypatch` so these tests need no
running Postgres. See ``tests/integration/`` for the real-DB variants.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from claim2code_api import __version__


def test_health_liveness_always_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "version": __version__}


def test_ready_returns_200_when_db_is_up(client: TestClient, db_ready: None) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["db"] is True


def test_ready_returns_503_when_db_is_down(client: TestClient, db_unready: None) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["db"] is False
    assert "connection refused" in body["error"]


def test_openapi_doc_is_served(client: TestClient) -> None:
    """Phase 1 acceptance: ``/docs`` (Swagger UI) is reachable."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert b"swagger" in response.content.lower()
