"""Tests for the paper HTTP GET endpoints.

We override the ``get_db`` FastAPI dep via ``app.dependency_overrides``
with a session-like MagicMock that returns canned Paper /
AnalysisState rows. This keeps the tests DB-free while still
exercising the full request-handler plumbing (route registration,
response-model validation, 404 path).

Integration tests that hit a real Postgres live under
``tests/integration/``.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from claim2code_api.app.deps import get_db
from claim2code_api.app.main import create_app

# ============================================================
# Test fixtures
# ============================================================


def _fake_paper_row() -> SimpleNamespace:
    pid = uuid4()
    aid = uuid4()
    return SimpleNamespace(
        paper_id=pid,
        analysis_id=aid,
        filename="attention.pdf",
        sha256="a" * 64,
        size_bytes=1_234_567,
        n_pages=12,
        n_chars=49_321,
        parse_status="done",
        uploaded_at=datetime(2026, 8, 10, tzinfo=UTC),
        parsed_at=datetime(2026, 8, 10, 0, 1, tzinfo=UTC),
        error=None,
    )


def _fake_analysis_row(claims: list[dict] | None = None) -> SimpleNamespace:
    aid = uuid4()
    return SimpleNamespace(
        analysis_id=aid,
        paper_status="done",
        claims=claims if claims is not None else [{"claim_id": "c1", "text": "x"}],
    )


@pytest.fixture
def session() -> MagicMock:
    """A MagicMock session. Tests assign
    ``session.query.return_value.filter.return_value.one_or_none.return_value``
    to control what each handler reads."""
    return MagicMock(name="session")


@pytest.fixture
def client(session: MagicMock) -> TestClient:
    """A TestClient with ``get_db`` overridden to return ``session``.

    Using ``app.dependency_overrides`` (rather than
    ``patch.object``) is the FastAPI-idiomatic way to swap a dep,
    and crucially it works *after* the route was decorated —
    ``Depends(get_db)`` captures the function reference at
    decorator time, so monkey-patching the module afterwards is
    ineffective.
    """
    app = create_app()

    def fake_dep() -> Generator[MagicMock, None, None]:
        yield session

    app.dependency_overrides[get_db] = fake_dep
    return TestClient(app)


# ============================================================
# GET /papers/{paper_id}
# ============================================================


def test_get_paper_returns_metadata(client: TestClient, session: MagicMock) -> None:
    row = _fake_paper_row()
    session.query.return_value.filter.return_value.one_or_none.return_value = row

    response = client.get(f"/papers/{row.paper_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["paper_id"] == str(row.paper_id)
    assert body["analysis_id"] == str(row.analysis_id)
    assert body["filename"] == "attention.pdf"
    assert body["sha256"] == "a" * 64
    assert body["size_bytes"] == 1_234_567
    assert body["n_pages"] == 12
    assert body["n_chars"] == 49_321
    assert body["parse_status"] == "done"
    assert body["error"] is None


def test_get_paper_returns_404_when_missing(client: TestClient, session: MagicMock) -> None:
    session.query.return_value.filter.return_value.one_or_none.return_value = None

    response = client.get(f"/papers/{uuid4()}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_paper_uses_paper_model(client: TestClient, session: MagicMock) -> None:
    """The query targets the Paper model, not AnalysisState."""
    row = _fake_paper_row()
    session.query.return_value.filter.return_value.one_or_none.return_value = row
    client.get(f"/papers/{row.paper_id}")
    session.query.assert_called()
    called_arg = session.query.call_args.args[0]
    from claim2code_api.db.models import Paper

    assert called_arg is Paper


# ============================================================
# GET /analyses/{analysis_id}/claims
# ============================================================


def test_get_claims_returns_claims(client: TestClient, session: MagicMock) -> None:
    canned_claims = [
        {
            "claim_id": "c1",
            "text": "Method A",
            "claim_type": "METHOD",
            "confidence": 0.9,
            "paper_evidence": {
                "section_title": "Method",
                "page": 3,
                "char_start": 0,
                "char_end": 10,
                "quote": "x",
            },
        },
        {
            "claim_id": "c2",
            "text": "Hyperparameter B",
            "claim_type": "HYPERPARAMETER",
            "confidence": 0.8,
            "paper_evidence": {
                "section_title": "Method",
                "page": 3,
                "char_start": 0,
                "char_end": 10,
                "quote": "y",
            },
        },
    ]
    row = _fake_analysis_row(claims=canned_claims)
    session.query.return_value.filter.return_value.one_or_none.return_value = row

    response = client.get(f"/analyses/{row.analysis_id}/claims")

    assert response.status_code == 200
    body = response.json()
    assert body["analysis_id"] == str(row.analysis_id)
    assert body["paper_status"] == "done"
    assert body["claims"] == canned_claims


def test_get_claims_returns_empty_list(client: TestClient, session: MagicMock) -> None:
    """An analysis with no claims returns claims=[] (not 404)."""
    row = _fake_analysis_row(claims=[])
    session.query.return_value.filter.return_value.one_or_none.return_value = row

    response = client.get(f"/analyses/{row.analysis_id}/claims")

    assert response.status_code == 200
    assert response.json()["claims"] == []


def test_get_claims_returns_404_when_missing(client: TestClient, session: MagicMock) -> None:
    session.query.return_value.filter.return_value.one_or_none.return_value = None

    response = client.get(f"/analyses/{uuid4()}/claims")

    assert response.status_code == 404


def test_get_claims_uses_analysis_state_model(client: TestClient, session: MagicMock) -> None:
    row = _fake_analysis_row()
    session.query.return_value.filter.return_value.one_or_none.return_value = row
    client.get(f"/analyses/{row.analysis_id}/claims")
    session.query.assert_called()
    called_arg = session.query.call_args.args[0]
    from claim2code_api.db.models import AnalysisState

    assert called_arg is AnalysisState


# ============================================================
# Route registration
# ============================================================


def test_get_routes_are_registered(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/papers/{paper_id}" in paths
    assert "get" in paths["/papers/{paper_id}"]
    assert "/analyses/{analysis_id}/claims" in paths
    assert "get" in paths["/analyses/{analysis_id}/claims"]
