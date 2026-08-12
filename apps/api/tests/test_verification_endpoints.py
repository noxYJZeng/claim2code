"""Unit tests for :mod:`claim2code_api.verification.router`.

We mock the workflow (and the underlying session) so the router
tests stay CI-safe — no DB, no LLM. The workflow's logic is
covered separately by ``test_verification_workflow.py``.

Coverage:

  - POST 404 when the workflow raises HTTP 404 (analysis missing).
  - POST 422 when the workflow raises ``VerificationNotFoundError``.
  - POST 500 when the workflow raises ``VerificationLLMError``.
  - POST 200 happy path returns the run response shape.
  - GET 404 when the analysis row is missing.
  - GET 200 returns the persisted list (re-validated through
    ``Verification.model_validate``).
  - GET 200 with empty persisted list returns an empty response.
  - GET tolerates malformed entries in the JSONB.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from claim2code_api.app.main import create_app
from claim2code_api.papers.schemas import PaperEvidence
from claim2code_api.verification.exceptions import (
    VerificationLLMError,
    VerificationNotFoundError,
)
from claim2code_api.verification.router import (
    get_verifications,
    run_verification,
)
from claim2code_api.verification.schemas import (
    Verification,
    VerificationStatus,
)
from claim2code_api.verification.workflow import VerificationWorkflowResult

# ============================================================
# Helpers
# ============================================================


def _paper_evidence() -> PaperEvidence:
    return PaperEvidence(
        section_title="Method", page=1, char_start=0, char_end=10, quote="x",
    )


def _sample_verification(claim_id: str = "c1") -> Verification:
    return Verification(
        claim_id=claim_id,
        status=VerificationStatus.UNVERIFIED,
        confidence=0.0,
        explanation="x",
        paper_evidence=_paper_evidence(),
        code_evidence=[],
        discrepancies=[],
        hyperparameters=[],
    )


@dataclass
class _StubAnalysisState:
    analysis_id: UUID
    verifications: list[dict[str, Any]] = field(default_factory=list)


def _make_db(*, state: _StubAnalysisState | None) -> MagicMock:
    db = MagicMock()
    item = [state] if state else []
    q = MagicMock()
    q.filter.return_value.one_or_none.return_value = item[0] if item else None
    db.query.return_value = q
    return db


# ============================================================
# POST /analyses/{analysis_id}/verifications — error paths
# ============================================================


def test_post_passes_unknown_claim_id_to_422() -> None:
    """The router maps ``VerificationNotFoundError`` → HTTPException 422."""
    # Smoke: the exception type is what the router catches.
    err = VerificationNotFoundError("claim_id 'c-unknown' not found in analysis x")
    assert isinstance(err, Exception)
    assert "c-unknown" in str(err)


def test_post_passes_llm_error_to_500() -> None:
    err = VerificationLLMError("connection/timeout: APITimeoutError")
    assert isinstance(err, Exception)
    assert "APITimeoutError" in str(err)


# ============================================================
# POST happy path — TestClient + mocked workflow
# ============================================================


def test_post_returns_run_response() -> None:
    """A mocked workflow returns a VerificationWorkflowResult;
    the router maps it to VerificationRunResponse."""
    v = _sample_verification("c1")
    fake_result = VerificationWorkflowResult(
        analysis_id=uuid4(),
        verifications=[v],
        n_verified=0, n_partially_verified=0, n_contradicted=0, n_unverified=1,
        errors=[],
    )

    # Override the FastAPI dependency on get_db to return a stub session
    # (the workflow is patched, so the session is never touched).
    app = create_app()
    fake_db = MagicMock()
    app.dependency_overrides = {}  # type: ignore[attr-defined]

    with patch(
        "claim2code_api.verification.router.run_verification_workflow",
        return_value=fake_result,
    ):
        # The route uses Depends(get_db); we route around it by calling
        # the endpoint function directly with a MagicMock.
        resp = run_verification(
            analysis_id=fake_result.analysis_id,
            body=None,
            db=fake_db,
        )
    assert resp.n_unverified == 1
    assert resp.verifications[0].claim_id == "c1"


def test_post_404_when_workflow_raises_http_exception() -> None:
    from fastapi import HTTPException

    fake_db = MagicMock()
    with patch(
        "claim2code_api.verification.router.run_verification_workflow",
        side_effect=HTTPException(status_code=404, detail="not found"),
    ):
        with pytest.raises(HTTPException) as excinfo:
            run_verification(analysis_id=uuid4(), body=None, db=fake_db)
    assert excinfo.value.status_code == 404


def test_post_422_when_workflow_raises_verification_not_found() -> None:
    fake_db = MagicMock()
    with patch(
        "claim2code_api.verification.router.run_verification_workflow",
        side_effect=VerificationNotFoundError("c-x not found"),
    ):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as excinfo:
            run_verification(analysis_id=uuid4(), body=None, db=fake_db)
    assert excinfo.value.status_code == 422


def test_post_500_when_workflow_raises_verification_llm_error() -> None:
    fake_db = MagicMock()
    with patch(
        "claim2code_api.verification.router.run_verification_workflow",
        side_effect=VerificationLLMError("connection/timeout: boom"),
    ):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as excinfo:
            run_verification(analysis_id=uuid4(), body=None, db=fake_db)
    assert excinfo.value.status_code == 500


def test_post_passes_body_to_workflow() -> None:
    """``body`` is forwarded to ``run_verification_workflow``."""
    fake_db = MagicMock()
    fake_result = VerificationWorkflowResult(
        analysis_id=uuid4(),
        verifications=[], n_verified=0, n_partially_verified=0,
        n_contradicted=0, n_unverified=0, errors=[],
    )
    from claim2code_api.verification.schemas import VerificationRunRequest
    body = VerificationRunRequest(claim_ids=["c1", "c2"], top_k=10)
    with patch(
        "claim2code_api.verification.router.run_verification_workflow",
        return_value=fake_result,
    ) as mock_wf:
        run_verification(
            analysis_id=uuid4(),
            body=body,
            db=fake_db,
        )
    mock_wf.assert_called_once()
    kwargs = mock_wf.call_args.kwargs
    assert kwargs["claim_ids"] == ["c1", "c2"]
    assert kwargs["top_k"] == 10


# ============================================================
# GET happy path — TestClient
# ============================================================


def test_get_404_when_analysis_missing() -> None:
    fake_db = _make_db(state=None)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as excinfo:
        get_verifications(analysis_id=uuid4(), db=fake_db)
    assert excinfo.value.status_code == 404


def test_get_returns_empty_when_no_verifications() -> None:
    state = _StubAnalysisState(analysis_id=uuid4(), verifications=[])
    fake_db = _make_db(state=state)
    resp = get_verifications(analysis_id=state.analysis_id, db=fake_db)
    assert resp.analysis_id == state.analysis_id
    assert resp.verifications == []


def test_get_returns_persisted_verifications() -> None:
    v = _sample_verification("c1")
    state = _StubAnalysisState(
        analysis_id=uuid4(),
        verifications=[json.loads(v.model_dump_json())],
    )
    fake_db = _make_db(state=state)
    resp = get_verifications(analysis_id=state.analysis_id, db=fake_db)
    assert len(resp.verifications) == 1
    assert resp.verifications[0].claim_id == "c1"


def test_get_tolerates_malformed_entries() -> None:
    """A bad row in the JSONB is logged + skipped, not 500."""
    v = _sample_verification("c1")
    good = json.loads(v.model_dump_json())
    state = _StubAnalysisState(
        analysis_id=uuid4(),
        verifications=[good, {"completely": "malformed"}],
    )
    fake_db = _make_db(state=state)
    resp = get_verifications(analysis_id=state.analysis_id, db=fake_db)
    # The good entry survived; the bad one was skipped.
    assert len(resp.verifications) == 1
    assert resp.verifications[0].claim_id == "c1"


# ============================================================
# End-to-end via TestClient (no DB or LLM needed because
# the workflow + the route's db dependency are both stubbed).
# ============================================================


def test_post_endpoint_round_trip_via_test_client() -> None:
    """A full TestClient POST against a stubbed workflow."""
    app = create_app()

    v = _sample_verification("c1")
    fake_result = VerificationWorkflowResult(
        analysis_id=uuid4(),
        verifications=[v],
        n_verified=0, n_partially_verified=0, n_contradicted=0, n_unverified=1,
        errors=[],
    )

    # Stub the workflow at the router module so the import is short-circuited.
    with patch(
        "claim2code_api.verification.router.run_verification_workflow",
        return_value=fake_result,
    ):
        # Override get_db to a MagicMock so the route's Depends doesn't
        # touch a real Postgres connection.
        from claim2code_api.app.deps import get_db
        app.dependency_overrides[get_db] = lambda: MagicMock()

        client = TestClient(app)
        resp = client.post(
            f"/analyses/{fake_result.analysis_id}/verifications",
            json={},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["analysis_id"] == str(fake_result.analysis_id)
        assert body["n_unverified"] == 1
        assert len(body["verifications"]) == 1
        assert body["verifications"][0]["claim_id"] == "c1"