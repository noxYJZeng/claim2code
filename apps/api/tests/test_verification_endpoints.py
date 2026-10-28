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
    fake_db.commit.assert_called_once()  # verdicts persist for GET


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

# ============================================================
# GET /analyses/{analysis_id}/code
# ============================================================


def _code_db(repo):
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = repo
    return db


def _repo_at(path, url="https://github.com/o/r.git"):
    from types import SimpleNamespace
    return SimpleNamespace(clone_path=str(path), url=url, default_branch="main")


def test_code_excerpt_returns_cited_lines_with_context(tmp_path) -> None:
    from claim2code_api.verification.router import get_code_excerpt

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("".join(f"line {n}\n" for n in range(1, 21)))
    out = get_code_excerpt(
        analysis_id=uuid4(), file_path="pkg/m.py", start_line=5, end_line=7, context=2,
        db=_code_db(_repo_at(tmp_path)),
    )
    assert (out.start_line, out.end_line, out.total_lines) == (5, 7, 20)
    assert [ln.number for ln in out.lines] == [3, 4, 5, 6, 7, 8, 9]
    assert out.lines[2].text == "line 5"
    # Not a git checkout → falls back to the default branch.
    assert out.url == "https://github.com/o/r/blob/main/pkg/m.py#L5-L7"


def test_code_excerpt_clamps_ranges_past_the_end(tmp_path) -> None:
    from claim2code_api.verification.router import get_code_excerpt

    (tmp_path / "m.py").write_text("a\nb\nc\n")
    out = get_code_excerpt(
        analysis_id=uuid4(), file_path="m.py", start_line=2, end_line=99, context=0,
        db=_code_db(_repo_at(tmp_path, url="https://gitlab.com/o/r")),
    )
    assert (out.start_line, out.end_line) == (2, 3)
    assert [ln.text for ln in out.lines] == ["b", "c"]
    assert out.url is None


@pytest.mark.parametrize("path", ["../secret.txt", "/etc/passwd", "missing.py"])
def test_code_excerpt_refuses_paths_outside_the_clone(tmp_path, path) -> None:
    from fastapi import HTTPException

    from claim2code_api.verification.router import get_code_excerpt

    clone = tmp_path / "clone"
    clone.mkdir()
    (tmp_path / "secret.txt").write_text("nope")
    with pytest.raises(HTTPException) as excinfo:
        get_code_excerpt(
            analysis_id=uuid4(), file_path=path, start_line=1, end_line=1, context=0,
            db=_code_db(_repo_at(clone)),
        )
    assert excinfo.value.status_code == 404


def test_code_excerpt_404_without_a_repository() -> None:
    from fastapi import HTTPException

    from claim2code_api.verification.router import get_code_excerpt

    with pytest.raises(HTTPException) as excinfo:
        get_code_excerpt(
            analysis_id=uuid4(), file_path="m.py", start_line=1, end_line=1, context=0,
            db=_code_db(None),
        )
    assert excinfo.value.status_code == 404


def test_code_excerpt_tolerates_a_line_suffix_on_old_paths(tmp_path) -> None:
    from claim2code_api.verification.router import get_code_excerpt

    (tmp_path / "m.py").write_text("a\nb\n")
    out = get_code_excerpt(
        analysis_id=uuid4(), file_path="m.py:1-2", start_line=1, end_line=2, context=0,
        db=_code_db(_repo_at(tmp_path)),
    )
    assert out.file_path == "m.py"

