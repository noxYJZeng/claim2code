"""Unit tests for :mod:`claim2code_api.verification.workflow`.

The workflow is the integration point for Phase 6 — it stitches
together Phase 5's ``hybrid_ranker``, the four deterministic
checkers, and ``LLMClient.verdict_for_claim``. Tests mock all
three so the test stays CI-safe.

Coverage:

  - 404 when the analysis row is missing.
  - 422-equivalent (``VerificationNotFoundError``) when
    ``claim_ids`` references an unknown claim id.
  - Empty / missing claims → empty verifications list (not error).
  - No repository bound → candidates is empty, LLM is still invoked
    for each claim, structural guard coerces to UNVERIFIED.
  - HIGH deterministic mismatch still obtains source-backed citations.
  - Per-claim isolation: one LLM failure doesn't abort the run.
  - Persistence: ``AnalysisState.verifications`` is written.
  - Acceptance criterion: every returned verification satisfies
    ``paper_evidence AND code_evidence OR status==UNVERIFIED``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from claim2code_api.papers.schemas import Claim, ClaimType, PaperEvidence
from claim2code_api.retrieval.hybrid_ranker import HybridMatch
from claim2code_api.verification import workflow as workflow_mod
from claim2code_api.verification.exceptions import VerificationNotFoundError
from claim2code_api.verification.schemas import VerificationStatus
from claim2code_api.verification.workflow import run_verification_workflow

# ============================================================
# Helpers
# ============================================================


def _paper_evidence(quote: str = "we use lr=5e-5") -> PaperEvidence:
    return PaperEvidence(
        section_title="Method", page=1, char_start=0, char_end=len(quote),
        quote=quote,
    )


def _claim(cid: str, *, text: str = "lr=5e-5", quote: str | None = None) -> Claim:
    return Claim(
        claim_id=cid,
        claim_type=ClaimType.HYPERPARAMETER,
        text=text,
        confidence=0.9,
        paper_evidence=_paper_evidence(quote or text),
    )


def _claim_dict(cid: str, *, text: str = "lr=5e-5") -> dict[str, Any]:
    c = _claim(cid, text=text)
    return json.loads(c.model_dump_json())


@dataclass
class _StubAnalysisState:
    """In-memory stand-in for :class:`AnalysisState` — only the
    fields the workflow touches are populated."""

    analysis_id: UUID
    claims: list[dict[str, Any]] = field(default_factory=list)
    verifications: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _StubRepository:
    repository_id: UUID
    analysis_id: UUID


def _make_db(
    *,
    state: _StubAnalysisState | None,
    repo: _StubRepository | None = None,
    code_symbols: list[MagicMock] | None = None,
) -> MagicMock:
    """Build a MagicMock ``Session`` that responds to the workflow's
    three queries: AnalysisState by id, Repository by analysis_id,
    CodeSymbol bulk-fetch by symbol_id."""
    db = MagicMock()

    # The workflow calls db.query() multiple times. We queue results
    # so each .filter().one_or_none() / .all() call gets the right thing.
    queue: list[Any] = []
    if state is not None:
        queue.append([state])  # AnalysisState
    if repo is not None:
        queue.append([repo])  # Repository
    if code_symbols:
        queue.append(code_symbols)  # CodeSymbol bulk fetch

    def _query(_model: Any) -> MagicMock:
        q = MagicMock()
        item = queue.pop(0) if queue else []
        q.filter.return_value.one_or_none.return_value = item[0] if item else None
        q.filter.return_value.all.return_value = item
        return q

    db.query.side_effect = _query
    return db


# ============================================================
# 404 path
# ============================================================


def test_workflow_404_when_analysis_missing() -> None:
    db = _make_db(state=None)
    with pytest.raises(HTTPException) as excinfo:
        run_verification_workflow(db=db, analysis_id=uuid4())
    assert excinfo.value.status_code == 404


# ============================================================
# 422 path — unknown claim_id
# ============================================================


def test_workflow_raises_on_unknown_claim_id() -> None:
    state = _StubAnalysisState(
        analysis_id=uuid4(),
        claims=[_claim_dict("c1"), _claim_dict("c2")],
    )
    db = _make_db(state=state)
    with pytest.raises(VerificationNotFoundError, match="c-unknown"):
        run_verification_workflow(
            db=db, analysis_id=state.analysis_id,
            claim_ids=["c-unknown"],
        )


# ============================================================
# Empty claims
# ============================================================


def test_workflow_empty_claims_returns_empty_verifications() -> None:
    state = _StubAnalysisState(analysis_id=uuid4(), claims=[])
    db = _make_db(state=state)
    result = run_verification_workflow(db=db, analysis_id=state.analysis_id)
    assert result.verifications == []
    assert result.n_verified == 0
    assert state.verifications == []  # empty list serialised


# ============================================================
# No repository bound — candidates empty
# ============================================================


def test_workflow_no_repository_returns_unverified() -> None:
    """Without a repository row, hybrid_ranker is never called,
    candidates stays empty, and the LLM still runs but cites no
    code → structural guard coerces to UNVERIFIED."""
    state = _StubAnalysisState(
        analysis_id=uuid4(),
        claims=[_claim_dict("c1")],
    )
    db = _make_db(state=state, repo=None)

    # Stub the LLM to return a fake Verification with no code_evidence.
    fake_llm = MagicMock()
    fake_llm.verdict_for_claim.return_value = MagicMock(
        claim_id="c1",
        status=VerificationStatus.VERIFIED,
        confidence=0.5,
        explanation="matches",
        paper_evidence=state.claims[0]["paper_evidence"],
        code_evidence=[],
        discrepancies=[],
        hyperparameters=[],
    )
    with patch.object(workflow_mod, "get_llm_client", return_value=fake_llm):
        result = run_verification_workflow(db=db, analysis_id=state.analysis_id)

    assert len(result.verifications) == 1
    assert result.verifications[0].status == VerificationStatus.UNVERIFIED


# ============================================================
# HIGH deterministic still needs source citations
# ============================================================


def test_workflow_requests_citations_on_high_severity() -> None:
    """A numeric mismatch alone cannot satisfy the source-evidence guard."""
    sym_id = uuid4()
    state = _StubAnalysisState(
        analysis_id=uuid4(),
        claims=[_claim_dict("c1", text="lr=5e-5")],
    )
    repo = _StubRepository(repository_id=uuid4(), analysis_id=state.analysis_id)
    sym = MagicMock(
        symbol_id=sym_id, kind="function", language="python",
        file_path="m.py", start_line=1, start_col=0, end_line=10, end_col=0,
        name="train", signature="def train(learning_rate=1e-3): ...",
        docstring="x",
    )
    # The .name is a regular attribute (used to render RetrieveMatch),
    # but MagicMock turns every attribute access into a Mock — set
    # spec=None to keep the string value.
    sym.name = "train"
    sym.signature = "def train(learning_rate=1e-3): ..."
    sym.docstring = "x"
    sym.file_path = "m.py"
    sym.kind = "function"
    sym.language = "python"
    db = _make_db(state=state, repo=repo, code_symbols=[sym])

    # hybrid_ranker returns one match.
    hybrid_match = HybridMatch(
        symbol_id=sym_id,
        final_score=0.95, rank_semantic=1, score_semantic=0.9,
        rank_lexical=1, score_lexical=0.5, rank_structural=1,
        struct_boost=1.0,
    )
    fake_llm = MagicMock()
    from claim2code_api.verification.schemas import Verification
    fake_llm.verdict_for_claim.return_value = Verification(
        claim_id="c1", status=VerificationStatus.UNVERIFIED,
        confidence=0.0, explanation="No readable source.",
        paper_evidence=_paper_evidence("lr=5e-5"),
    )
    with patch.object(
        workflow_mod, "hybrid_ranker", return_value=[hybrid_match],
    ), patch.object(workflow_mod, "get_embedding_client", return_value=MagicMock()), \
         patch.object(workflow_mod, "get_llm_client", return_value=fake_llm):
        result = run_verification_workflow(db=db, analysis_id=state.analysis_id)

    fake_llm.verdict_for_claim.assert_called_once()
    # Status: HIGH + no code_evidence → UNVERIFIED (structural guard).
    assert result.verifications[0].status == VerificationStatus.UNVERIFIED
    # The HIGH discrepancy is recorded.
    assert any(
        d.severity.value == "HIGH"
        for d in result.verifications[0].discrepancies
    )


# ============================================================
# Per-claim isolation — one LLM failure does not abort
# ============================================================


def test_workflow_isolates_per_claim_llm_failure() -> None:
    """One claim's LLM call raises; the other claim still completes."""
    state = _StubAnalysisState(
        analysis_id=uuid4(),
        claims=[_claim_dict("c1"), _claim_dict("c2")],
    )
    db = _make_db(state=state, repo=None)

    fake_llm = MagicMock()
    fake_llm.verdict_for_claim.side_effect = [
        RuntimeError("boom"),  # first call fails
        MagicMock(  # second call succeeds with a UNVERIFIED shape
            claim_id="c2",
            status=VerificationStatus.UNVERIFIED,
            confidence=0.0,
            explanation="no match",
            paper_evidence=state.claims[1]["paper_evidence"],
            code_evidence=[], discrepancies=[], hyperparameters=[],
        ),
    ]
    with patch.object(workflow_mod, "get_llm_client", return_value=fake_llm):
        result = run_verification_workflow(db=db, analysis_id=state.analysis_id)

    assert len(result.verifications) == 2
    assert len(result.errors) == 1
    assert "boom" in result.errors[0]
    # Both verifications exist; the failed one has UNVERIFIED status.
    statuses = sorted(v.status.value for v in result.verifications)
    assert statuses == ["UNVERIFIED", "UNVERIFIED"]


# ============================================================
# Persistence — JSONB is written
# ============================================================


def test_workflow_persists_to_analysis_state() -> None:
    state = _StubAnalysisState(
        analysis_id=uuid4(),
        claims=[_claim_dict("c1")],
    )
    db = _make_db(state=state, repo=None)

    fake_llm = MagicMock()
    fake_llm.verdict_for_claim.return_value = MagicMock(
        claim_id="c1", status=VerificationStatus.VERIFIED,
        confidence=0.5, explanation="x",
        paper_evidence=state.claims[0]["paper_evidence"],
        code_evidence=[], discrepancies=[], hyperparameters=[],
    )
    with patch.object(workflow_mod, "get_llm_client", return_value=fake_llm):
        run_verification_workflow(db=db, analysis_id=state.analysis_id)

    # The workflow wrote the JSONB back onto the state object.
    assert state.verifications is not None
    assert len(state.verifications) == 1
    assert state.verifications[0]["claim_id"] == "c1"


# ============================================================
# Acceptance criterion — paper_evidence AND code_evidence OR UNVERIFIED
# ============================================================


def test_workflow_satisfies_acceptance_criterion() -> None:
    """Every returned verification must satisfy
    ``paper_evidence AND code_evidence OR status==UNVERIFIED``.

    This is the Phase 6 acceptance criterion — it must hold even
    when the LLM disagrees.
    """
    state = _StubAnalysisState(
        analysis_id=uuid4(),
        claims=[_claim_dict("c1"), _claim_dict("c2"), _claim_dict("c3")],
    )
    db = _make_db(state=state, repo=None)

    # LLM tries to set every status to VERIFIED with no code_evidence.
    # The structural guard must override to UNVERIFIED.
    for cid in ("c1", "c2", "c3"):
        fake_v = MagicMock(
            claim_id=cid,
            status=VerificationStatus.VERIFIED,
            confidence=0.9,
            explanation="trust me",
            paper_evidence=state.claims[0]["paper_evidence"],
            code_evidence=[],  # intentionally empty
            discrepancies=[],
            hyperparameters=[],
        )
        fake_v.status = VerificationStatus.VERIFIED

    fake_llm = MagicMock()
    fake_llm.verdict_for_claim.side_effect = [
        MagicMock(
            claim_id="c1", status=VerificationStatus.VERIFIED,
            confidence=0.9, explanation="x",
            paper_evidence=state.claims[0]["paper_evidence"],
            code_evidence=[], discrepancies=[], hyperparameters=[],
        ),
        MagicMock(
            claim_id="c2", status=VerificationStatus.VERIFIED,
            confidence=0.9, explanation="x",
            paper_evidence=state.claims[1]["paper_evidence"],
            code_evidence=[], discrepancies=[], hyperparameters=[],
        ),
        MagicMock(
            claim_id="c3", status=VerificationStatus.VERIFIED,
            confidence=0.9, explanation="x",
            paper_evidence=state.claims[2]["paper_evidence"],
            code_evidence=[], discrepancies=[], hyperparameters=[],
        ),
    ]
    with patch.object(workflow_mod, "get_llm_client", return_value=fake_llm):
        result = run_verification_workflow(db=db, analysis_id=state.analysis_id)

    # Acceptance: every verification has paper_evidence, and either
    # code_evidence is non-empty OR status is UNVERIFIED.
    for v in result.verifications:
        assert v.paper_evidence is not None
        if not v.code_evidence:
            assert v.status == VerificationStatus.UNVERIFIED


# ============================================================
# Claim-id subset — only the requested claims are verified
# ============================================================


def test_workflow_only_verifies_requested_claims() -> None:
    state = _StubAnalysisState(
        analysis_id=uuid4(),
        claims=[_claim_dict("c1"), _claim_dict("c2"), _claim_dict("c3")],
    )
    db = _make_db(state=state, repo=None)

    fake_llm = MagicMock()
    fake_llm.verdict_for_claim.return_value = MagicMock(
        claim_id="placeholder",
        status=VerificationStatus.UNVERIFIED,
        confidence=0.0,
        explanation="x",
        paper_evidence=state.claims[0]["paper_evidence"],
        code_evidence=[], discrepancies=[], hyperparameters=[],
    )
    with patch.object(workflow_mod, "get_llm_client", return_value=fake_llm):
        result = run_verification_workflow(
            db=db, analysis_id=state.analysis_id,
            claim_ids=["c2"],
        )

    assert len(result.verifications) == 1
    assert result.verifications[0].claim_id == "c2"


def test_workflow_subset_run_keeps_other_claims_verdicts() -> None:
    state = _StubAnalysisState(
        analysis_id=uuid4(),
        claims=[_claim_dict("c1"), _claim_dict("c2")],
        verifications=[
            {"claim_id": "c1", "status": "VERIFIED"},
            {"claim_id": "c2", "status": "VERIFIED"},
        ],
    )
    db = _make_db(state=state, repo=None)

    with patch.object(workflow_mod, "get_llm_client", return_value=MagicMock()):
        run_verification_workflow(db=db, analysis_id=state.analysis_id, claim_ids=["c2"])

    assert [e["claim_id"] for e in state.verifications] == ["c1", "c2"]
    assert state.verifications[0] == {"claim_id": "c1", "status": "VERIFIED"}
    assert state.verifications[1]["status"] == "UNVERIFIED"  # no repo → re-run verdict
