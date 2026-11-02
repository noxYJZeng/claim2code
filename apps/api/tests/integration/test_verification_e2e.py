"""End-to-end test for the verification pipeline.

This is an opt-in integration test that requires a real Postgres
(``DATABASE_URL`` env var) and mocks the OpenAI LLM call. It
exercises the full flow that ``POST /analyses/{id}/verifications``
would, against actual tables:

  1. INSERT one ``analysis_states`` row.
  2. INSERT one ``repositories`` row + N ``code_symbols`` rows.
  3. Populate ``analysis_states.claims`` with 5 hand-written
     claims (some with deterministic hyperparameter mismatches,
     some with metric disagreements, some with no candidates).
  4. Run ``run_verification_workflow`` end-to-end (the LLM is
     mocked).
  5. Assert the persisted ``verifications`` JSONB is non-empty
     AND the acceptance criterion holds
     (``paper_evidence AND code_evidence OR status==UNVERIFIED``).

Run with::

    DATABASE_URL=postgresql://... pytest -m integration \\
        tests/integration/test_verification_e2e.py

The ``-m integration`` marker keeps it out of CI's default run
(the GitHub workflow runs only unit tests). The
``skipif(no DATABASE_URL)`` guard skips when the env var is missing.
"""
from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from claim2code_api.app.config import get_settings
from claim2code_api.db.models import AnalysisState, CodeSymbol, Repository
from claim2code_api.papers.schemas import Claim, ClaimType, PaperEvidence
from claim2code_api.verification import workflow as workflow_mod
from claim2code_api.verification.schemas import (
    Verification,
    VerificationStatus,
)
from claim2code_api.verification.workflow import run_verification_workflow

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("DATABASE_URL"),
        reason="DATABASE_URL not set; skipping live DB integration",
    ),
]


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def db_session():
    engine = create_engine(get_settings().database_url, future=True)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    s = session_local()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def seeded_analysis(db_session: Any) -> dict[str, Any]:
    """Insert one analysis + repo + 5 code symbols + 5 claims."""
    analysis_id = uuid4()
    repository_id = uuid4()
    db_session.add(AnalysisState(
        analysis_id=analysis_id,
        paper_status="done",
        repository_status="done",
        claims=[], verifications=[], symbols=[], mappings=[], errors=[],
    ))
    db_session.add(Repository(
        repository_id=repository_id,
        analysis_id=analysis_id,
        url="https://example.com/repo",
        owner="example",
        name="repo",
        default_branch="main",
        clone_path="/tmp/fake_clone",
        n_files=1,
        n_symbols=5,
        index_status="done",
    ))
    # No ORM relationship links these models, so flush the parents
    # before their CodeSymbol children reference them.
    db_session.flush()
    # Five CodeSymbol rows with deliberately varied signatures
    # so deterministic checkers find / miss matches.
    sym_ids = [uuid4() for _ in range(5)]
    for i, sid in enumerate(sym_ids):
        signatures = [
            "def train(learning_rate=1e-3, batch_size=32): ...",  # c1: lr mismatch
            "def evaluate(dataset='mnli', metric='accuracy'): ...",  # c2: dataset match
            "def pretrain(hidden_size=768, num_epochs=3): ...",  # c3: hp match
            "def load_model(name='gpt2'): ...",  # c4: model name match
            "def forward(x): return x",  # c5: trivial, no relevance
        ]
        docstrings = [
            "Train loop.",
            "Evaluation harness.",
            "Pretraining loop.",
            "Model loader.",
            "Forward pass.",
        ]
        db_session.add(CodeSymbol(
            symbol_id=sid,
            repository_id=repository_id,
            kind="function",
            language="python",
            file_path=f"src/m{i}.py",
            start_line=1, start_col=0, end_line=10, end_col=0,
            name=f"sym{i}",
            signature=signatures[i],
            docstring=docstrings[i],
            lex_tsv=None,
        ))
    db_session.flush()

    # Five claims that target each of the five symbols.
    pe_factory = lambda quote: PaperEvidence(  # noqa: E731
        section_title="Method", page=1, char_start=0,
        char_end=len(quote), quote=quote,
    )
    claims = [
        Claim(
            claim_id="c1", claim_type=ClaimType.HYPERPARAMETER,
            text="We use AdamW with lr=5e-5.",
            confidence=0.9, paper_evidence=pe_factory("we use lr=5e-5"),
        ),
        Claim(
            claim_id="c2", claim_type=ClaimType.EVALUATION,
            text="We evaluate on MNLI.",
            confidence=0.85, paper_evidence=pe_factory("evaluated on MNLI"),
        ),
        Claim(
            claim_id="c3", claim_type=ClaimType.HYPERPARAMETER,
            text="Pretrain with hidden_size=768 for 3 epochs.",
            confidence=0.9,
            paper_evidence=pe_factory("hidden_size=768 for 3 epochs"),
        ),
        Claim(
            claim_id="c4", claim_type=ClaimType.METHOD,
            text="We use the GPT-2 model.",
            confidence=0.95, paper_evidence=pe_factory("we use the gpt2 model"),
        ),
        Claim(
            claim_id="c5", claim_type=ClaimType.METHOD,
            text="We use a novel attention mechanism.",
            confidence=0.7, paper_evidence=pe_factory("novel attention"),
        ),
    ]
    state = db_session.query(AnalysisState).filter(
        AnalysisState.analysis_id == analysis_id,
    ).one()
    state.claims = [json.loads(c.model_dump_json()) for c in claims]
    db_session.flush()

    return {
        "analysis_id": analysis_id,
        "repository_id": repository_id,
        "sym_ids": sym_ids,
    }


# ============================================================
# Helpers
# ============================================================


# ============================================================
# Mock LLM
# ============================================================


def _mock_llm_client() -> MagicMock:
    """Build a MagicMock LLMClient that emits structured
    ``Verification`` objects with empty code_evidence for
    every claim. The structural guard then coerces to UNVERIFIED.

    The point of the test is the workflow plumbing + DB
    persistence, not the LLM's verdict.
    """
    fake = MagicMock()

    def _verdict(*, claim: Claim, paper_evidence: PaperEvidence,
                 candidates: list, deterministic_discrepancies: list) -> Verification:
        return Verification(
            claim_id=claim.claim_id,
            status=VerificationStatus.UNVERIFIED,
            confidence=0.0,
            explanation="e2e mock: no real LLM call",
            paper_evidence=paper_evidence,
            code_evidence=[],
            discrepancies=[],
            hyperparameters=[],
        )

    fake.verdict_for_claim.side_effect = _verdict
    return fake


# ============================================================
# Tests
# ============================================================


def test_e2e_round_trip_persists_verifications(
    db_session: Any, seeded_analysis: dict[str, Any],
) -> None:
    """Run the full workflow against a live DB; assert JSONB written."""
    analysis_id: UUID = seeded_analysis["analysis_id"]

    # Patch the LLM + hybrid_ranker (the live repo has no
    # embeddings, so hybrid_ranker would return empty — that's
    # fine; we still verify the plumbing).
    with patch.object(
        workflow_mod, "get_llm_client", return_value=_mock_llm_client(),
    ), patch.object(
        workflow_mod, "get_embedding_client", return_value=MagicMock(),
    ):
        result = run_verification_workflow(
            db=db_session, analysis_id=analysis_id, top_k=5,
        )

    # 5 verifications, one per claim.
    assert len(result.verifications) == 5
    # All UNVERIFIED (mock LLM emits empty code_evidence → structural guard).
    assert all(v.status == VerificationStatus.UNVERIFIED for v in result.verifications)

    # Persisted to JSONB.
    state = db_session.query(AnalysisState).filter(
        AnalysisState.analysis_id == analysis_id,
    ).one()
    assert len(state.verifications) == 5


def test_e2e_acceptance_criterion_holds(
    db_session: Any, seeded_analysis: dict[str, Any],
) -> None:
    """The Phase 6 acceptance criterion must hold: every verification
    has paper_evidence AND code_evidence OR status==UNVERIFIED."""
    analysis_id: UUID = seeded_analysis["analysis_id"]

    with patch.object(
        workflow_mod, "get_llm_client", return_value=_mock_llm_client(),
    ), patch.object(
        workflow_mod, "get_embedding_client", return_value=MagicMock(),
    ):
        result = run_verification_workflow(
            db=db_session, analysis_id=analysis_id, top_k=5,
        )

    for v in result.verifications:
        # Every verification has paper_evidence (the schema requires it).
        assert v.paper_evidence is not None
        # Either code_evidence is non-empty OR status is UNVERIFIED.
        if not v.code_evidence:
            assert v.status == VerificationStatus.UNVERIFIED


def test_e2e_subset_claim_ids(
    db_session: Any, seeded_analysis: dict[str, Any],
) -> None:
    """Verify only the requested subset of claim ids."""
    analysis_id: UUID = seeded_analysis["analysis_id"]

    with patch.object(
        workflow_mod, "get_llm_client", return_value=_mock_llm_client(),
    ), patch.object(
        workflow_mod, "get_embedding_client", return_value=MagicMock(),
    ):
        result = run_verification_workflow(
            db=db_session, analysis_id=analysis_id,
            claim_ids=["c1", "c3"], top_k=5,
        )

    assert len(result.verifications) == 2
    assert sorted(v.claim_id for v in result.verifications) == ["c1", "c3"]