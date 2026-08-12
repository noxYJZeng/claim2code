"""End-to-end test for the related-work pipeline.

Opt-in integration test (mirrors ``test_verification_e2e.py``).
Requires a real Postgres (``DATABASE_URL``) + mocks the scholar
HTTP calls and the OpenAI LLM call.

Exercises the full flow that ``POST /analyses/{id}/related-work``
would, against actual tables:

  1. INSERT one ``analysis_states`` row.
  2. Populate ``analysis_states.claims`` with 1 hand-written claim.
  3. Run ``run_related_work_workflow`` end-to-end (S2 + LLM mocked).
  4. Assert the persisted ``related_work`` JSONB is non-empty AND
     the acceptance criterion holds (every paper has metadata +
     score + comparison or comparison=None).

Run with::

    DATABASE_URL=postgresql://... pytest -m integration \\
        tests/integration/test_literature_e2e.py

The ``-m integration`` marker keeps it out of CI's default run.
"""
from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from claim2code_api.app.config import get_settings
from claim2code_api.db.models import AnalysisState
from claim2code_api.literature import workflow as workflow_mod
from claim2code_api.literature.schemas import Comparison, ScholarPaper
from claim2code_api.literature.workflow import run_related_work_workflow
from claim2code_api.papers.schemas import Claim, ClaimType, PaperEvidence

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
def db_session() -> Any:
    engine = create_engine(get_settings().database_url, future=True)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    s = session_local()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def seeded_analysis(db_session: Any) -> dict[str, Any]:
    """Insert one analysis with 1 claim ready for related-work."""
    analysis_id = uuid4()
    db_session.add(AnalysisState(
        analysis_id=analysis_id,
        paper_status="done",
        repository_status="done",
        claims=[],
        verifications=[],
        related_work=[],
        symbols=[],
        mappings=[],
        errors=[],
    ))
    db_session.flush()

    claim = Claim(
        claim_id="c1",
        claim_type=ClaimType.METHOD,
        text="We use a transformer encoder.",
        confidence=0.9,
        paper_evidence=PaperEvidence(
            section_title="Method",
            page=3,
            char_start=0,
            char_end=30,
            quote="we use a transformer encoder",
        ),
    )
    state = db_session.query(AnalysisState).filter(
        AnalysisState.analysis_id == analysis_id,
    ).one()
    state.claims = [json.loads(claim.model_dump_json())]
    db_session.flush()

    return {"analysis_id": analysis_id}


def _stub_scholar_client() -> MagicMock:
    """A ScholarClient stub returning 3 mock papers."""
    client = MagicMock(name="scholar_client")
    client.search.return_value = [
        ScholarPaper(
            paper_id=f"p{i}",
            title=f"Paper {i}",
            authors=["Alice"],
            year=2020 + i,
            citation_count=100 * (3 - i),
            abstract=f"Abstract for paper {i}",
        )
        for i in range(1, 4)
    ]
    return client


def _stub_embedding_client() -> MagicMock:
    embed = MagicMock(name="embedding_client")

    def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]

    embed.embed.side_effect = _embed
    return embed


def _stub_llm_client() -> MagicMock:
    llm = MagicMock(name="llm_client")

    def _extract(*, claim: Claim, paper: ScholarPaper) -> Comparison:
        return Comparison(
            problem="p", method="m", difference="d", agreement="a",
        )

    llm.extract_comparison.side_effect = _extract
    return llm


# ============================================================
# Tests
# ============================================================


def test_e2e_round_trip_persists_related_work(
    db_session: Any, seeded_analysis: dict[str, Any],
) -> None:
    """Run the workflow end-to-end against live DB; assert JSONB written."""
    analysis_id = seeded_analysis["analysis_id"]

    with patch.object(
        workflow_mod, "get_llm_client", return_value=_stub_llm_client(),
    ), patch.object(
        workflow_mod, "get_embedding_client", return_value=_stub_embedding_client(),
    ):
        result = run_related_work_workflow(
            db=db_session,
            analysis_id=analysis_id,
            claim_id="c1",
            top_k=3,
            scholar_client=_stub_scholar_client(),
            fallback_client=None,
        )

    # 3 ranked papers, all with valid 6-field Comparison.
    assert result.n_papers == 3
    assert result.n_with_comparison == 3

    # Persisted to JSONB.
    state = db_session.query(AnalysisState).filter(
        AnalysisState.analysis_id == analysis_id,
    ).one()
    assert len(state.related_work) == 1
    assert state.related_work[0]["claim_id"] == "c1"
    assert len(state.related_work[0]["papers"]) == 3


def test_e2e_acceptance_criterion_holds(
    db_session: Any, seeded_analysis: dict[str, Any],
) -> None:
    """Every paper has metadata + score ∈ [0,1] + (comparison or None)."""
    analysis_id = seeded_analysis["analysis_id"]

    with patch.object(
        workflow_mod, "get_llm_client", return_value=_stub_llm_client(),
    ), patch.object(
        workflow_mod, "get_embedding_client", return_value=_stub_embedding_client(),
    ):
        result = run_related_work_workflow(
            db=db_session,
            analysis_id=analysis_id,
            claim_id="c1",
            top_k=3,
            scholar_client=_stub_scholar_client(),
            fallback_client=None,
        )

    for rp in result.entry.papers:
        # Metadata fields populated.
        assert rp.paper.title
        assert rp.paper.year
        assert rp.paper.abstract
        # Score in [0, 1].
        assert 0.0 <= rp.score <= 1.0
        # Comparison is non-None (mock LLM always returns valid Comparison).
        assert rp.comparison is not None
        assert rp.comparison.problem
        assert rp.comparison.method


def test_e2e_force_refresh_overwrites_cached(
    db_session: Any, seeded_analysis: dict[str, Any],
) -> None:
    """force_refresh=True re-runs even when an entry already exists."""
    analysis_id = seeded_analysis["analysis_id"]

    # First run: populates the cache.
    with patch.object(
        workflow_mod, "get_llm_client", return_value=_stub_llm_client(),
    ), patch.object(
        workflow_mod, "get_embedding_client", return_value=_stub_embedding_client(),
    ):
        first = run_related_work_workflow(
            db=db_session,
            analysis_id=analysis_id,
            claim_id="c1",
            top_k=3,
            scholar_client=_stub_scholar_client(),
            fallback_client=None,
        )

    # Second run with force_refresh: should re-execute (entry replaced, not skipped).
    with patch.object(
        workflow_mod, "get_llm_client", return_value=_stub_llm_client(),
    ), patch.object(
        workflow_mod, "get_embedding_client", return_value=_stub_embedding_client(),
    ):
        second = run_related_work_workflow(
            db=db_session,
            analysis_id=analysis_id,
            claim_id="c1",
            top_k=3,
            force_refresh=True,
            scholar_client=_stub_scholar_client(),
            fallback_client=None,
        )

    assert first.entry.generated_at != second.entry.generated_at  # replaced, not cached
    # Still only one entry in JSONB (cache-by-replace).
    state = db_session.query(AnalysisState).filter(
        AnalysisState.analysis_id == analysis_id,
    ).one()
    assert len(state.related_work) == 1