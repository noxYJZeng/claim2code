"""Unit tests for :mod:`claim2code_api.literature.workflow`.

Coverage:

  - **404**: analysis row missing → HTTPException(404).
  - **422**: claim_id not in state.claims → LiteratureNotFoundError.
  - **Cache hit**: existing entry + force_refresh=False returns cached.
  - **Cache miss**: fresh run persists new entry.
  - **Force refresh**: bypasses cache, re-runs.
  - **Per-paper isolation**: LLM failure on paper #3 doesn't abort.
  - **Both clients fail**: empty entry persisted + error recorded.
  - **Acceptance test**: every RankedPaper satisfies the spec.
  - **JSONB write-back**: state.related_work is updated.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from claim2code_api.literature.exceptions import LiteratureNotFoundError
from claim2code_api.literature.schemas import (
    Comparison,
    RankedPaper,
    RelatedWorkEntry,
    ScholarPaper,
    ScholarSource,
)
from claim2code_api.literature.workflow import run_related_work_workflow
from claim2code_api.papers.schemas import Claim, ClaimType, PaperEvidence


def _claim(claim_id: str = "c1") -> Claim:
    ev = PaperEvidence(
        section_title="Method",
        page=3,
        char_start=0,
        char_end=30,
        quote="we use a transformer encoder",
    )
    return Claim(
        claim_id=claim_id,
        claim_type=ClaimType.METHOD,
        text="We use a transformer encoder.",
        confidence=0.9,
        paper_evidence=ev,
    )


def _paper(*, paper_id: str = "p1", year: int = 2024, citation_count: int = 50) -> ScholarPaper:
    return ScholarPaper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        authors=["Alice"],
        year=year,
        citation_count=citation_count,
        abstract="We propose a novel transformer for X.",
    )


def _ranked(
    *, paper_id: str = "p1", comparison: Comparison | None = None,
) -> RankedPaper:
    return RankedPaper(
        paper=ScholarPaper(
            paper_id=paper_id,
            title=f"Paper {paper_id}",
            year=2024,
            citation_count=50,
            abstract="...",
        ),
        score=0.7,
        score_breakdown={"semantic": 0.7, "recency": 0.5, "citation": 0.5, "method_overlap": 0.0},
        comparison=comparison,
    )


def _stub_scholar_client(*, papers: list[ScholarPaper] | None = None) -> MagicMock:
    """A ScholarClient stub returning ``papers``."""
    client = MagicMock(name="scholar_client")
    client.search.return_value = papers if papers is not None else [_paper()]
    return client


def _stub_embedding_client() -> MagicMock:
    """An EmbeddingClient stub: returns the right number of vectors per call."""
    embed = MagicMock(name="embedding_client")

    def _embed(texts: list[str]) -> list[list[float]]:
        # Each call gets its own batch — return one vector per input.
        return [[0.1] * 1536 for _ in texts]

    embed.embed.side_effect = _embed
    return embed


def _stub_llm_client(*, fail_on_paper_id: str | None = None) -> MagicMock:
    """An LLMClient stub.

    If ``fail_on_paper_id`` is set, the LLM raises
    LiteratureComparisonError when extracting for that paper.
    """
    from claim2code_api.literature.exceptions import LiteratureComparisonError

    llm = MagicMock(name="llm_client")

    def _extract(*, claim: Any, paper: Any) -> Comparison:
        if paper.paper_id == fail_on_paper_id:
            raise LiteratureComparisonError(f"forced fail on {paper.paper_id}")
        return Comparison(
            problem="p", method="m", difference="d", agreement="a",
        )

    llm.extract_comparison.side_effect = _extract
    return llm


def _make_state(
    *,
    claims: list[dict[str, Any]] | None = None,
    related_work: list[dict[str, Any]] | None = None,
    analysis_id: UUID | None = None,
) -> MagicMock:
    """A MagicMock AnalysisState carrying the given claims + related_work."""
    state = MagicMock(name="analysis_state")
    state.analysis_id = analysis_id or uuid4()
    state.claims = claims or [_claim().model_dump(mode="json")]
    state.related_work = related_work or []
    state.errors = []
    return state


def _make_db(state: MagicMock | None) -> MagicMock:
    """A MagicMock Session whose query returns ``state``."""
    db = MagicMock(name="db")
    db.query.return_value.filter.return_value.one_or_none.return_value = state
    return db


# ============================================================
# 1. 404 / 422 / cache
# ============================================================


class TestErrors:
    def test_404_when_analysis_missing(self) -> None:
        db = _make_db(state=None)
        with pytest.raises(HTTPException) as excinfo:
            run_related_work_workflow(
                db=db,
                analysis_id=uuid4(),
                claim_id="c1",
            )
        assert excinfo.value.status_code == 404

    def test_422_when_claim_missing(self) -> None:
        db = _make_db(state=_make_state(claims=[_claim("c99").model_dump(mode="json")]))
        with pytest.raises(LiteratureNotFoundError):
            run_related_work_workflow(
                db=db,
                analysis_id=uuid4(),
                claim_id="c1",
            )


class TestCache:
    def test_cache_hit_returns_existing(self) -> None:
        """If state.related_work has this claim_id, return it without re-running."""
        existing = RelatedWorkEntry(
            claim_id="c1",
            papers=[_ranked()],
            generated_at=datetime(2026, 8, 12, tzinfo=UTC),
            source=ScholarSource.SEMANTIC_SCHOLAR,
        )
        db = _make_db(state=_make_state(related_work=[existing.model_dump(mode="json")]))
        result = run_related_work_workflow(
            db=db,
            analysis_id=uuid4(),
            claim_id="c1",
        )
        # Result reflects the cache; no new papers added.
        assert result.entry.claim_id == "c1"
        assert len(result.entry.papers) == 1
        assert result.errors == []

    def test_force_refresh_bypasses_cache(self) -> None:
        """force_refresh=True bypasses cache + re-runs the pipeline."""
        existing = RelatedWorkEntry(
            claim_id="c1",
            papers=[],
            generated_at=datetime(2020, 1, 1, tzinfo=UTC),
            source=ScholarSource.SEMANTIC_SCHOLAR,
        )
        db = _make_db(state=_make_state(related_work=[existing.model_dump(mode="json")]))
        result = run_related_work_workflow(
            db=db,
            analysis_id=uuid4(),
            claim_id="c1",
            force_refresh=True,
            scholar_client=_stub_scholar_client(papers=[_paper(), _paper(paper_id="p2")]),
            fallback_client=None,
            embedding_client=_stub_embedding_client(),
            llm_client=_stub_llm_client(),
        )
        assert len(result.entry.papers) == 2


# ============================================================
# 2. Pipeline
# ============================================================


class TestPipeline:
    def test_fresh_run_persists_entry(self) -> None:
        """A fresh run writes a new RelatedWorkEntry to state.related_work."""
        state = _make_state()
        db = _make_db(state=state)
        run_related_work_workflow(
            db=db,
            analysis_id=state.analysis_id,
            claim_id="c1",
            top_k=2,
            scholar_client=_stub_scholar_client(papers=[
                _paper(paper_id="p1"), _paper(paper_id="p2"), _paper(paper_id="p3"),
            ]),
            fallback_client=None,
            embedding_client=_stub_embedding_client(),
            llm_client=_stub_llm_client(),
        )
        # State.related_work now has one entry for c1.
        assert len(state.related_work) == 1
        assert state.related_work[0]["claim_id"] == "c1"

    def test_per_paper_isolation(self) -> None:
        """One paper's LLM failure does not abort the rest."""
        state = _make_state()
        db = _make_db(state=state)
        result = run_related_work_workflow(
            db=db,
            analysis_id=state.analysis_id,
            claim_id="c1",
            top_k=3,
            scholar_client=_stub_scholar_client(papers=[
                _paper(paper_id="p1"),
                _paper(paper_id="p2"),
                _paper(paper_id="p3"),
            ]),
            fallback_client=None,
            embedding_client=_stub_embedding_client(),
            llm_client=_stub_llm_client(fail_on_paper_id="p2"),
        )
        # All 3 papers still present; only p2 has comparison=None.
        assert result.n_papers == 3
        assert result.n_with_comparison == 2
        assert len(result.errors) == 1
        # Per-paper comparison state:
        by_id = {rp.paper.paper_id: rp.comparison for rp in result.entry.papers}
        assert by_id["p1"] is not None
        assert by_id["p2"] is None
        assert by_id["p3"] is not None

    def test_both_clients_fail_returns_empty(self) -> None:
        """Both S2 and OpenAlex fail → empty entry + error recorded."""
        state = _make_state()
        db = _make_db(state=state)

        primary = MagicMock(name="primary")
        primary.search.side_effect = __import__(
            "claim2code_api.literature.exceptions", fromlist=["LiteratureSearchError"]
        ).LiteratureSearchError("primary down")

        fallback = MagicMock(name="fallback")
        fallback.search.side_effect = __import__(
            "claim2code_api.literature.exceptions", fromlist=["LiteratureSearchError"]
        ).LiteratureSearchError("fallback down")

        result = run_related_work_workflow(
            db=db,
            analysis_id=state.analysis_id,
            claim_id="c1",
            top_k=5,
            scholar_client=primary,
            fallback_client=fallback,
            embedding_client=_stub_embedding_client(),
            llm_client=_stub_llm_client(),
        )
        assert result.n_papers == 0
        assert result.entry.papers == []
        assert len(result.errors) >= 2  # one per failed client

    def test_state_errors_updated_on_failure(self) -> None:
        """When both clients fail, the error is appended to state.errors."""
        state = _make_state()
        db = _make_db(state=state)
        from claim2code_api.literature.exceptions import LiteratureSearchError

        primary = MagicMock(name="primary")
        primary.search.side_effect = LiteratureSearchError("primary down")
        fallback = MagicMock(name="fallback")
        fallback.search.side_effect = LiteratureSearchError("fallback down")

        run_related_work_workflow(
            db=db,
            analysis_id=state.analysis_id,
            claim_id="c1",
            scholar_client=primary,
            fallback_client=fallback,
            embedding_client=_stub_embedding_client(),
            llm_client=_stub_llm_client(),
        )
        assert len(state.errors) >= 2
        assert all(e["stage"] == "related_work" for e in state.errors)


# ============================================================
# 3. Acceptance test
# ============================================================


class TestAcceptance:
    def test_every_ranked_paper_satisfies_spec(self) -> None:
        """Every RankedPaper has metadata + score ∈ [0,1] + (comparison is None OR has all 6 fields)."""
        state = _make_state()
        db = _make_db(state=state)
        result = run_related_work_workflow(
            db=db,
            analysis_id=state.analysis_id,
            claim_id="c1",
            top_k=3,
            scholar_client=_stub_scholar_client(papers=[
                _paper(paper_id="p1"), _paper(paper_id="p2"), _paper(paper_id="p3"),
                _paper(paper_id="p4"), _paper(paper_id="p5"),
            ]),
            fallback_client=None,
            embedding_client=_stub_embedding_client(),
            llm_client=_stub_llm_client(fail_on_paper_id="p3"),
        )

        for rp in result.entry.papers:
            # Metadata fields populated
            assert rp.paper.title
            assert rp.paper.year
            assert rp.paper.abstract
            # Score ∈ [0, 1]
            assert 0.0 <= rp.score <= 1.0
            # Comparison is None OR has all 6 fields populated
            if rp.comparison is not None:
                assert rp.comparison.problem
                assert rp.comparison.method
                assert rp.comparison.difference
                assert rp.comparison.agreement
                # dataset / metric are allowed None