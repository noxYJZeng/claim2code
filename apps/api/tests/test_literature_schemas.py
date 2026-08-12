"""Unit tests for :mod:`claim2code_api.literature.schemas`.

Coverage:

  - ``ScholarSource`` StrEnum membership + values.
  - ``ScholarPaper`` field validation: required fields, year bounds,
    extra="forbid", citation_count >= 0.
  - ``Comparison`` field validation: 6 required strings; dataset /
    metric are nullable.
  - ``RankedPaper`` score ∈ [0, 1]; comparison is optional.
  - ``RelatedWorkEntry`` wraps RankedPaper list + claim_id.
  - ``RelatedWorkRunRequest`` defaults top_k=5; force_refresh=False;
    claim_id is required.
  - ``RelatedWorkRunResponse`` carries n_papers, n_with_comparison,
    errors list.
  - ``RelatedWorkListResponse`` carries entries list.
  - model_dump roundtrip (mode="json") is lossless.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from claim2code_api.literature.schemas import (
    Comparison,
    RankedPaper,
    RelatedWorkEntry,
    RelatedWorkListResponse,
    RelatedWorkRunRequest,
    RelatedWorkRunResponse,
    ScholarPaper,
    ScholarSource,
)

# ============================================================
# 1. ScholarSource
# ============================================================


class TestScholarSource:
    def test_values(self) -> None:
        """The two source labels are present."""
        assert ScholarSource.SEMANTIC_SCHOLAR.value == "semantic_scholar"
        assert ScholarSource.OPENALEX.value == "openalex"

    def test_membership(self) -> None:
        """Only the two known values are valid."""
        assert ScholarSource("semantic_scholar") is ScholarSource.SEMANTIC_SCHOLAR
        assert ScholarSource("openalex") is ScholarSource.OPENALEX
        with pytest.raises(ValueError, match="arxiv"):
            ScholarSource("arxiv")


# ============================================================
# 2. ScholarPaper
# ============================================================


def _paper(**overrides) -> ScholarPaper:
    defaults = {
        "paper_id": "p1",
        "title": "A test paper",
        "authors": ["Alice", "Bob"],
        "year": 2024,
        "venue": "NeurIPS",
        "citation_count": 10,
        "tldr": "short summary",
        "abstract": "We propose a novel transformer.",
        "pdf_url": "https://example.com/p.pdf",
        "external_ids": {"DOI": "10.1234/xyz"},
    }
    defaults.update(overrides)
    return ScholarPaper(**defaults)  # type: ignore[arg-type]


class TestScholarPaper:
    def test_minimal_valid(self) -> None:
        """All optional fields can be None / empty / missing."""
        p = ScholarPaper(
            paper_id="x", title="t", year=2024, citation_count=0, abstract="a",
        )
        assert p.authors == []
        assert p.venue is None
        assert p.tldr is None
        assert p.pdf_url is None
        assert p.external_ids == {}

    def test_year_bounds(self) -> None:
        """year must be 1900 ≤ year ≤ 2100."""
        with pytest.raises(ValidationError):
            _paper(year=1899)
        with pytest.raises(ValidationError):
            _paper(year=2101)

    def test_citation_count_nonneg(self) -> None:
        """citation_count >= 0."""
        with pytest.raises(ValidationError):
            _paper(citation_count=-1)

    def test_extra_forbid(self) -> None:
        """Unknown kwargs are rejected (Pydantic extra='forbid')."""
        with pytest.raises(ValidationError):
            ScholarPaper(  # type: ignore[call-arg]
                paper_id="x", title="t", year=2024, citation_count=0,
                abstract="a", nonsense=42,
            )


# ============================================================
# 3. Comparison
# ============================================================


def _comparison(**overrides) -> Comparison:
    defaults = {
        "problem": "We tackle X.",
        "method": "We use a transformer.",
        "dataset": "SQuAD",
        "metric": "F1",
        "difference": "We differ from prior work in Y.",
        "agreement": "We agree on W.",
    }
    defaults.update(overrides)
    return Comparison(**defaults)


class TestComparison:
    def test_all_required(self) -> None:
        """problem / method / difference / agreement are required strings."""
        with pytest.raises(ValidationError):
            Comparison(method="y", difference="z", agreement="w")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            Comparison(problem="x", difference="z", agreement="w")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            Comparison(problem="x", method="y", agreement="w")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            Comparison(problem="x", method="y", difference="z")  # type: ignore[call-arg]

    def test_dataset_metric_nullable(self) -> None:
        """dataset / metric are allowed to be None."""
        c = Comparison(
            problem="p", method="m", difference="d", agreement="a",
        )
        assert c.dataset is None
        assert c.metric is None

    def test_extra_forbid(self) -> None:
        """Unknown kwargs are rejected."""
        with pytest.raises(ValidationError):
            Comparison(  # type: ignore[call-arg]
                problem="p", method="m", difference="d", agreement="a",
                bogus="extra",
            )


# ============================================================
# 4. RankedPaper
# ============================================================


class TestRankedPaper:
    def test_construction(self) -> None:
        """RankedPaper carries paper + score + breakdown + comparison."""
        rp = RankedPaper(
            paper=_paper(),
            score=0.7,
            score_breakdown={"semantic": 0.9, "recency": 0.5, "citation": 0.4, "method_overlap": 0.0},
            comparison=_comparison(),
        )
        assert rp.score == pytest.approx(0.7)
        assert rp.comparison is not None
        assert rp.score_breakdown["semantic"] == 0.9

    def test_comparison_optional(self) -> None:
        """comparison is allowed to be None (LLM failure guard)."""
        rp = RankedPaper(
            paper=_paper(),
            score=0.5,
            score_breakdown={},
            comparison=None,
        )
        assert rp.comparison is None

    def test_score_bounds(self) -> None:
        """score must be 0 ≤ score ≤ 1."""
        with pytest.raises(ValidationError):
            RankedPaper(paper=_paper(), score=-0.1, score_breakdown={})
        with pytest.raises(ValidationError):
            RankedPaper(paper=_paper(), score=1.1, score_breakdown={})

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            RankedPaper(  # type: ignore[call-arg]
                paper=_paper(), score=0.5, score_breakdown={}, mystery=True,
            )


# ============================================================
# 5. RelatedWorkEntry
# ============================================================


class TestRelatedWorkEntry:
    def test_construction(self) -> None:
        """RelatedWorkEntry is per-claim wrapper + list of papers."""
        e = RelatedWorkEntry(
            claim_id="c1",
            papers=[
                RankedPaper(
                    paper=_paper(),
                    score=0.5,
                    score_breakdown={"semantic": 0.5},
                    comparison=None,
                ),
            ],
            generated_at=datetime(2026, 8, 12, tzinfo=UTC),
            source=ScholarSource.SEMANTIC_SCHOLAR,
        )
        assert e.claim_id == "c1"
        assert len(e.papers) == 1
        assert e.source is ScholarSource.SEMANTIC_SCHOLAR

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            RelatedWorkEntry(  # type: ignore[call-arg]
                claim_id="c1",
                papers=[],
                generated_at=datetime(2026, 8, 12, tzinfo=UTC),
                source=ScholarSource.SEMANTIC_SCHOLAR,
                bogus=42,
            )


# ============================================================
# 6. API request / response shapes
# ============================================================


class TestRelatedWorkRunRequest:
    def test_defaults(self) -> None:
        """top_k default 5; force_refresh default False."""
        req = RelatedWorkRunRequest(claim_id="c1")
        assert req.top_k == 5
        assert req.force_refresh is False

    def test_top_k_bounds(self) -> None:
        """top_k must be 1 ≤ top_k ≤ 20."""
        RelatedWorkRunRequest(claim_id="c1", top_k=1)
        RelatedWorkRunRequest(claim_id="c1", top_k=20)
        with pytest.raises(ValidationError):
            RelatedWorkRunRequest(claim_id="c1", top_k=0)
        with pytest.raises(ValidationError):
            RelatedWorkRunRequest(claim_id="c1", top_k=21)

    def test_claim_id_required(self) -> None:
        with pytest.raises(ValidationError):
            RelatedWorkRunRequest()  # type: ignore[call-arg]

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            RelatedWorkRunRequest(claim_id="c1", mystery=42)  # type: ignore[call-arg]


class TestRelatedWorkRunResponse:
    def test_carries_n_papers_and_errors(self) -> None:
        """n_papers / n_with_comparison / errors are exposed."""
        e = RelatedWorkEntry(
            claim_id="c1",
            papers=[],
            generated_at=datetime(2026, 8, 12, tzinfo=UTC),
            source=ScholarSource.SEMANTIC_SCHOLAR,
        )
        from uuid import uuid4

        resp = RelatedWorkRunResponse(
            analysis_id=uuid4(),
            entry=e,
            n_papers=0,
            n_with_comparison=0,
            errors=[],
        )
        assert resp.errors == []
        assert resp.n_papers == 0


class TestRelatedWorkListResponse:
    def test_construction(self) -> None:
        from uuid import uuid4

        r = RelatedWorkListResponse(analysis_id=uuid4(), entries=[])
        assert r.entries == []


# ============================================================
# 7. Roundtrip
# ============================================================


def test_model_dump_roundtrip() -> None:
    """model_dump(mode='json') + model_validate roundtrips losslessly."""
    e = RelatedWorkEntry(
        claim_id="c1",
        papers=[
            RankedPaper(
                paper=_paper(),
                score=0.5,
                score_breakdown={"semantic": 0.5},
                comparison=_comparison(),
            ),
        ],
        generated_at=datetime(2026, 8, 12, 1, 2, 3, tzinfo=UTC),
        source=ScholarSource.OPENALEX,
    )
    dumped = e.model_dump(mode="json")
    restored = RelatedWorkEntry.model_validate(dumped)
    assert restored.claim_id == e.claim_id
    assert restored.source == e.source
    assert len(restored.papers) == 1
    assert restored.papers[0].score == pytest.approx(e.papers[0].score)
    assert restored.papers[0].comparison is not None
    assert restored.generated_at == e.generated_at