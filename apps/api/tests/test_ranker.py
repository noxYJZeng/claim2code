"""Unit tests for :mod:`claim2code_api.literature.ranker`.

Coverage:

  - ``RankingWeights`` defaults sum to 1.0; custom weights; sum != 1.0 raises.
  - ``rank_papers`` empty list → empty result.
  - ``rank_papers`` length mismatch raises ValueError.
  - All scores in [0, 1].
  - Sort order is desc by score.
  - Breakdown exposes all 4 legs.
  - Recency: newer paper ranks higher (ceteris paribus).
  - Citation: more-cited paper ranks higher (ceteris paribus).
  - Semantic: higher sim ranks higher (ceteris paribus).
  - method_overlap = 0.0 in v1.
  - Single-paper list returns 1 item.
  - All-zero citations (no max) → no div-by-zero; all citation scores 0.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from claim2code_api.literature.ranker import (
    DEFAULT_WEIGHTS,
    RankingWeights,
    rank_papers,
)
from claim2code_api.literature.schemas import ScholarPaper


def _paper(
    *, paper_id: str, year: int, citation_count: int = 0, abstract: str = "...",
) -> ScholarPaper:
    return ScholarPaper(
        paper_id=paper_id,
        title=f"Title {paper_id}",
        year=year,
        citation_count=citation_count,
        abstract=abstract,
    )


NOW = datetime(2026, 1, 1)


# ============================================================
# 1. RankingWeights
# ============================================================


class TestRankingWeights:
    def test_defaults_sum_to_one(self) -> None:
        """Defaults sum to 1.0 (sem=0.55 + rec=0.15 + cite=0.20 + method=0.10)."""
        w = RankingWeights()
        assert w.semantic + w.recency + w.citation + w.method_overlap == pytest.approx(1.0)

    def test_default_singleton_present(self) -> None:
        """``DEFAULT_WEIGHTS`` is a usable singleton."""
        assert DEFAULT_WEIGHTS.semantic == 0.55

    def test_sum_not_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            RankingWeights(semantic=0.5, recency=0.2, citation=0.1, method_overlap=0.05)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            RankingWeights(semantic=-0.1, recency=0.4, citation=0.5, method_overlap=0.2)


# ============================================================
# 2. rank_papers
# ============================================================


class TestRankPapersEmpty:
    def test_empty_papers(self) -> None:
        assert rank_papers([], sims=[], now=NOW) == []

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            rank_papers([_paper(paper_id="x", year=2020)], sims=[], now=NOW)


class TestRankPapersScores:
    def test_scores_in_unit_interval(self) -> None:
        papers = [_paper(paper_id="a", year=2024), _paper(paper_id="b", year=2020)]
        ranked = rank_papers(papers, sims=[0.7, 0.3], now=NOW)
        for rp in ranked:
            assert 0.0 <= rp.score <= 1.0

    def test_breakdown_has_all_four_legs(self) -> None:
        """score_breakdown exposes semantic/recency/citation/method_overlap."""
        papers = [_paper(paper_id="a", year=2024)]
        ranked = rank_papers(papers, sims=[0.5], now=NOW)
        assert set(ranked[0].score_breakdown) == {"semantic", "recency", "citation", "method_overlap"}

    def test_method_overlap_zero_in_v1(self) -> None:
        """method_overlap is 0.0 in v1 (placeholder for future phase)."""
        papers = [_paper(paper_id="a", year=2024)]
        ranked = rank_papers(papers, sims=[0.5], now=NOW)
        assert ranked[0].score_breakdown["method_overlap"] == 0.0


class TestRankPapersOrdering:
    def test_sort_desc(self) -> None:
        """Sorted by score desc."""
        papers = [
            _paper(paper_id="low", year=2010, citation_count=0),
            _paper(paper_id="mid", year=2024, citation_count=100),
            _paper(paper_id="high", year=2024, citation_count=100),
        ]
        ranked = rank_papers(papers, sims=[0.2, 0.7, 0.95], now=NOW)
        assert [rp.paper.paper_id for rp in ranked] == ["high", "mid", "low"]

    def test_newer_beats_older_same_sim(self) -> None:
        """Ceteris paribus, newer paper ranks higher (recency)."""
        papers = [_paper(paper_id="old", year=2010), _paper(paper_id="new", year=2024)]
        ranked = rank_papers(papers, sims=[0.5, 0.5], now=NOW)
        assert ranked[0].paper.paper_id == "new"

    def test_more_cited_beats_less_cited(self) -> None:
        """Ceteris paribus, more-cited paper ranks higher (citation)."""
        papers = [
            _paper(paper_id="few", year=2024, citation_count=1),
            _paper(paper_id="many", year=2024, citation_count=100),
        ]
        ranked = rank_papers(papers, sims=[0.5, 0.5], now=NOW)
        assert ranked[0].paper.paper_id == "many"

    def test_higher_sim_beats_lower_sim(self) -> None:
        """Ceteris paribus, higher sim paper ranks higher (semantic)."""
        papers = [
            _paper(paper_id="low", year=2024),
            _paper(paper_id="high", year=2024),
        ]
        ranked = rank_papers(papers, sims=[0.1, 0.9], now=NOW)
        assert ranked[0].paper.paper_id == "high"


class TestRankPapersEdgeCases:
    def test_single_paper(self) -> None:
        """Single-paper list returns 1 item with score in [0, 1]."""
        papers = [_paper(paper_id="only", year=2024)]
        ranked = rank_papers(papers, sims=[0.5], now=NOW)
        assert len(ranked) == 1
        assert 0.0 <= ranked[0].score <= 1.0

    def test_zero_citations_no_div_by_zero(self) -> None:
        """All-zero citations doesn't div-by-zero (max=0 → all zero scores)."""
        papers = [
            _paper(paper_id="a", year=2024, citation_count=0),
            _paper(paper_id="b", year=2024, citation_count=0),
        ]
        ranked = rank_papers(papers, sims=[0.9, 0.1], now=NOW)
        # Both have citation = 0; the leg contributes 0 to both.
        # Sorting falls back to semantic + recency → a (0.9) ranks higher.
        assert ranked[0].paper.paper_id == "a"
        assert ranked[0].score_breakdown["citation"] == 0.0

    def test_future_year_clamped_to_zero_age(self) -> None:
        """Future-dated paper has age=0 → recency=1.0 (defensive clamp)."""
        papers = [_paper(paper_id="future", year=2099)]
        ranked = rank_papers(papers, sims=[0.5], now=NOW)
        assert ranked[0].score_breakdown["recency"] == pytest.approx(1.0, abs=1e-6)


class TestRankPapersComparisonNull:
    def test_comparison_is_none_after_ranking(self) -> None:
        """The ranker returns RankedPaper with comparison=None (workflow fills it)."""
        papers = [_paper(paper_id="a", year=2024)]
        ranked = rank_papers(papers, sims=[0.5], now=NOW)
        assert ranked[0].comparison is None