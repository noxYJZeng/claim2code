"""Ranker for related-work papers.

Implements a 4-leg linear-combination ranker over :class:`ScholarPaper`
candidates returned by :mod:`scholar_client`:

    score = w_sem   * semantic
          + w_rec   * recency
          + w_cite  * citation
          + w_method* method_overlap

Each leg is normalised to ``[0, 1]`` before combination, so the
weights behave as percentages and the final ``score`` stays in
``[0, 1]``. The ranker exposes each leg's contribution as the
``score_breakdown`` map on :class:`RankedPaper` (used by the UI
"why this paper?" tooltip).

Legs
----

  - **semantic** (``w_sem=0.55``) — cosine similarity between the
    claim's query embedding and the paper's abstract embedding,
    already in ``[0, 1]`` after cosine + length normalisation. The
    caller passes ``sims`` from
    :meth:`EmbeddingClient.embed`.

  - **recency** (``w_rec=0.15``) — exponential decay over the
    paper's age in years: ``exp(-age / 5)``. A 5-year-old paper
    scores ~``e^-1`` ≈ 0.37; a 10-year-old scores ~``e^-2`` ≈ 0.14.
    Papers older than ~30 years get ~``e^-6`` ≈ 0.0025 (essentially
    zero but still counted). Future-dated papers (``year >
    current_year``) clamp to ``0`` decay = ``1.0`` (defensive; S2
    occasionally has pre-prints or data glitches).

  - **citation** (``w_cite=0.20``) — normalised log citation count:
    ``log(1 + citations) / log(1 + max_citations_in_batch)``. The
    top-cited paper in the batch gets ``1.0``; papers with zero
    citations get ``0.0``.

  - **method_overlap** (``w_method=0.10``) — placeholder leg.
    Returns ``0.0`` in v1 because we don't yet have a deterministic
    method-overlap signal (would need a method taxonomy + extractor;
    Phase 6's :class:`Discrepancy` only covers model names). The
    breakdown key is present on every :class:`RankedPaper` so a
    future phase can populate it without an API change.

Why a linear combination (not a learned ranker)?
------------------------------------------------

  - Phase 7 has <60 candidate papers per call (after the S2 /
    OpenAlex fetch); a learned ranker is over-engineered.
  - Each leg is interpretable: the ``score_breakdown`` map shows
    why a paper ranked where it did, and a UI tooltip can say
    "75% semantic match, 12% recency".
  - Weights are a single dataclass with sensible defaults; future
    phases can tune them via :class:`Settings` without API changes.
  - Phase 5's hybrid retriever uses the same shape (linear combo
    of semantic / lexical / structural); Phase 7 is a sister
    ranker with different legs.

Usage::

    ranked = rank_papers(
        papers,
        sims=[0.9, 0.8, 0.7, 0.6, 0.5],
        weights=RankingWeights(),   # defaults
    )
    # ranked[0].score is the highest combined score;
    # ranked[0].score_breakdown == {"semantic": 0.9, "recency": 0.8, ...}
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from claim2code_api.literature.schemas import RankedPaper, ScholarPaper

# Half-life for the recency decay. 5 years matches the typical
# ML conference half-life ("Deep Learning is fast moving").
_RECENCY_HALF_LIFE_YEARS = 5.0

# Recency decay constant: ``exp(-age / TAU)`` — at age = TAU the
# score is ``1/e`` ≈ 0.37. Slightly different from "half-life"
# but commonly used in IR.
_RECENCY_TAU = _RECENCY_HALF_LIFE_YEARS


@dataclass(frozen=True)
class RankingWeights:
    """Weights for the 4-leg linear combination. Defaults match Phase 5."""

    semantic: float = 0.55
    recency: float = 0.15
    citation: float = 0.20
    method_overlap: float = 0.10

    def __post_init__(self) -> None:
        """Sanity-check the weights sum to ~1.0 and are non-negative."""
        total = self.semantic + self.recency + self.citation + self.method_overlap
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"RankingWeights must sum to 1.0 (got {total:.6f}); "
                "check semantic + recency + citation + method_overlap"
            )
        if any(w < 0.0 for w in (self.semantic, self.recency, self.citation, self.method_overlap)):
            raise ValueError("RankingWeights must be non-negative")


# Module-level singleton — avoids calling ``RankingWeights()`` in
# function default (ruff B008). Same object every call.
DEFAULT_WEIGHTS: RankingWeights = RankingWeights()


def rank_papers(
    papers: list[ScholarPaper],
    *,
    sims: list[float],
    weights: RankingWeights = DEFAULT_WEIGHTS,
    now: datetime | None = None,
) -> list[RankedPaper]:
    """Rank :class:`ScholarPaper` candidates by the 4-leg linear combo.

    :param papers: candidate papers from :class:`ScholarClient.search`.
    :param sims: per-paper cosine similarity in ``[0, 1]``, in the
        same order as ``papers``. Out-of-range values clamp.
    :param weights: weight dataclass; defaults match Phase 5.
    :param now: reference timestamp (defaults to UTC now). Injectable
        so tests can pin the "current year".
    :returns: ``list[RankedPaper]`` sorted by combined ``score`` desc.
        The ranker returns ``len(papers)`` items — it does NOT
        truncate to top-K; the workflow truncates after ranking so the
        caller controls ``top_k`` (request shape, 1–20).
    :raises ValueError: If ``len(sims) != len(papers)``.
    """
    if len(papers) != len(sims):
        raise ValueError(
            f"rank_papers: papers ({len(papers)}) and sims ({len(sims)}) must match"
        )
    if not papers:
        return []

    # 1) Recency: compute age for each paper.
    current_year = (now or datetime.now(tz=UTC)).year
    ages = [max(0, current_year - p.year) for p in papers]

    # 2) Citation: normalise by the batch max (log scale).
    max_cites = max(p.citation_count for p in papers)
    citation_scores = _normalise_citations([p.citation_count for p in papers], max_cites)

    # 3) Recency scores from ages.
    recency_scores = [_recency_decay(age) for age in ages]

    # 4) Semantic: clamp to [0, 1] (defensive — cosine should already be).
    semantic_scores = [max(0.0, min(1.0, s)) for s in sims]

    # 5) Method overlap: 0.0 in v1; exposed on breakdown for future phases.
    method_scores = [0.0] * len(papers)

    # 6) Combine.
    ranked: list[RankedPaper] = []
    for paper, sem, rec, cite, meth in zip(
        papers, semantic_scores, recency_scores, citation_scores, method_scores, strict=True
    ):
        score = (
            weights.semantic * sem
            + weights.recency * rec
            + weights.citation * cite
            + weights.method_overlap * meth
        )
        # Clamp again — weights sum to 1.0 by __post_init__ so score
        # is bounded by [0, 1] in normal cases, but defensive against
        # rounding + future changes to weights.
        score = max(0.0, min(1.0, score))
        breakdown = {
            "semantic": sem,
            "recency": rec,
            "citation": cite,
            "method_overlap": meth,
        }
        ranked.append(
            RankedPaper(
                paper=paper,
                score=score,
                score_breakdown=breakdown,
                comparison=None,  # populated by the workflow after LLM call
            )
        )

    # 7) Sort by score desc. ``sorted`` is stable, so papers with
    # identical scores preserve insertion order.
    ranked.sort(key=lambda rp: rp.score, reverse=True)
    return ranked


def _recency_decay(age_years: int) -> float:
    """Exponential decay: ``exp(-age / TAU)``."""
    return math.exp(-age_years / _RECENCY_TAU)


def _normalise_citations(counts: list[int], max_cites: int) -> list[float]:
    """Normalise citation counts to ``[0, 1]`` by ``log(1+n) / log(1+max)``.

    Zero max (every paper has 0 citations) returns ``[0.0]`` for all —
    avoids a div-by-zero and reflects that no paper is "more cited".
    """
    if max_cites <= 0:
        return [0.0] * len(counts)
    denom = math.log1p(max_cites)
    return [math.log1p(n) / denom for n in counts]


__all__ = ["RankingWeights", "rank_papers", "DEFAULT_WEIGHTS"]