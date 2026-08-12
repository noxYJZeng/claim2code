"""Related Work / literature search for selected claims.

Given a single ``Claim`` from an analysis, Phase 7 surfaces up to
5 comparable papers with structured Problem / Method / Dataset /
Metric / Difference / Agreement fields. The pipeline mirrors
Phase 6's verification shape:

  - **Scholarly search** via :class:`SemanticScholarClient`
    (primary, with ``x-api-key`` opt-in) or :class:`OpenAlexClient`
    (fallback; polite pool with ``mailto=``).
  - **Ranking** by a 4-leg linear combination
    (``semantic × recency × citation × method_overlap``) — the
    ``method_overlap`` leg is 0.0 in v1 and reserved for Phase 7.5.
  - **Comparison extraction** via :class:`LLMClient.extract_comparison`
    — one LLM call per paper with the same "no model knowledge"
    rule as Phase 6's verifier (every comparison field must cite
    the verbatim abstract or TLDR from the inputs).
  - **Persistence** as JSONB on
    ``AnalysisState.related_work: list[RelatedWorkEntry]`` — the
    column was added in migration ``0006_analysis_state_related_work``.

Module layout:

  exceptions          — domain errors (mirrors ``verification/exceptions.py``)
  schemas             — Pydantic models (StrEnums + ScholarPaper +
                        Comparison + RankedPaper + RelatedWorkEntry +
                        request/response shapes)
  scholar_client      — SemanticScholarClient + OpenAlexClient + Protocol
  ranker              — RankingWeights + rank_papers() (linear combination)
  comparator          — extract_comparison() thin wrapper around the LLM
  workflow            — run_related_work_workflow() orchestrator
  router              — POST + GET /analyses/{id}/related-work
"""
from __future__ import annotations

__all__ = [
    "exceptions",
    "schemas",
    "scholar_client",
    "ranker",
    "comparator",
    "workflow",
    "router",
]