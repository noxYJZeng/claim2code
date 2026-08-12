"""Hybrid retrieval orchestrator (Phase 5).

This package turns the three per-leg retrieval scorers
(semantic from Phase 4, lexical + structural from Phase 5)
into a single ranked result. The hybrid ranker is the
*default* read path for ``POST /repositories/{id}/retrieve``;
Phase 4's semantic-only behaviour is preserved via the
``mode="semantic"`` flag.

Submodules:

  - :mod:`claim2code_api.retrieval.hybrid_ranker` —
    ``hybrid_ranker`` orchestrates the three legs, applies
    Reciprocal Rank Fusion, multiplies by the structural
    boost, and returns top-k ``HybridMatch`` records.
"""

from __future__ import annotations

from claim2code_api.retrieval.hybrid_ranker import (
    HybridMatch,
    HybridWeights,
    hybrid_ranker,
)

__all__ = [
    "HybridMatch",
    "HybridWeights",
    "hybrid_ranker",
]