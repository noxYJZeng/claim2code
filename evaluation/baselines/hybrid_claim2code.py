"""Hybrid baseline — RRF fusion over (semantic, lexical, structural).

Baseline C in the eval harness — *our* system. Wraps
:func:`claim2code_api.retrieval.hybrid_ranker.hybrid_ranker`.

Default weights match the pre-declared triple ``{semantic:
0.55, lexical: 0.25, structural: 0.20}``. The runner can
override via the ``weights`` keyword on :func:`run`.
"""
from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from claim2code_api.indexing.embedding_client import get_embedding_client
from claim2code_api.retrieval.hybrid_ranker import (
    HybridMatch,
    HybridWeights,
    hybrid_ranker,
)

NAME = "hybrid_claim2code"


def run(
    db: Session,
    repository_id: UUID,
    claim: str,
    *,
    top_k: int = 5,
    weights: HybridWeights | None = None,
) -> list[tuple[UUID, float]]:
    """Return ``top_k`` ``(symbol_id, final_score)`` pairs.

    ``final_score`` is the RRF-fused score × structural
    multiplier, normalised by ``(k + 1)`` so the theoretical
    max (``sum(weights)``, all legs rank 1) maps to ``1.0``
    for the default weights. Requires the live OpenAI
    embedding client (the semantic leg hits OpenAI).

    :param weights: optional override for the default
        ``HybridWeights()``. None → defaults.
    """
    matches = hybrid_ranker(
        db,
        repository_id,
        claim,
        top_k=top_k,
        weights=weights if weights is not None else HybridWeights(),
        embedding_client=get_embedding_client(),
    )
    return [(m.symbol_id, float(m.final_score)) for m in matches]


__all__ = ["NAME", "HybridMatch", "run"]