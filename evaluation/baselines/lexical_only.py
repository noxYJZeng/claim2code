"""Lexical-only baseline — Postgres tsvector + ts_rank_cd (Phase 5).

Baseline B in the eval harness. Wraps
:func:`claim2code_api.indexing.lexical.lexical_search`.

This is the Phase 5 lexical leg on its own — no embedding, no
structural boost. The eval runner uses it to study the per-leg
contribution to the hybrid's overall Recall@5.
"""
from __future__ import annotations

from uuid import UUID

from claim2code_api.indexing.lexical import lexical_search
from sqlalchemy.orm import Session

NAME = "lexical_only"


def run(
    db: Session,
    repository_id: UUID,
    claim: str,
    *,
    top_k: int = 5,
) -> list[tuple[UUID, float]]:
    """Return ``top_k`` ``(symbol_id, score)`` pairs.

    ``score`` is the raw ``ts_rank_cd`` value (unbounded;
    typically ``[0, 1]``). Doesn't hit OpenAI — fails only on
    Postgres-side errors.
    """
    return lexical_search(db, repository_id, claim, top_k=top_k)


__all__ = ["NAME", "run"]