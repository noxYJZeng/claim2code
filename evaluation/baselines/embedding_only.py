"""Embedding-only baseline — pgvector cosine similarity (Phase 4).

Baseline A in the eval harness. Wraps
:func:`claim2code_api.indexing.retriever.retrieve_semantic`.

This is the Phase 4 read path on its own — no lexical signal,
no structural boost. Establishes the floor for the "hybrid
> embedding-only" claim that the Phase 5 acceptance criterion
requires.
"""
from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from claim2code_api.indexing.embedding_client import get_embedding_client
from claim2code_api.indexing.retriever import retrieve_semantic

NAME = "embedding_only"


def run(
    db: Session,
    repository_id: UUID,
    claim: str,
    *,
    top_k: int = 5,
) -> list[tuple[UUID, float]]:
    """Return ``top_k`` ``(symbol_id, similarity)`` pairs.

    ``similarity`` is the pgvector cosine similarity (1.0 -
    cosine_distance), in ``[0, 1]``.

    Requires the live OpenAI embedding client (or whatever
    ``get_embedding_client()`` is wired to). If the embedding
    call fails, :class:`EmbeddingError` propagates and the
    runner will mark the baseline's row with an error.
    """
    rows = retrieve_semantic(
        db,
        repository_id,
        claim,
        top_k=top_k,
        embedding_client=get_embedding_client(),
    )
    return [(sym.symbol_id, float(sim)) for sym, sim in rows]


__all__ = ["NAME", "run"]