"""Claim2Code's verification retrieval — hybrid pool reranked by the chat model.

Baseline D: exactly what the verification workflow uses before the
verdict call (``retrieval/reranker.py``): hybrid's top ``POOL_SIZE``,
reordered by the configured chat model, cut to ``top_k``. Requires the
embedding and chat models.
"""
from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from claim2code_api.indexing.embedding_client import get_embedding_client
from claim2code_api.llm.client import get_llm_client
from claim2code_api.retrieval.reranker import retrieve_reranked

NAME = "claim2code_rerank"


def run(
    db: Session,
    repository_id: UUID,
    claim: str,
    *,
    top_k: int = 5,
) -> list[tuple[UUID, float]]:
    """Return ``top_k`` ``(symbol_id, score)`` pairs in reranked order.

    The score is ``1 / rank``: reranked order is not a fused score.
    """
    matches = retrieve_reranked(
        db, repository_id, claim, top_k=top_k,
        ranker=get_llm_client(), embedding_client=get_embedding_client(),
    )
    return [(m.symbol_id, 1.0 / rank) for rank, m in enumerate(matches, start=1)]


__all__ = ["NAME", "run"]
