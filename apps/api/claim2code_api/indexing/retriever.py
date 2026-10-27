"""Top-K cosine search over the embedding index.

The read-side of Phase 4. Given a paper claim (a natural-language
sentence), the retriever returns the ``top_k`` ``CodeSymbol`` rows
that are most semantically similar to the claim's embedding.

The input is a single string. The output is a list of
``(symbol, similarity)`` pairs ordered by similarity
descending (``1.0 - cosine_distance``). Cosine distance runs
``0`` for identical vectors up to ``2`` for opposites; we
surface ``similarity = 1.0 - distance`` so the API returns an
intuitive ``0..1`` "match" score.

Why one embedding per query
---------------------------
A claim is a single sentence. We embed it once. The Phase 5
hybrid retrieval layer might chunk long passages (Phase 5+),
but for v1 a sentence is the natural unit.

Why "analysis_id" not just "repository_id"
------------------------------------------
Both columns are indexed. The ``WHERE analysis_id = $2`` is
deliberate: if the user hits the same sym-link in two different
analyses, the cosine matches within the analysis scope (no
cross-analysis bleed). The retriever also accepts a
``repository_id`` so the API can scope to one repo explicitly.

What this does NOT do
---------------------
  - No chunking. A claim is one sentence; we embed it whole.
  - No re-ranking. The top-k order is the cosine order; Phase 5
    adds LLM re-ranking if needed.
  - No early-exit on low similarity. The API consumer decides
    whether the top-k is "good enough".
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from claim2code_api.indexing.embedding_client import EmbeddingClient
from claim2code_api.indexing.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


# pgvector's cosine distance operator. The query references it
# directly via ``op('<=>')`` so the SQL string is portable across
# pgvector versions without an ORM extension dependency.
_PGVECTOR_COSINE_OP = "<=>"


def retrieve_semantic(
    db: Session,
    repository_id: UUID,
    claim: str,
    *,
    top_k: int = 5,
    embedding_client: EmbeddingClient | None = None,
    exclude_path_pattern: str | None = None,
) -> list[tuple[Any, float]]:
    """Return the top-``top_k`` symbols most similar to ``claim``.

    :param db: a SQLAlchemy session (caller owns the lifecycle).
    :param repository_id: scope the search to this repository.
    :param claim: the natural-language claim string to embed.
    :param top_k: number of results to return (default 5).
    :param embedding_client: optional override for the embedding
        client. Defaults to ``get_embedding_client()``.
    :param exclude_path_pattern: optional POSIX regex (matched
        case-insensitively); symbols whose ``file_path`` matches are
        skipped.
    :returns: a list of ``(CodeSymbol, similarity)`` tuples in
        similarity-descending order. ``similarity`` is
        ``1.0 - cosine_distance`` (so ``0..1``, higher is better).
    :raises EmbeddingError: the embed call failed irrecoverably.
    """
    if not claim.strip():
        # Empty claim → empty result. Saves a round-trip.
        return []

    if embedding_client is None:
        from claim2code_api.indexing.embedding_client import get_embedding_client

        embedding_client = get_embedding_client()

    # 1. Embed the claim (1 OpenAI call).
    try:
        claim_vec = embedding_client.embed([claim])[0]
    except EmbeddingError:
        raise

    # 2. Cosine-distance query on the repo's embeddings.
    params: dict[str, Any] = {
        "claim_vec": claim_vec,
        "repository_id": str(repository_id),
        "top_k": top_k,
    }
    if exclude_path_pattern is None:
        sql = text(
            "SELECT symbol_id, "
            f"embedding {op_str()} CAST(:claim_vec AS vector) AS distance "
            "FROM code_embeddings "
            "WHERE repository_id = :repository_id "
            f"ORDER BY embedding {op_str()} CAST(:claim_vec AS vector) "
            "LIMIT :top_k"
        )
    else:
        sql = text(
            "SELECT e.symbol_id, "
            f"e.embedding {op_str()} CAST(:claim_vec AS vector) AS distance "
            "FROM code_embeddings e JOIN code_symbols s ON s.symbol_id = e.symbol_id "
            "WHERE e.repository_id = :repository_id "
            "  AND s.file_path !~* :exclude "
            f"ORDER BY e.embedding {op_str()} CAST(:claim_vec AS vector) "
            "LIMIT :top_k"
        )
        params["exclude"] = exclude_path_pattern
    rows = db.execute(sql, params).fetchall()

    if not rows:
        return []

    # 3. Bulk-fetch the CodeSymbol rows.
    symbol_ids = [r.symbol_id for r in rows]
    from claim2code_api.db.models import CodeSymbol

    symbols = db.query(CodeSymbol).filter(CodeSymbol.symbol_id.in_(symbol_ids)).all()
    by_id = {s.symbol_id: s for s in symbols}

    # 4. Build the (symbol, similarity) list, preserving distance-asc order.
    out: list[tuple[Any, float]] = []
    for r in rows:
        sym = by_id.get(r.symbol_id)
        if sym is None:
            # Orphaned embedding (race with a delete). Skip.
            continue
        out.append((sym, 1.0 - float(r.distance)))
    return out


def op_str() -> str:
    """Return the pgvector cosine operator string.

    Exposed as a module-level helper so tests can patch it
    (e.g. swap to a SQLite-friendly operator on test fixtures).
    """
    return _PGVECTOR_COSINE_OP


__all__ = ["retrieve_semantic", "op_str"]
