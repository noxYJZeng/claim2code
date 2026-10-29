"""Rerank hybrid candidates by asking the chat model which code implements a claim.

Hybrid fusion ranks by surface agreement: an identifier or CLI flag that
shares the claim's words (``--distillation-type``) can outrank the function
that computes what the claim describes. The verifier reads the code anyway,
so before verification the chat model sees a wider hybrid pool (each
candidate's location, docstring and first lines of source) and names the
most direct implementations. Those move to the front in the model's order;
the rest keep their hybrid order behind them.

The pool and listing come only from the claim and the repository, never
from annotations. When the call fails the hybrid order is returned
unchanged, so retrieval never gets worse than hybrid alone.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from claim2code_api.db.models import CodeSymbol, Repository
from claim2code_api.indexing.embedding_client import EmbeddingClient
from claim2code_api.indexing.indexer import _read_source_range
from claim2code_api.retrieval.hybrid_ranker import HybridMatch, HybridWeights, hybrid_ranker

logger = logging.getLogger(__name__)

# Hybrid candidates shown to the model. Measured on the development
# claims: the annotated symbol sits in hybrid's top 45 for 97% of
# claims, against 89% for the top 30.
POOL_SIZE = 45
# Per candidate: the first lines of source, and a docstring excerpt.
_SOURCE_LINES = 15
_SOURCE_CHARS = 700
_DOC_CHARS = 200


class Ranker(Protocol):
    def rank_candidates(self, *, claim: str, candidates: str, n_candidates: int) -> list[int]: ...


def describe(number: int, symbol: CodeSymbol, source: str | None) -> str:
    """One numbered candidate as the model sees it."""
    lines = [f"[{number}] {symbol.kind} {symbol.name} — "
             f"{symbol.file_path}:{symbol.start_line}-{symbol.end_line}"]
    doc = " ".join((symbol.docstring or "").split())[:_DOC_CHARS]
    if doc:
        lines.append(f"doc: {doc}")
    head = "\n".join((source or "").splitlines()[:_SOURCE_LINES])[:_SOURCE_CHARS]
    if head:
        lines.append(head)
    return "\n".join(lines)


def rerank(
    db: Session,
    repository_id: UUID,
    claim: str,
    matches: list[HybridMatch],
    *,
    ranker: Ranker,
) -> list[HybridMatch]:
    """Reorder ``matches`` with the model's picks first.

    Each returned match carries ``rank_rerank`` (1-based) when the
    model named it. On any ranker failure ``matches`` is returned
    as is.
    """
    if len(matches) < 2:
        return matches
    try:
        repo = db.get(Repository, repository_id)
        clone = Path(repo.clone_path) if repo is not None and repo.clone_path else None
        rows = db.query(CodeSymbol).filter(CodeSymbol.symbol_id.in_([m.symbol_id for m in matches])).all()
        by_id = {row.symbol_id: row for row in rows}
        shown = [m for m in matches if m.symbol_id in by_id]
        listing = "\n\n".join(
            describe(i, by_id[m.symbol_id], _source(clone, by_id[m.symbol_id]))
            for i, m in enumerate(shown, start=1)
        )
        picks = ranker.rank_candidates(claim=claim, candidates=listing, n_candidates=len(shown))
        first = [replace(shown[n - 1], rank_rerank=rank) for rank, n in enumerate(picks, start=1)]
    except Exception as exc:  # noqa: BLE001 — fall back to the hybrid order
        logger.warning("rerank failed; keeping hybrid order: %s", exc)
        return matches
    picked = {m.symbol_id for m in first}
    return first + [m for m in matches if m.symbol_id not in picked]


def retrieve_reranked(
    db: Session,
    repository_id: UUID,
    claim: str,
    *,
    top_k: int,
    ranker: Ranker,
    weights: HybridWeights | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> list[HybridMatch]:
    """Hybrid top ``POOL_SIZE``, reranked by the model, cut to ``top_k``."""
    pool = hybrid_ranker(
        db, repository_id, claim,
        top_k=max(top_k, POOL_SIZE), weights=weights, embedding_client=embedding_client,
    )
    return rerank(db, repository_id, claim, pool, ranker=ranker)[:top_k]


def _source(clone: Path | None, symbol: CodeSymbol) -> str | None:
    if clone is None:
        return None
    return _read_source_range(
        clone_path=clone, rel_path=symbol.file_path,
        start_line=symbol.start_line, end_line=symbol.end_line,
    ) or None


__all__ = ["POOL_SIZE", "Ranker", "describe", "rerank", "retrieve_reranked"]
