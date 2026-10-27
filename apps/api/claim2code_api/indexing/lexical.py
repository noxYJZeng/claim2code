"""Lexical search over code_symbols via Postgres tsvector + ts_rank_cd.

The read-side of the Phase 5 *lexical* leg. Given a paper claim
(natural-language sentence), return the ``top_k`` ``CodeSymbol``
rows whose generated ``lex_tsv`` matches the claim's tokens,
ranked by Postgres's cover-density variant of ``ts_rank``.

The output is a list of ``(symbol_id, score)`` pairs ordered by
``ts_rank_cd`` descending — the same shape as
:func:`claim2code_api.indexing.retriever.retrieve_semantic`. The
hybrid ranker fuses these via Reciprocal Rank Fusion (RRF), so
the raw score scale doesn't matter — only the ranking does.

Why ``ts_rank_cd`` over ``ts_rank``
-----------------------------------

``ts_rank_cd`` is the cover-density variant that weights term
proximity: a symbol whose docstring contains two of the claim's
tokens right next to each other ranks higher than the same
two tokens spread across the file. For paper-claim-to-code
matches, proximity matters (a docstring that says "scaled
dot-product attention" should beat the same two words split
between docstring and signature).

Why an OR of the claim's terms
------------------------------

Claims are whole sentences. ``websearch_to_tsquery`` /
``plainto_tsquery`` AND every term together, so a 15-word claim
("The modified forward pass for LoRA is Wx + BAx, where B is
initialized to 0 ...") matched no symbol at all and the lexical
leg was silently empty. We take ``plainto_tsquery``'s stemmed,
stop-word-free lexemes and OR them (``' & '`` → ``' | '``);
``ts_rank_cd`` then rewards symbols that hit more terms, closer
together. Normalization flag ``1`` divides by ``1 + log(length)``
so huge docstring constants don't win on sheer term count.

What this does NOT do
---------------------

  - No fuzzy match. ``pg_trgm`` powers fuzzy name lookup in
    Phase 6's ``ModelNameNormalizer``; lexical_search here is
    exact-token-match only.
  - No synonym expansion. Postgres ``english`` config has the
    built-in Porter stemmer, so "running" matches "run" but
    "ML" doesn't match "machine learning".
  - No multi-language tokenisation. English-only; fine for ML
    repos whose claims are written in English prose.
"""
from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# The single SQL string used by ``lexical_search``. Inlined so
# the tests can grep for the canonical tokens (``plainto_tsquery``,
# ``ts_rank_cd``, ``@@``).
_LEXICAL_SQL = text(
    "SELECT symbol_id, ts_rank_cd(lex_tsv, query, 1) AS score "
    "FROM code_symbols, "
    "     to_tsquery('english', "
    "       replace(plainto_tsquery('english', :claim)::text, ' & ', ' | ')"
    "     ) AS query "
    "WHERE repository_id = :repository_id "
    "  AND lex_tsv @@ query "
    "  AND (CAST(:exclude AS text) IS NULL OR file_path !~* CAST(:exclude AS text)) "
    "ORDER BY score DESC "
    "LIMIT :top_k"
)


def lexical_search(
    db: Session,
    repository_id: UUID,
    claim: str,
    *,
    top_k: int = 5,
    exclude_path_pattern: str | None = None,
) -> list[tuple[UUID, float]]:
    """Return top-``top_k`` symbols by lexical overlap with ``claim``.

    :param db: a SQLAlchemy session (caller owns the lifecycle).
    :param repository_id: scope the search to this repository.
    :param claim: the natural-language claim string. Tokenised by
        Postgres's ``english`` config (lowercase + Porter stem +
        stopword removal) via ``plainto_tsquery``; terms are OR-ed.
    :param top_k: number of results to return (default 5).
    :param exclude_path_pattern: optional POSIX regex (matched
        case-insensitively); symbols whose ``file_path`` matches are
        skipped.
    :returns: ``list[(symbol_id, score)]`` ordered by
        ``ts_rank_cd`` descending. ``score`` is the raw rank
        value (unbounded; ~ 0.0–1.0 in practice).
    """
    if not claim.strip():
        # Empty claim → empty result. Saves a round-trip and
        # avoids passing an empty query to to_tsquery
        # which raises.
        return []

    rows = db.execute(
        _LEXICAL_SQL,
        {
            "claim": claim,
            "repository_id": str(repository_id),
            "top_k": top_k,
            "exclude": exclude_path_pattern,
        },
    ).fetchall()

    return [(r.symbol_id, float(r.score)) for r in rows]


__all__ = ["lexical_search"]