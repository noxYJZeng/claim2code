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

Why ``websearch_to_tsquery``
----------------------------

``websearch_to_tsquery`` parses the claim the way a search
engine would — quoted phrases stay together, ``OR`` is honoured,
``-foo`` excludes the term. Much friendlier than ``to_tsquery``
(which requires pre-formatted ``lexemes``) for arbitrary user
input. We don't expose ``OR`` / quotes to the user; we just
benefit from the parser.

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
# the tests can grep for the canonical tokens (``websearch_to_tsquery``,
# ``ts_rank_cd``, ``@@``).
_LEXICAL_SQL = text(
    "SELECT symbol_id, ts_rank_cd(lex_tsv, query) AS score "
    "FROM code_symbols, "
    "     websearch_to_tsquery('english', :claim) AS query "
    "WHERE repository_id = :repository_id "
    "  AND lex_tsv @@ query "
    "ORDER BY score DESC "
    "LIMIT :top_k"
)


def lexical_search(
    db: Session,
    repository_id: UUID,
    claim: str,
    *,
    top_k: int = 5,
) -> list[tuple[UUID, float]]:
    """Return top-``top_k`` symbols by lexical overlap with ``claim``.

    :param db: a SQLAlchemy session (caller owns the lifecycle).
    :param repository_id: scope the search to this repository.
    :param claim: the natural-language claim string. Tokenised by
        Postgres's ``english`` config (lowercase + Porter stem +
        stopword removal) via ``websearch_to_tsquery``.
    :param top_k: number of results to return (default 5).
    :returns: ``list[(symbol_id, score)]`` ordered by
        ``ts_rank_cd`` descending. ``score`` is the raw rank
        value (unbounded; ~ 0.0–1.0 in practice).
    """
    if not claim.strip():
        # Empty claim → empty result. Saves a round-trip and
        # avoids passing an empty query to websearch_to_tsquery
        # which raises.
        return []

    rows = db.execute(
        _LEXICAL_SQL,
        {
            "claim": claim,
            "repository_id": str(repository_id),
            "top_k": top_k,
        },
    ).fetchall()

    return [(r.symbol_id, float(r.score)) for r in rows]


__all__ = ["lexical_search"]