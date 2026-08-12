"""Structural boosters — import-graph proximity + file-name token match.

The Phase 5 *structural* leg. Given a set of candidate symbol_ids
(typically the union of semantic + lexical top-k candidates from
the hybrid ranker), compute a per-symbol **multiplier** that the
hybrid ranker applies after RRF fusion.

Returns ``dict[symbol_id, multiplier]`` where ``1.0`` means "no
boost". Values range ``1.0..2.2`` (max + 0.9 from import-graph,
max + 0.3 from file-name).

Why "weak" structural in v1
---------------------------

The Phase 5 roadmap calls for ``called-from-training-loop
boost``, ``loss-metric boost``, ``dependency-graph proximity``.
None of these are implementable today:

  - **Call graph** doesn't exist. ``caller_symbol_id`` is a
    docstring-only placeholder; no column, no migration, no
    resolver. The parser emits parent → method edges in
    *name only* — ``parent_symbol_id`` is always NULL in
    practice.
  - **Training-loop / loss-metric** both require understanding
    which symbols are *called from the training loop*. Without
    a call graph, we can't tell.

What v1 *can* do with real data:

  - **Import-graph proximity** — the parser already emits
    ``kind='import'`` rows with the imported target in
    ``signature``. Files that import a lib the claim mentions
    are likely the file containing that code.
  - **File-name token match** — the file path's tokens
    (``pkg/attention.py`` → ``pkg``, ``attention``, ``py``)
    overlap with claim tokens often enough to be useful
    signal ("the model uses attention" → attention.py).

The full struct leg (call graph + loss-metric) lands in Phase
5.5 once ``caller_symbol_id`` is wired in.

Why "multiplier" not "additive score"
-------------------------------------

The hybrid ranker applies RRF over the three legs' *ranks*.
After rank fusion it normalises and applies structural as a
multiplier:

  - RRF produces the candidate ordering.
  - The multiplier nudges scores within the top-k without
    breaking the ordering.

Additive would change rank order in ways that depend on the
RRF score scale, which is brittle. Multiplicative is the
standard pattern (Elasticsearch function_score, Vespa
ranking expressions).
"""
from __future__ import annotations

import logging
import re
from uuid import UUID

from sqlalchemy.orm import Session

from claim2code_api.db.models import CodeSymbol

logger = logging.getLogger(__name__)

# The maximum boost contributed by import-graph proximity.
# ``1.0 + 0.9 = 1.9`` upper bound. Caps the signal so a file
# that imports many of the claim's libs can't dominate.
_MAX_IMPORT_BOOST = 0.9

# Per-matched-library contribution.
_IMPORT_PER_MATCH = 0.3

# The maximum boost contributed by file-name token match.
# ``1.0 + 0.30 = 1.30`` upper bound.
_MAX_FILE_BOOST = 0.30

# Per-matched-token contribution.
_FILE_PER_MATCH = 0.05

# Minimum token length to count as a "word". Files shorter
# than this (e.g. "a.py", "os") are excluded from file-name
# matching to avoid noise.
_FILE_TOKEN_MIN_LEN = 3

# Pattern: alphanumeric runs (Unicode letters + digits +
# underscore). Used to split file paths and import targets
# into tokens.
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(text: str, min_len: int = 1) -> set[str]:
    """Lower-cased alphanumeric tokens of ``text`` with length >= ``min_len``."""
    if not text:
        return set()
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= min_len}


def structural_boost(
    db: Session,
    candidate_ids: list[UUID],
    claim: str,
) -> dict[UUID, float]:
    """Compute per-symbol structural boost multipliers.

    :param db: a SQLAlchemy session (caller owns the lifecycle).
    :param candidate_ids: the symbol_ids to compute boosts for.
    :param claim: the paper claim string. Tokenised by the
        same alphanumeric-run rule as file paths / imports so
        matching is consistent.
    :returns: ``dict[symbol_id, multiplier]``. Every candidate
        id appears in the dict (default ``1.0`` if no signals
        apply or the row is missing). Missing symbols get
        ``1.0`` so the hybrid ranker can multiply blindly.
    """
    # Always return every input id, default 1.0.
    out: dict[UUID, float] = {sid: 1.0 for sid in candidate_ids}
    if not candidate_ids or not claim or not claim.strip():
        return out

    claim_tokens = _tokens(claim, min_len=_FILE_TOKEN_MIN_LEN)
    if not claim_tokens:
        # After stripping 1-2 char tokens (e.g. "a", "of"), the
        # claim has nothing useful to match against.
        return out

    # 1. Load the candidate rows. We need file_path + the
    #    file's imports — but file imports are themselves
    #    ``code_symbols`` rows where ``kind='import'``.
    symbols = (
        db.query(CodeSymbol)
        .filter(CodeSymbol.symbol_id.in_(candidate_ids))
        .all()
    )
    if not symbols:
        return out

    file_paths = sorted({s.file_path for s in symbols if s.file_path})

    # 2. Load the file's import targets in one query.
    import_rows = (
        db.query(CodeSymbol)
        .filter(
            CodeSymbol.file_path.in_(file_paths),
            CodeSymbol.kind == "import",
        )
        .all()
    )
    # file_path -> set of imported-target tokens (any length).
    file_import_tokens: dict[str, set[str]] = {}
    for imp in import_rows:
        # ``signature`` is the dotted import path (e.g.
        # ``"numpy as np"``); fall back to ``name`` if absent.
        target = imp.signature or imp.name or ""
        if not target:
            continue
        toks = _tokens(target, min_len=1)
        if toks:
            file_import_tokens.setdefault(imp.file_path, set()).update(toks)

    # 3. Compute per-symbol boost.
    for sym in symbols:
        # (a) Import-graph proximity — claim tokens that
        # appear in the file's imported targets.
        file_tokens = file_import_tokens.get(sym.file_path, set())
        matched_libs = len(claim_tokens & file_tokens)
        import_boost = min(_IMPORT_PER_MATCH * matched_libs, _MAX_IMPORT_BOOST)

        # (b) File-name token match — claim tokens that
        # appear in the file_path (split on ``/`` and ``.``
        # by the token regex).
        path_tokens = _tokens(sym.file_path, min_len=_FILE_TOKEN_MIN_LEN)
        matched_path = len(claim_tokens & path_tokens)
        file_boost = min(_FILE_PER_MATCH * matched_path, _MAX_FILE_BOOST)

        out[sym.symbol_id] = 1.0 + import_boost + file_boost

    return out


__all__ = ["structural_boost"]