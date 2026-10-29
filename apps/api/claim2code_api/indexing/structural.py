"""Repository-scoped import and path relevance multipliers.

The hybrid ranker separately retrieves identifier candidates. This module
boosts its resulting pool using imports mentioned in the claim and split
snake/camel words from file paths. Multipliers remain bounded to 1..2.2.
"""
from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from claim2code_api.db.models import CodeSymbol
from claim2code_api.indexing.identifiers import identifier_words

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

def _tokens(text: str, min_len: int = 1) -> set[str]:
    """Lower-cased alphanumeric tokens of ``text`` with length >= ``min_len``."""
    if not text:
        return set()
    return identifier_words(text, min_length=min_len)


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
    import_query = (
        db.query(CodeSymbol)
        .filter(
            CodeSymbol.file_path.in_(file_paths),
            CodeSymbol.kind == "import",
        )
    )
    repository_ids = {getattr(s, "repository_id", None) for s in symbols} - {None}
    if repository_ids:
        import_query = import_query.filter(CodeSymbol.repository_id.in_(repository_ids))
    import_rows = import_query.all()
    # A same-named path may exist in several indexed repositories.
    # Keep imports scoped to the candidate's repository, including when
    # callers intentionally score candidates from multiple repositories.
    file_import_tokens: dict[tuple[UUID | None, str], set[str]] = {}
    for imp in import_rows:
        # ``signature`` is the dotted import path (e.g.
        # ``"numpy as np"``); fall back to ``name`` if absent.
        target = imp.signature or imp.name or ""
        if not target:
            continue
        toks = _tokens(target, min_len=1)
        if toks:
            key = (getattr(imp, "repository_id", None), imp.file_path)
            file_import_tokens.setdefault(key, set()).update(toks)

    # 3. Compute per-symbol boost.
    for sym in symbols:
        # (a) Import-graph proximity — claim tokens that
        # appear in the file's imported targets.
        key = (getattr(sym, "repository_id", None), sym.file_path)
        file_tokens = file_import_tokens.get(key, set())
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
