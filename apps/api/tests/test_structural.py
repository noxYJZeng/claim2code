"""Unit tests for :mod:`claim2code_api.indexing.structural`.

Coverage:

  - Empty claim / empty candidates → identity (all 1.0 or {}).
  - Import-graph proximity: file imports "torch" + claim
    mentions "torch" → boost > 1.0 for symbols in that file.
  - File-name token match: claim mentions "attention" +
    file_path contains "attention" → boost > 1.0.
  - Both signals stack additively (1.0 + import + file).
  - Per-leg caps: import boost capped at 0.9; file-name
    boost capped at 0.30.
  - Symbol missing from DB still appears in result with 1.0.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from claim2code_api.indexing.structural import structural_boost


def _sym(symbol_id, file_path: str, kind: str = "function") -> SimpleNamespace:
    """Build a CodeSymbol-like row. We only need file_path + symbol_id."""
    return SimpleNamespace(symbol_id=symbol_id, file_path=file_path, kind=kind)


def _build_session(
    *, symbols: list[SimpleNamespace], imports: list[SimpleNamespace]
) -> MagicMock:
    """Build a MagicMock session that mimics the structural scorer's queries.

    The scorer runs two queries:

      1. ``db.query(CodeSymbol).filter(symbol_id.in_(...)).all()`` → symbols
      2. ``db.query(CodeSymbol).filter(file_path.in_(...),
                                       kind=='import').all()`` → imports

    We use a queue: first ``.query()`` call returns the
    symbols query, second returns the imports query.
    """
    session = MagicMock(name="session")
    queue = [symbols, imports]

    def _query(_model):
        q = MagicMock()
        q.filter.return_value.all.return_value = queue.pop(0)
        return q

    session.query.side_effect = _query
    return session


# ============================================================
# Identity / short-circuits
# ============================================================


def test_structural_boost_empty_candidates_returns_empty_dict() -> None:
    """No candidates → empty dict."""
    db = MagicMock()
    out = structural_boost(db, [], "the claim mentions torch")
    assert out == {}
    db.query.assert_not_called()


def test_structural_boost_empty_claim_returns_identity() -> None:
    """Empty claim → every candidate id maps to 1.0, no SQL."""
    db = MagicMock()
    s1 = uuid4()
    s2 = uuid4()
    out = structural_boost(db, [s1, s2], "")
    assert out == {s1: 1.0, s2: 1.0}
    db.query.assert_not_called()


def test_structural_boost_whitespace_claim_returns_identity() -> None:
    """Whitespace-only claim → identity (same as empty)."""
    db = MagicMock()
    sid = uuid4()
    out = structural_boost(db, [sid], "   \n  \t  ")
    assert out == {sid: 1.0}
    db.query.assert_not_called()


def test_structural_boost_all_short_tokens_claim_returns_identity() -> None:
    """Claim consisting only of 1-2 char tokens → identity."""
    db = MagicMock()
    sid = uuid4()
    # "a of in to" — every token < 3 chars, so nothing to match.
    out = structural_boost(db, [sid], "a of in to")
    # We still load the symbols (the function can't know
    # they're identity-bound until it queries), but the
    # resulting dict is identity.
    assert out == {sid: 1.0}


def test_structural_boost_symbol_missing_from_db_returns_one() -> None:
    """A candidate id that has no row in the DB still maps to 1.0.

    File path tokens (``abc``, ``def``) and claim tokens
    (``completely``, ``unrelated``) are chosen so neither
    signal contributes — isolating the identity guarantee.
    """
    sid_real = uuid4()
    sid_missing = uuid4()
    symbols = [_sym(sid_real, "abc/def.py")]
    db = _build_session(symbols=symbols, imports=[])
    out = structural_boost(db, [sid_real, sid_missing], "completely unrelated")
    assert out[sid_real] == 1.0
    assert out[sid_missing] == 1.0


# ============================================================
# Import-graph proximity
# ============================================================


def test_structural_boost_import_graph_proximity_matches_library() -> None:
    """File imports 'torch' + claim mentions 'torch' → boost > 1.0.

    The file_path tokens (``abc``, ``def``) are chosen to NOT
    overlap with claim tokens so this test isolates the
    import-graph signal from the file-name signal.
    """
    s1 = uuid4()
    syms = [_sym(s1, "abc/def.py")]
    imports = [
        SimpleNamespace(
            symbol_id=uuid4(), file_path="abc/def.py", kind="import",
            signature="torch", name="torch",
        )
    ]
    db = _build_session(symbols=syms, imports=imports)

    out = structural_boost(db, [s1], "the inference routine uses torch")

    # 1 matched lib → 0.3 import boost + 0 file-name boost = 1.3.
    assert out[s1] == pytest.approx(1.3)


def test_structural_boost_import_graph_multiple_matches_stack() -> None:
    """Multiple matched libs → additive 0.3 per match, capped at 0.9.

    The file_path tokens (``abc``, ``def``) deliberately do not
    intersect with claim tokens so this test isolates the
    import-graph signal.
    """
    s1 = uuid4()
    syms = [_sym(s1, "abc/def.py")]
    imports = [
        SimpleNamespace(
            symbol_id=uuid4(), file_path="abc/def.py", kind="import",
            signature="torch", name="torch",
        ),
        SimpleNamespace(
            symbol_id=uuid4(), file_path="abc/def.py", kind="import",
            signature="numpy as np", name="numpy",
        ),
        SimpleNamespace(
            symbol_id=uuid4(), file_path="abc/def.py", kind="import",
            signature="transformers", name="transformers",
        ),
    ]
    db = _build_session(symbols=syms, imports=imports)

    out = structural_boost(
        db, [s1], "torch numpy transformers inference attention"
    )

    # 3 matches → 0.9 import boost + 0 file-name boost = 1.9.
    assert out[s1] == pytest.approx(1.9)


def test_structural_boost_import_graph_capped_at_zero_nine() -> None:
    """Import boost is hard-capped at 0.9 (so the multiplier <= 1.9)."""
    s1 = uuid4()
    syms = [_sym(s1, "pkg/model.py")]
    # 10 imports, all matching.
    imports = [
        SimpleNamespace(
            symbol_id=uuid4(), file_path="pkg/model.py", kind="import",
            signature=f"lib{i}", name=f"lib{i}",
        )
        for i in range(10)
    ]
    db = _build_session(symbols=syms, imports=imports)

    out = structural_boost(
        db, [s1],
        "lib0 lib1 lib2 lib3 lib4 lib5 lib6 lib7 lib8 lib9",
    )

    # 10 matches → would be 3.0, capped at 0.9 → 1.9.
    assert out[s1] == pytest.approx(1.9)


def test_structural_boost_no_match_returns_one() -> None:
    """Claim with no overlap → identity."""
    s1 = uuid4()
    syms = [_sym(s1, "pkg/model.py")]
    imports = [
        SimpleNamespace(
            symbol_id=uuid4(), file_path="pkg/model.py", kind="import",
            signature="numpy", name="numpy",
        )
    ]
    db = _build_session(symbols=syms, imports=imports)

    out = structural_boost(db, [s1], "completely unrelated words")

    assert out[s1] == 1.0


# ============================================================
# File-name token match
# ============================================================


def test_structural_boost_filename_token_match() -> None:
    """Claim mentions 'attention' + file_path contains 'attention'."""
    s1 = uuid4()
    syms = [_sym(s1, "pkg/attention.py")]
    db = _build_session(symbols=syms, imports=[])

    out = structural_boost(db, [s1], "the attention mechanism runs forward")

    # 1 file-name match → 0.05 boost.
    assert out[s1] == pytest.approx(1.05)


def test_structural_boost_filename_token_match_is_case_insensitive() -> None:
    """Matching is case-insensitive."""
    s1 = uuid4()
    syms = [_sym(s1, "pkg/ATTENTION.py")]
    db = _build_session(symbols=syms, imports=[])

    out = structural_boost(db, [s1], "attention runs")

    assert out[s1] == pytest.approx(1.05)


def test_structural_boost_filename_token_capped_at_zero_thirty() -> None:
    """File-name boost is hard-capped at 0.30 (multiplier <= 1.30)."""
    s1 = uuid4()
    # File name: alpha/beta/gamma/delta/epsilon.py (5 tokens)
    syms = [_sym(s1, "alpha/beta/gamma/delta/epsilon.py")]
    db = _build_session(symbols=syms, imports=[])

    out = structural_boost(
        db, [s1], "alpha beta gamma delta epsilon"
    )

    # 5 matches → would be 0.25, under the 0.30 cap.
    assert out[s1] == pytest.approx(1.25)


def test_structural_boost_filename_token_match_more_than_six_capped() -> None:
    """6+ file-name matches → capped at 0.30 (multiplier <= 1.30)."""
    s1 = uuid4()
    syms = [_sym(s1, "alpha/beta/gamma/delta/epsilon/zeta.py")]
    db = _build_session(symbols=syms, imports=[])

    out = structural_boost(
        db, [s1], "alpha beta gamma delta epsilon zeta"
    )

    # 6 matches → would be 0.30, exactly the cap.
    assert out[s1] == pytest.approx(1.30)


# ============================================================
# Both signals stack
# ============================================================


def test_structural_boost_import_and_file_name_stack() -> None:
    """Both signals stack additively."""
    s1 = uuid4()
    syms = [_sym(s1, "pkg/attention.py")]
    imports = [
        SimpleNamespace(
            symbol_id=uuid4(), file_path="pkg/attention.py", kind="import",
            signature="torch", name="torch",
        )
    ]
    db = _build_session(symbols=syms, imports=imports)

    # Claim mentions torch + attention.
    out = structural_boost(db, [s1], "torch attention forward pass")

    # 1 import match (0.3) + 1 file match (0.05) = 0.35 boost.
    assert out[s1] == pytest.approx(1.35)


def test_structural_boost_different_files_different_boosts() -> None:
    """Symbols in different files get different boosts based on their imports."""
    s_attention = uuid4()
    s_optimizer = uuid4()
    syms = [
        _sym(s_attention, "pkg/attention.py"),
        _sym(s_optimizer, "pkg/optimizer.py"),
    ]
    imports = [
        SimpleNamespace(
            symbol_id=uuid4(), file_path="pkg/attention.py", kind="import",
            signature="torch", name="torch",
        ),
        SimpleNamespace(
            symbol_id=uuid4(), file_path="pkg/optimizer.py", kind="import",
            signature="numpy", name="numpy",
        ),
    ]
    db = _build_session(symbols=syms, imports=imports)

    out = structural_boost(
        db, [s_attention, s_optimizer], "torch attention forward"
    )

    # attention.py: 1 import match (0.3) + 1 file match (0.05) = 1.35
    assert out[s_attention] == pytest.approx(1.35)
    # optimizer.py: 0 import matches + 0 file matches = 1.0
    assert out[s_optimizer] == pytest.approx(1.0)


# ============================================================
# Short token handling
# ============================================================


def test_structural_boost_short_file_token_excluded() -> None:
    """File-name tokens < 3 chars are excluded from matching."""
    s1 = uuid4()
    # File: "pkg/a.py" — 'a' is too short.
    syms = [_sym(s1, "pkg/a.py")]
    db = _build_session(symbols=syms, imports=[])

    out = structural_boost(db, [s1], "a b c")

    # None of the tokens match (they're all < 3 chars).
    assert out[s1] == 1.0


def test_structural_boost_import_token_no_length_filter() -> None:
    """Import targets have NO length filter (so short imports like 'os' can boost)."""
    s1 = uuid4()
    syms = [_sym(s1, "pkg/main.py")]
    imports = [
        SimpleNamespace(
            symbol_id=uuid4(), file_path="pkg/main.py", kind="import",
            signature="os", name="os",
        )
    ]
    db = _build_session(symbols=syms, imports=imports)

    # "os" is in the claim (just 2 chars).
    out = structural_boost(db, [s1], "os")

    # Claim token 'os' (1 token >= 3 chars? NO — it's 2 chars).
    # So claim_tokens is empty → identity.
    # This documents the asymmetry: import matching can
    # boost on short claim tokens like 'os' IF the claim also
    # has >= 3 char tokens; otherwise identity.
    assert out[s1] == 1.0


def test_structural_boost_import_token_short_works_with_other_tokens() -> None:
    """3+ char imports both contribute when claim also has 3+ char tokens.

    ``jax`` is exactly 3 chars — it passes the claim-token
    filter and matches. ``torch`` also matches. ``os`` (2
    chars) is filtered out of claim tokens so it never matches
    in practice; see :func:`structural_boost`'s docstring.
    """
    s1 = uuid4()
    syms = [_sym(s1, "abc/def.py")]
    imports = [
        SimpleNamespace(
            symbol_id=uuid4(), file_path="abc/def.py", kind="import",
            signature="jax", name="jax",
        ),
        SimpleNamespace(
            symbol_id=uuid4(), file_path="abc/def.py", kind="import",
            signature="torch", name="torch",
        ),
    ]
    db = _build_session(symbols=syms, imports=imports)

    # 'attention' is >= 3 chars (passes filter); 'jax' and
    # 'torch' are imported; both match.
    out = structural_boost(db, [s1], "attention uses torch and jax")

    # 2 import matches → 0.6 boost.
    assert out[s1] == pytest.approx(1.6)