"""Unit tests for :mod:`claim2code_api.indexing.lexical`.

Coverage:

  - ``lexical_search`` returns ``(symbol_id, score)`` pairs in
    ``ts_rank_cd`` descending order.
  - Empty / whitespace claim short-circuits without hitting the DB.
  - The query uses ``websearch_to_tsquery`` (search-engine parser),
    ``@@`` (tsvector match), and ``ts_rank_cd`` (cover-density
    rank).
  - The query is scoped to the requested ``repository_id``.
  - ``top_k`` is honoured.
  - Empty result → ``[]``.

Uses a MagicMock session so the unit tests don't need a real
Postgres. Integration tests in
``tests/integration/test_lexical_pgvector.py`` exercise the
real ``@@`` round-trip.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from claim2code_api.indexing.lexical import lexical_search


# ============================================================
# Happy path
# ============================================================


def test_lexical_search_returns_top_k_in_score_descending_order() -> None:
    """5 rows returned by the SQL query → 5 (symbol_id, score) tuples,
    ordered by score (ts_rank_cd) descending."""
    repo_id = uuid4()
    s1 = uuid4()
    s2 = uuid4()
    s3 = uuid4()
    rows = [
        SimpleNamespace(symbol_id=s1, score=0.85),  # best match
        SimpleNamespace(symbol_id=s2, score=0.62),
        SimpleNamespace(symbol_id=s3, score=0.31),
    ]
    session = MagicMock(name="session")
    session.execute.return_value.fetchall.return_value = rows

    out = lexical_search(session, repo_id, "the claim")

    assert out == [(s1, 0.85), (s2, 0.62), (s3, 0.31)]


def test_lexical_search_embeds_claim_in_query() -> None:
    """The claim is forwarded as a SQL parameter (no string concat)."""
    repo_id = uuid4()
    session = MagicMock(name="session")
    session.execute.return_value.fetchall.return_value = []

    lexical_search(session, repo_id, "scaled dot-product attention")

    params = session.execute.call_args.args[1]
    assert params["claim"] == "scaled dot-product attention"


def test_lexical_search_passes_repository_id_to_query() -> None:
    """The query is scoped to the requested repository_id."""
    repo_id = uuid4()
    session = MagicMock(name="session")
    session.execute.return_value.fetchall.return_value = []

    lexical_search(session, repo_id, "x")

    assert session.execute.call_args.args[1]["repository_id"] == str(repo_id)


def test_lexical_search_honours_top_k() -> None:
    """top_k=2 → query param is 2."""
    repo_id = uuid4()
    session = MagicMock(name="session")
    session.execute.return_value.fetchall.return_value = []

    lexical_search(session, repo_id, "x", top_k=2)

    assert session.execute.call_args.args[1]["top_k"] == 2


def test_lexical_search_uses_default_top_k_5() -> None:
    """No top_k argument → default 5."""
    repo_id = uuid4()
    session = MagicMock(name="session")
    session.execute.return_value.fetchall.return_value = []

    lexical_search(session, repo_id, "x")

    assert session.execute.call_args.args[1]["top_k"] == 5


# ============================================================
# SQL tokens
# ============================================================


def test_lexical_search_query_uses_websearch_to_tsquery() -> None:
    """The SQL uses ``websearch_to_tsquery`` (search-engine parser)."""
    repo_id = uuid4()
    session = MagicMock(name="session")
    session.execute.return_value.fetchall.return_value = []

    lexical_search(session, repo_id, "x")

    sql = session.execute.call_args.args[0]
    assert "websearch_to_tsquery" in str(sql)


def test_lexical_search_query_uses_tsvector_match_operator() -> None:
    """The SQL uses the pgvector ``@@`` match operator."""
    repo_id = uuid4()
    session = MagicMock(name="session")
    session.execute.return_value.fetchall.return_value = []

    lexical_search(session, repo_id, "x")

    sql = session.execute.call_args.args[0]
    assert "@@" in str(sql)


def test_lexical_search_query_uses_ts_rank_cd() -> None:
    """The SQL uses ``ts_rank_cd`` (cover-density variant)."""
    repo_id = uuid4()
    session = MagicMock(name="session")
    session.execute.return_value.fetchall.return_value = []

    lexical_search(session, repo_id, "x")

    sql = session.execute.call_args.args[0]
    assert "ts_rank_cd" in str(sql)


def test_lexical_search_query_targets_lex_tsv_column() -> None:
    """The SQL references the ``lex_tsv`` column (Phase 5 generated col)."""
    repo_id = uuid4()
    session = MagicMock(name="session")
    session.execute.return_value.fetchall.return_value = []

    lexical_search(session, repo_id, "x")

    sql = session.execute.call_args.args[0]
    assert "lex_tsv" in str(sql)


def test_lexical_search_query_uses_english_config() -> None:
    """The SQL uses Postgres's ``english`` text-search config."""
    repo_id = uuid4()
    session = MagicMock(name="session")
    session.execute.return_value.fetchall.return_value = []

    lexical_search(session, repo_id, "x")

    sql = session.execute.call_args.args[0]
    assert "'english'" in str(sql)


# ============================================================
# Empty / edge cases
# ============================================================


def test_lexical_search_empty_claim_short_circuits() -> None:
    """Empty claim → no SQL, empty result."""
    repo_id = uuid4()
    session = MagicMock(name="session")

    out = lexical_search(session, repo_id, "")

    assert out == []
    session.execute.assert_not_called()


def test_lexical_search_whitespace_only_claim_short_circuits() -> None:
    """Whitespace-only claim is treated as empty."""
    repo_id = uuid4()
    session = MagicMock(name="session")

    out = lexical_search(session, repo_id, "   \n  \t  ")

    assert out == []
    session.execute.assert_not_called()


def test_lexical_search_empty_query_result_returns_empty_list() -> None:
    """No matching symbols → empty list."""
    repo_id = uuid4()
    session = MagicMock(name="session")
    session.execute.return_value.fetchall.return_value = []

    out = lexical_search(session, repo_id, "a very specific nonsense claim")

    assert out == []


# ============================================================
# Score type
# ============================================================


def test_lexical_search_coerces_score_to_float() -> None:
    """Decimal / numpy scores are coerced to ``float``."""
    from decimal import Decimal

    repo_id = uuid4()
    s1 = uuid4()
    rows = [SimpleNamespace(symbol_id=s1, score=Decimal("0.75"))]
    session = MagicMock(name="session")
    session.execute.return_value.fetchall.return_value = rows

    out = lexical_search(session, repo_id, "x")

    assert isinstance(out[0][1], float)
    assert out[0][1] == pytest.approx(0.75)