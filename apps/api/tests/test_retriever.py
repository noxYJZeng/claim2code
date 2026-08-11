"""Unit tests for :mod:`claim2code_api.indexing.retriever`.

Coverage:

  - ``retrieve_semantic`` returns ``(symbol, similarity)`` pairs in
    similarity-descending order.
  - ``similarity`` is ``1.0 - distance`` (intuitive 0..1 range).
  - The cosine query uses the pgvector ``<=>`` operator.
  - An empty claim short-circuits without calling the embed client.
  - The query is scoped to the requested repository_id.
  - ``top_k`` is honoured.
  - EmbeddingError is re-raised.
  - Orphaned embedding rows (symbol_id not in code_symbols) are
    silently skipped.
  - The function works with a MagicMock session (no real Postgres).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from claim2code_api.db.models import CodeSymbol
from claim2code_api.indexing.embedding_client import EmbeddingClient
from claim2code_api.indexing.exceptions import EmbeddingError
from claim2code_api.indexing.retriever import retrieve_semantic

# ============================================================
# Helpers
# ============================================================


def _make_vec(seed: int = 0) -> list[float]:
    """A 4-dim vector placeholder."""
    return [float(seed + j) / 10.0 for j in range(4)]


def _fake_embedding_client(
    payload: list[list[float]] | Exception,
) -> EmbeddingClient:
    class _Stub:
        def __init__(self, payload) -> None:
            self.payload = payload
            self.calls: list[list[str]] = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            if isinstance(self.payload, Exception):
                raise self.payload
            return self.payload

    return _Stub(payload)


def _make_symbol(symbol_id) -> MagicMock:
    s = MagicMock(spec=CodeSymbol)
    s.symbol_id = symbol_id
    return s


def _build_session(
    *,
    rows: list[SimpleNamespace] | None = None,
    symbols: list[CodeSymbol | MagicMock] | None = None,
) -> MagicMock:
    """Build a MagicMock session that mimics the surface the retriever uses.

    session.execute(text(...)).fetchall() → rows
    session.query(CodeSymbol).filter(CodeSymbol.symbol_id.in_(...)).all() → symbols
    """
    session = MagicMock(name="session")
    session.execute.return_value.fetchall.return_value = rows or []
    sym_query = MagicMock()
    sym_query.filter.return_value.all.return_value = symbols or []
    session.query.side_effect = lambda model: sym_query if model is CodeSymbol else MagicMock()
    return session


# ============================================================
# Happy path
# ============================================================


def test_retrieve_returns_top_k_in_similarity_descending_order() -> None:
    """5 rows returned by the SQL query → 5 (symbol, similarity) tuples,
    ordered by similarity (1 - distance) descending."""
    repo_id = uuid4()
    s1 = _make_symbol(uuid4())
    s2 = _make_symbol(uuid4())
    s3 = _make_symbol(uuid4())
    rows = [
        SimpleNamespace(symbol_id=s1.symbol_id, distance=0.05),  # best match
        SimpleNamespace(symbol_id=s2.symbol_id, distance=0.20),
        SimpleNamespace(symbol_id=s3.symbol_id, distance=0.50),
    ]
    session = _build_session(rows=rows, symbols=[s1, s2, s3])
    client = _fake_embedding_client([_make_vec(0)])

    out = retrieve_semantic(session, repo_id, "claim text", embedding_client=client)

    assert len(out) == 3
    assert out[0][0] is s1
    assert out[0][1] == pytest.approx(0.95)
    assert out[1][0] is s2
    assert out[1][1] == pytest.approx(0.80)
    assert out[2][0] is s3
    assert out[2][1] == pytest.approx(0.50)


def test_retrieve_similarity_is_one_minus_distance() -> None:
    """The score is exactly ``1.0 - distance`` (cosine → 0..1 similarity)."""
    repo_id = uuid4()
    s = _make_symbol(uuid4())
    rows = [SimpleNamespace(symbol_id=s.symbol_id, distance=0.123)]
    session = _build_session(rows=rows, symbols=[s])
    client = _fake_embedding_client([_make_vec(0)])

    out = retrieve_semantic(session, repo_id, "x", embedding_client=client)

    assert out[0][1] == pytest.approx(1.0 - 0.123)


def test_retrieve_embeds_only_the_claim() -> None:
    """The client is called once with the claim as a single-element list."""
    repo_id = uuid4()
    session = _build_session(rows=[])
    client = _fake_embedding_client([_make_vec(0)])

    retrieve_semantic(session, repo_id, "the claim", embedding_client=client)

    assert len(client.calls) == 1
    assert client.calls[0] == ["the claim"]


def test_retrieve_passes_claim_vector_to_query() -> None:
    """The embedded claim vector is forwarded to the SQL query."""
    repo_id = uuid4()
    session = _build_session(rows=[])
    client = _fake_embedding_client([_make_vec(7)])

    retrieve_semantic(session, repo_id, "x", embedding_client=client)

    # The first positional arg to execute() is the SQL text;
    # the second is the params dict.
    params = session.execute.call_args.args[1]
    assert params["claim_vec"] == _make_vec(7)
    assert params["repository_id"] == str(repo_id)
    assert params["top_k"] == 5  # default


def test_retrieve_honours_top_k() -> None:
    """top_k=2 → query param is 2."""
    repo_id = uuid4()
    session = _build_session(rows=[])
    client = _fake_embedding_client([_make_vec(0)])

    retrieve_semantic(session, repo_id, "x", top_k=2, embedding_client=client)

    assert session.execute.call_args.args[1]["top_k"] == 2


def test_retrieve_honours_custom_repository_id() -> None:
    """The query is scoped to the requested repository_id (not the
    session's repository_id)."""
    repo_id = uuid4()
    session = _build_session(rows=[])
    client = _fake_embedding_client([_make_vec(0)])

    retrieve_semantic(session, repo_id, "x", embedding_client=client)

    assert session.execute.call_args.args[1]["repository_id"] == str(repo_id)


def test_retrieve_query_uses_pgvector_cosine_operator() -> None:
    """The SQL contains the pgvector ``<=>`` cosine operator."""
    repo_id = uuid4()
    session = _build_session(rows=[])
    client = _fake_embedding_client([_make_vec(0)])

    retrieve_semantic(session, repo_id, "x", embedding_client=client)

    sql = session.execute.call_args.args[0]
    sql_text = str(sql)
    assert "<=>" in sql_text
    # The query targets the embeddings table.
    assert "code_embeddings" in sql_text
    # It filters by repository_id.
    assert "repository_id" in sql_text
    # It orders by distance.
    assert "ORDER BY" in sql_text


# ============================================================
# Empty / edge cases
# ============================================================


def test_retrieve_empty_claim_short_circuits() -> None:
    """Empty claim → no embed call, no SQL, empty result."""
    repo_id = uuid4()
    session = _build_session(rows=[])
    client = _fake_embedding_client([_make_vec(0)])

    out = retrieve_semantic(session, repo_id, "", embedding_client=client)

    assert out == []
    assert client.calls == []
    session.execute.assert_not_called()


def test_retrieve_whitespace_only_claim_short_circuits() -> None:
    """A whitespace-only claim is also treated as empty."""
    repo_id = uuid4()
    session = _build_session(rows=[])
    client = _fake_embedding_client([_make_vec(0)])

    out = retrieve_semantic(session, repo_id, "   \n  ", embedding_client=client)

    assert out == []
    session.execute.assert_not_called()


def test_retrieve_empty_query_result_returns_empty_list() -> None:
    """No matching embeddings → empty list, no bulk-fetch."""
    repo_id = uuid4()
    session = _build_session(rows=[])
    client = _fake_embedding_client([_make_vec(0)])

    out = retrieve_semantic(session, repo_id, "x", embedding_client=client)

    assert out == []
    # The empty result short-circuits before the bulk-fetch.
    session.query.assert_not_called()


def test_retrieve_skips_orphaned_embedding_rows() -> None:
    """If an embedding's symbol_id has no matching CodeSymbol, skip it."""
    repo_id = uuid4()
    real_s = _make_symbol(uuid4())
    orphan_id = uuid4()
    rows = [
        SimpleNamespace(symbol_id=real_s.symbol_id, distance=0.1),
        SimpleNamespace(symbol_id=orphan_id, distance=0.2),
    ]
    session = _build_session(rows=rows, symbols=[real_s])  # orphan_id not in symbols
    client = _fake_embedding_client([_make_vec(0)])

    out = retrieve_semantic(session, repo_id, "x", embedding_client=client)

    # The orphan is silently dropped.
    assert len(out) == 1
    assert out[0][0] is real_s


# ============================================================
# Error handling
# ============================================================


def test_retrieve_reraises_embedding_error() -> None:
    """EmbeddingError propagates without being wrapped."""
    repo_id = uuid4()
    session = _build_session(rows=[])
    client = _fake_embedding_client(EmbeddingError("simulated outage"))

    with pytest.raises(EmbeddingError, match=r"simulated outage"):
        retrieve_semantic(session, repo_id, "x", embedding_client=client)


def test_retrieve_does_not_swallow_embedding_error() -> None:
    """The function does NOT return [] on embed failure — it raises."""
    repo_id = uuid4()
    session = _build_session(rows=[])
    client = _fake_embedding_client(EmbeddingError("boom"))

    with pytest.raises(EmbeddingError):
        retrieve_semantic(session, repo_id, "x", embedding_client=client)


# ============================================================
# Distance precision
# ============================================================


def test_retrieve_preserves_symbol_order_from_query() -> None:
    """The output order matches the SQL ORDER BY (distance ascending)."""
    repo_id = uuid4()
    s_first = _make_symbol(uuid4())
    s_second = _make_symbol(uuid4())
    s_third = _make_symbol(uuid4())
    rows = [
        SimpleNamespace(symbol_id=s_first.symbol_id, distance=0.10),
        SimpleNamespace(symbol_id=s_second.symbol_id, distance=0.30),
        SimpleNamespace(symbol_id=s_third.symbol_id, distance=0.50),
    ]
    session = _build_session(rows=rows, symbols=[s_first, s_second, s_third])
    client = _fake_embedding_client([_make_vec(0)])

    out = retrieve_semantic(session, repo_id, "x", embedding_client=client)

    assert [s for s, _ in out] == [s_first, s_second, s_third]


def test_retrieve_with_very_small_distance() -> None:
    """Near-zero distance → similarity near 1.0."""
    repo_id = uuid4()
    s = _make_symbol(uuid4())
    rows = [SimpleNamespace(symbol_id=s.symbol_id, distance=1e-6)]
    session = _build_session(rows=rows, symbols=[s])
    client = _fake_embedding_client([_make_vec(0)])

    out = retrieve_semantic(session, repo_id, "x", embedding_client=client)

    assert out[0][1] == pytest.approx(1.0 - 1e-6)
