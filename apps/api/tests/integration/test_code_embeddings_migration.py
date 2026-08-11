"""Integration smoke test for migration 0004_code_embeddings.

Applies the migrations against the test database (provided via
``DATABASE_URL`` in CI) and confirms the new ``code_embeddings``
table + the ``embedding_model`` column on ``repositories`` exist
with the expected shape + the HNSW index is registered.

Run with::

    pytest -m integration tests/integration/test_code_embeddings_migration.py

Requires a real Postgres+pgvector instance with the ``claim2code``
role + database (the GitHub Actions ``backend-test`` job spins one
up via the ``services`` block).
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from claim2code_api.app.config import get_settings

pytestmark = pytest.mark.integration


def _table_columns(conn, table: str) -> set[str]:
    rows = conn.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = :t"
        ),
        {"t": table},
    ).fetchall()
    return {r[0] for r in rows}


def _index_exists(conn, index_name: str) -> bool:
    rows = conn.execute(
        text("SELECT 1 FROM pg_indexes WHERE indexname = :n"),
        {"n": index_name},
    ).fetchall()
    return bool(rows)


def _index_uses_hnsw(conn, index_name: str) -> bool:
    """Returns True if the named index is an HNSW index (not a B-tree)."""
    rows = conn.execute(
        text(
            "SELECT 1 FROM pg_indexes "
            "WHERE indexname = :n AND indexdef ILIKE '%USING hnsw%'"
        ),
        {"n": index_name},
    ).fetchall()
    return bool(rows)


def test_repositories_gains_embedding_model_column() -> None:
    """The 0004 migration adds embedding_model to repositories."""
    engine = create_engine(get_settings().database_url, future=True)
    with engine.connect() as conn:
        cols = _table_columns(conn, "repositories")
    assert "embedding_model" in cols, "repositories.embedding_model missing"


def test_code_embeddings_table_exists() -> None:
    """All 7 columns of code_embeddings are present."""
    engine = create_engine(get_settings().database_url, future=True)
    with engine.connect() as conn:
        cols = _table_columns(conn, "code_embeddings")
    expected = {
        "embedding_id",
        "symbol_id",
        "repository_id",
        "analysis_id",
        "embedding",
        "model_name",
        "embedded_at",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"


def test_code_embeddings_btree_indexes_exist() -> None:
    """The two B-tree indexes (repository_id, analysis_id) are created."""
    engine = create_engine(get_settings().database_url, future=True)
    with engine.connect() as conn:
        for name in (
            "ix_code_embeddings_repository_id",
            "ix_code_embeddings_analysis_id",
        ):
            assert _index_exists(conn, name), f"missing index {name}"


def test_code_embeddings_hnsw_index_exists() -> None:
    """The cosine-distance HNSW index is created."""
    engine = create_engine(get_settings().database_url, future=True)
    with engine.connect() as conn:
        assert _index_exists(conn, "ix_code_embeddings_hnsw"), "HNSW index missing"
        assert _index_uses_hnsw(conn, "ix_code_embeddings_hnsw"), (
            "ix_code_embeddings_hnsw exists but is not an HNSW index"
        )


def test_unique_constraint_on_symbol_and_model() -> None:
    """The composite UNIQUE (symbol_id, model_name) is in place."""
    engine = create_engine(get_settings().database_url, future=True)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT 1 FROM information_schema.table_constraints "
                "WHERE table_name = 'code_embeddings' "
                "AND constraint_name = 'uq_code_embeddings_symbol_model'"
            )
        ).fetchall()
    assert rows, "uq_code_embeddings_symbol_model UNIQUE constraint missing"
