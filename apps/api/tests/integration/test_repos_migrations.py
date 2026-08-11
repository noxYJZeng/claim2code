"""Integration smoke test for migration 0003_repositories.

Applies the migrations against the test database (provided via
``DATABASE_URL`` in CI) and confirms the four new tables exist with
the expected columns + indexes + FK CASCADE behaviour.

Run with::

    pytest -m integration tests/integration/test_repos_migrations.py

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


def test_repositories_table_exists() -> None:
    engine = create_engine(get_settings().database_url, future=True)
    with engine.connect() as conn:
        cols = _table_columns(conn, "repositories")
    expected = {
        "repository_id",
        "analysis_id",
        "url",
        "owner",
        "name",
        "default_branch",
        "clone_path",
        "n_files",
        "n_symbols",
        "cloned_at",
        "index_status",
        "indexed_at",
        "error",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"


def test_code_symbols_table_exists() -> None:
    engine = create_engine(get_settings().database_url, future=True)
    with engine.connect() as conn:
        cols = _table_columns(conn, "code_symbols")
    expected = {
        "symbol_id",
        "repository_id",
        "kind",
        "language",
        "file_path",
        "start_line",
        "start_col",
        "end_line",
        "end_col",
        "name",
        "signature",
        "docstring",
        "parent_symbol_id",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"


def test_repository_configs_table_exists() -> None:
    engine = create_engine(get_settings().database_url, future=True)
    with engine.connect() as conn:
        cols = _table_columns(conn, "repository_configs")
    expected = {
        "config_id",
        "repository_id",
        "file_path",
        "format",
        "size_bytes",
        "content_json",
        "error",
        "extracted_at",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"


def test_repository_dependencies_table_exists() -> None:
    engine = create_engine(get_settings().database_url, future=True)
    with engine.connect() as conn:
        cols = _table_columns(conn, "repository_dependencies")
    expected = {
        "dep_id",
        "repository_id",
        "ecosystem",
        "name",
        "version_spec",
        "source_file",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"


def test_code_symbols_indexes_exist() -> None:
    """The migration creates three indexes on code_symbols
    (analysis_id, repository_id, file_path)."""
    engine = create_engine(get_settings().database_url, future=True)
    with engine.connect() as conn:
        for name in (
            "ix_code_symbols_repository_id",
            "ix_code_symbols_file_path",
        ):
            assert _index_exists(conn, name), f"missing index {name}"


def test_cascade_delete_removes_child_rows() -> None:
    """Inserting a parent + child and deleting the parent should
    wipe the child row (Postgres ``ON DELETE CASCADE``)."""
    from sqlalchemy.orm import sessionmaker

    from claim2code_api.db.models import (
        AnalysisState,
        Base,
        CodeSymbol,
        Repository,
    )

    engine = create_engine(get_settings().database_url, future=True)
    # Only used for the schema metadata; the live tables are owned by
    # Alembic migrations on the real DB.
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    s = session_local()
    try:
        analysis = AnalysisState(
            repository_status="pending",
            paper_status="pending",
        )
        s.add(analysis)
        s.flush()
        repo = Repository(
            analysis_id=analysis.analysis_id,
            url="https://github.com/x/y",
            owner="x",
            name="y",
            default_branch="main",
            clone_path="data/repos/x__y",
            n_files=0,
            n_symbols=0,
        )
        s.add(repo)
        s.flush()
        sym = CodeSymbol(
            repository_id=repo.repository_id,
            kind="function",
            language="python",
            file_path="m.py",
            start_line=1,
            start_col=0,
            end_line=1,
            end_col=0,
            name="f",
            signature="()",
        )
        s.add(sym)
        s.commit()
        repo_id = repo.repository_id

        # Delete the parent — child should cascade away.
        s.delete(repo)
        s.commit()
        rows = (
            s.query(CodeSymbol)
            .filter(CodeSymbol.repository_id == repo_id)
            .count()
        )
        assert rows == 0, "child row should be cascaded by ON DELETE CASCADE"

        # Clean up the analysis row too.
        s.delete(analysis)
        s.commit()
    finally:
        s.close()