"""Smoke test for the migration stack.

Applies the migrations against the test database (provided via the
``DATABASE_URL`` environment variable in CI) and confirms the
``analysis_states`` table exists with the expected columns.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from claim2code_api.app.config import get_settings

pytestmark = pytest.mark.integration


def test_analysis_states_table_exists() -> None:
    engine = create_engine(get_settings().database_url, future=True)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'analysis_states' ORDER BY ordinal_position"
            )
        ).fetchall()
    names = {r[0] for r in rows}
    assert "analysis_id" in names
    assert "paper_status" in names
    assert "repository_status" in names
    assert "claims" in names
    assert "started_at" in names


def test_pgvector_extension_is_installed() -> None:
    engine = create_engine(get_settings().database_url, future=True)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        ).fetchall()
    assert rows, "pgvector extension should be installed by migration 0001"
