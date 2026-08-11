"""init: pgvector extension + analysis_states table

Revision ID: 0001_init
Revises:
Create Date: 2026-08-10 00:00:00.000000

Phase 1 ships the first schema: the ``vector`` extension (so pgvector
becomes available for Phase 4) and the ``analysis_states`` table that
mirrors the ``AnalysisState`` shape in ``docs/architecture/data-flow.md``
§4. The wide JSONB columns let us evolve nested object schemas in
Phase 2-6 without further DDL.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_init"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # pgvector — used by the embedding tables in Phase 4.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # uuid-ossp — supplies ``uuid_generate_v4()`` which 0002 + 0003
    # use as the server-side default for primary keys. Postgres 13+
    # has ``gen_random_uuid()`` built-in via pgcrypto, but ``uuid-ossp``
    # is the more explicit option and matches the design doc. Enable
    # it here so the very next migration can reference the function.
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    op.create_table(
        "analysis_states",
        sa.Column(
            "analysis_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "paper_status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "repository_status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "claims",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "symbols",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "mappings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "verifications",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "errors",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_analysis_states_started_at",
        "analysis_states",
        ["started_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_analysis_states_started_at", table_name="analysis_states")
    op.drop_table("analysis_states")
    # We intentionally do NOT drop the vector / uuid-ossp extensions —
    # they are cluster-wide resources that may be in use by other
    # databases.
