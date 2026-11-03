"""analysis_states: reference_checks JSONB column

Revision ID: 0008_reference_checks
Revises: 0007_drop_lex_text_btree
Create Date: 2026-11-02 00:00:00.000000

One :class:`ReferenceCheck` per claim (claim_id, status, the
reference code, the per-input outcomes). Same JSONB-with-'[]'
default pattern as ``verifications`` and ``related_work``.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_reference_checks"
down_revision: str | None = "0007_drop_lex_text_btree"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "analysis_states",
        sa.Column(
            "reference_checks",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("analysis_states", "reference_checks")
