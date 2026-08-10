"""papers: uploaded PDF metadata table

Revision ID: 0002_papers
Revises: 0001_init
Create Date: 2026-08-10 12:00:00.000000

Adds the ``papers`` table that records every uploaded PDF: the
sha256 dedup key, filename, size, parse status, and timing. The
PDF bytes themselves live on disk under ``data/uploads/<sha256>.pdf``
— see :mod:`claim2code_api.papers.storage` for the I/O.

One ``analysis_id`` can have at most one paper (1:1 in Phase 2;
Phase 3 may widen this when multi-PDF analyses land). The FK is
``ON DELETE CASCADE`` so deleting an analysis cleans up its paper
rows automatically.

The sha256 column has a UNIQUE constraint (``uq_papers_sha256``)
that doubles as a deduplication key — uploading the same paper
twice yields the same sha256, which a future ``POST /papers``
implementation can use to short-circuit the re-parse.

Index list:

  * ``ix_papers_analysis_id`` — ``GET /papers/{analysis_id}`` and
    the future ``POST /analyses/{id}/repo`` workflow join on this.
  * ``uq_papers_sha256`` — UNIQUE on sha256 (dedup lookups).

The wide columns intentionally stay out of this table — claims
are persisted into ``analysis_states.claims`` as JSONB (matching
the rest of the schema) so the verifier can read them in a single
SELECT.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_papers"
down_revision: str | None = "0001_init"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "papers",
        sa.Column(
            "paper_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "analysis_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("analysis_states.analysis_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "sha256",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "filename",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "size_bytes",
            sa.BigInteger(),
            nullable=False,
        ),
        sa.Column(
            "n_pages",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "n_chars",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "parse_status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "parsed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "error",
            sa.Text(),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_papers_analysis_id",
        "papers",
        ["analysis_id"],
    )
    op.create_index(
        "uq_papers_sha256",
        "papers",
        ["sha256"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_papers_sha256", table_name="papers")
    op.drop_index("ix_papers_analysis_id", table_name="papers")
    op.drop_table("papers")
