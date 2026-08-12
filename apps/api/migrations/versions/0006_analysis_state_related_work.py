"""analysis_states: related_work JSONB column (Phase 7)

Revision ID: 0006_analysis_state_related_work
Revises: 0005_code_symbols_lexical
Create Date: 2026-08-12 00:00:00.000000

Phase 7 (Related Work) persists per-claim related-work results to a
new ``related_work: JSONB`` column on ``analysis_states``. Each
entry is a :class:`RelatedWorkEntry` (claim_id + ranked papers +
generated_at + source).

JSONB vs a new ``related_papers`` table
--------------------------------------

  - **JSONB is consistent with Phase 6's precedent.** Phase 6 ships
    `verifications: JSONB` for the same reason: the shape is still
    evolving (Phase 10 eval may add per-source aggregation, in
    which case a table split is reasonable) and the entries
    reference the same row's ``claims`` (no FK needed).
  - **No cross-row queries in v1.** Phase 7 only reads / writes
    ``related_work`` for the current ``analysis_id``; Phase 10's
    eval can read from a join of ``analysis_states`` rows without
    needing a per-paper table.
  - **One-migration cost.** A new table would also need a new
    Alembic revision + a new ORM model; Phase 7 ships a single
    column add and rides on the existing JSONB columns.

The column carries ``nullable=False`` + ``server_default='[]'``
+ ``default=list`` to mirror the other ``analysis_states``
JSONB columns (``claims``, ``symbols``, ``mappings``,
``verifications``). Existing rows are backfilled to ``[]`` by the
``server_default`` so the migration is non-disruptive.

Why no index on ``related_work``
--------------------------------

The JSONB is read / written as a unit (full replacement via
``state.related_work = [entry if e.claim_id != cid else new_entry
for e in state.related_work] + [new_entry]``); no element-level
filter runs in v1. A GIN index on
``jsonb_path_ops(claim_id)`` is a one-line add in a follow-up if
Phase 10 eval needs cross-analysis aggregation.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_analysis_state_related_work"
down_revision: str | None = "0005_code_symbols_lexical"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add ``related_work`` JSONB column to ``analysis_states``.

    The ``server_default='[]'`` backfills existing rows to an
    empty list so the migration is non-disruptive. Phase 7 code
    populates the column via the workflow's ``state.related_work =
    ...`` assignment (mirrors Phase 6's pattern).
    """
    op.add_column(
        "analysis_states",
        sa.Column(
            "related_work",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )


def downgrade() -> None:
    """Drop the ``related_work`` column. No data loss to migrate away."""
    op.drop_column("analysis_states", "related_work")