"""code_symbols: drop the B-tree index on lex_text

Revision ID: 0007_drop_lex_text_btree
Revises: 0006_analysis_state_related_work
Create Date: 2026-10-26 00:00:00.000000

0005 added ``ix_code_symbols_lex_text`` (a plain B-tree over the
generated ``lex_text`` column) "for LIKE-style debug queries". A
B-tree entry is capped at 2704 bytes, and ``lex_text`` concatenates
name + signature + docstring + file_path, so any symbol with a long
docstring fails the INSERT with ``ProgramLimitExceeded`` and aborts
repository ingestion (e.g. ``microsoft/LoRA``).

Nothing queries ``lex_text`` through that index — the lexical
retriever ranks on ``lex_tsv`` via ``ix_code_symbols_lex_tsv_gin`` —
so we drop it. The downgrade recreates it as 0005 had it.
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0007_drop_lex_text_btree"
down_revision: str | None = "0006_analysis_state_related_work"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_code_symbols_lex_text")


def downgrade() -> None:
    op.create_index(
        "ix_code_symbols_lex_text",
        "code_symbols",
        ["lex_text"],
    )
