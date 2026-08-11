"""code_embeddings: per-symbol pgvector rows + HNSW index

Revision ID: 0004_code_embeddings
Revises: 0003_repositories
Create Date: 2026-08-12 00:00:00.000000

Phase 4 (Semantic Indexing) adds the embedding layer that turns
each ``CodeSymbol`` into a vector and lets the retriever rank
candidates by cosine similarity.

The migration does two things:

  1. **Adds ``repositories.embedding_model``** — a VARCHAR(64) so
     the indexer can record which model version produced the rows
     in this repo. NULL means "not yet indexed". We put this on
     ``repositories`` rather than on ``code_embeddings`` so the
     retriever can filter by repository without a join.

  2. **Creates ``code_embeddings``** — one row per
     ``(symbol_id, model_name)`` tuple. The composite
     UNIQUE on (symbol_id, model_name) lets us upgrade the
     embedding model without destroying old rows: the indexer
     just writes new rows under the new model name, and the
     retriever filters by the active model.

Schema notes
------------

  - ``embedding VECTOR(1536)`` — hard-coded to match
    ``Settings.openai_embed_dim`` (Phase 1 polish). If the user
    changes dimensionality the migration must be revisited.
  - Denormalised ``repository_id`` + ``analysis_id`` so the
    retriever's cosine query is a one-table SELECT (no join
    through ``code_symbols``).
  - ``ON DELETE CASCADE`` on all 3 FKs. Dropping a repository
    drops its embeddings; dropping an analysis drops them all.

Index list
----------

  - ``ix_code_embeddings_repository_id`` — repo-scoped lookups.
  - ``ix_code_embeddings_analysis_id`` — analysis-scoped lookups.
  - ``ix_code_embeddings_hnsw`` — the cosine-distance HNSW index.
    ``m=16`` and ``ef_construction=64`` are the literature
    defaults; pgvector's HNSW needs both before it can serve
    cosine queries. The query-side ``ef_search`` is left at
    pgvector's default (40) and is tuned per-call by the
    retriever in commit 5 when needed.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0004_code_embeddings"
down_revision: str | None = "0003_repositories"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Add embedding_model to repositories so the indexer can
    #    record which model version produced the rows.
    op.add_column(
        "repositories",
        sa.Column(
            "embedding_model",
            sa.String(length=64),
            nullable=True,
        ),
    )

    # 2. Create code_embeddings.
    op.create_table(
        "code_embeddings",
        sa.Column(
            "embedding_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "symbol_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("code_symbols.symbol_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "repository_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("repositories.repository_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "analysis_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("analysis_states.analysis_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "embedding",
            Vector(dim=1536),
            nullable=False,
        ),
        sa.Column(
            "model_name",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "embedded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "symbol_id",
            "model_name",
            name="uq_code_embeddings_symbol_model",
        ),
    )

    op.create_index(
        "ix_code_embeddings_repository_id",
        "code_embeddings",
        ["repository_id"],
    )
    op.create_index(
        "ix_code_embeddings_analysis_id",
        "code_embeddings",
        ["analysis_id"],
    )

    # HNSW index on the embedding column. ``vector_cosine_ops``
    # is the cosine-distance operator class; ``m`` and
    # ``ef_construction`` are the index-build knobs.
    op.execute(
        "CREATE INDEX ix_code_embeddings_hnsw ON code_embeddings "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    # Drop in reverse order — HNSW first (depends on the table),
    # then B-tree indexes, then the table, then the column.
    op.execute("DROP INDEX IF EXISTS ix_code_embeddings_hnsw")
    op.drop_index("ix_code_embeddings_analysis_id", table_name="code_embeddings")
    op.drop_index("ix_code_embeddings_repository_id", table_name="code_embeddings")
    op.drop_table("code_embeddings")
    op.drop_column("repositories", "embedding_model")
