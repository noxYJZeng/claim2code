"""repositories: per-analysis GitHub repo + extracted symbols + configs + deps

Revision ID: 0003_repositories
Revises: 0002_papers
Create Date: 2026-08-11 12:00:00.000000

Adds the four Phase-3 tables that sit behind ``POST /repositories``:

  * ``repositories`` — the parent row per cloned repo. Mirrors the
    ``papers`` shape (FK → ``analysis_states.analysis_id`` with
    ``ON DELETE CASCADE``, status lifecycle, default-branch / clone
    path metadata). The canonical (owner, name) pair is UNIQUE so a
    re-clone against the same repo returns the existing row rather
    than duplicating storage.

  * ``code_symbols`` — one row per parsed symbol (function, class,
    method, import, constant, config_schema). Line / column are
    1-indexed / 0-indexed respectively to match what tree-sitter
    returns and what the frontend wants to render. The
    ``parent_symbol_id`` self-FK captures the class → method
    hierarchy; ``caller_symbol_id`` is the name-resolved call graph
    (see ``phase-3-design.md`` §2.2).

  * ``repository_configs`` — one row per parsed config file
    (YAML / JSON / TOML / INI). The parsed tree is stored as JSONB
    so Phase 4-5 can query without re-reading the file.

  * ``repository_dependencies`` — flattened dependency rows from
    ``requirements*.txt``, ``poetry.lock``, ``package-lock.json``,
    ``Pipfile.lock``. Phase 6's verdict logic joins on these.

Index list:

  * ``ix_repositories_analysis_id`` — fast lookup by analysis.
  * ``uq_repositories_owner_name`` — UNIQUE on (owner, name);
    dedup so re-cloning the same repo is idempotent.
  * ``ix_code_symbols_repository_id`` — the symbol fetch endpoint.
  * ``ix_code_symbols_file_path`` — group symbols by file for the
    frontend tree view.
  * ``ix_code_symbols_name`` — name-based call graph resolution.
  * ``ix_repository_configs_repository_id`` — the configs fetch.
  * ``ix_repository_dependencies_repository_id`` — the deps fetch.

Phase 4 will add a pgvector ``embedding VECTOR(1536)`` column on
``code_symbols`` via a follow-up migration (one migration per phase
keeps each diff reviewable).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_repositories"
down_revision: str | None = "0002_papers"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ---- repositories ----
    op.create_table(
        "repositories",
        sa.Column(
            "repository_id",
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
            "url",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "owner",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "name",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "default_branch",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("'main'"),
        ),
        sa.Column(
            "clone_path",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "n_files",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "n_symbols",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "cloned_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "index_status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "indexed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "error",
            sa.Text(),
            nullable=True,
        ),
        sa.UniqueConstraint("owner", "name", name="uq_repositories_owner_name"),
    )

    op.create_index(
        "ix_repositories_analysis_id",
        "repositories",
        ["analysis_id"],
    )

    # ---- code_symbols ----
    op.create_table(
        "code_symbols",
        sa.Column(
            "symbol_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "repository_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("repositories.repository_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "kind",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "language",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "file_path",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "start_line",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "start_col",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "end_line",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "end_col",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "name",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "signature",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "docstring",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "parent_symbol_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("code_symbols.symbol_id", ondelete="SET NULL"),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_code_symbols_repository_id",
        "code_symbols",
        ["repository_id"],
    )
    op.create_index(
        "ix_code_symbols_file_path",
        "code_symbols",
        ["repository_id", "file_path"],
    )
    op.create_index(
        "ix_code_symbols_name",
        "code_symbols",
        ["repository_id", "name"],
    )

    # ---- repository_configs ----
    op.create_table(
        "repository_configs",
        sa.Column(
            "config_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "repository_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("repositories.repository_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "file_path",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "format",
            sa.String(length=16),
            nullable=False,
        ),
        sa.Column(
            "size_bytes",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "content_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "error",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "extracted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_index(
        "ix_repository_configs_repository_id",
        "repository_configs",
        ["repository_id"],
    )

    # ---- repository_dependencies ----
    op.create_table(
        "repository_dependencies",
        sa.Column(
            "dep_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "repository_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("repositories.repository_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "ecosystem",
            sa.String(length=16),
            nullable=False,
        ),
        sa.Column(
            "name",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "version_spec",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "source_file",
            sa.Text(),
            nullable=False,
        ),
    )

    op.create_index(
        "ix_repository_dependencies_repository_id",
        "repository_dependencies",
        ["repository_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_repository_dependencies_repository_id", table_name="repository_dependencies")
    op.drop_table("repository_dependencies")

    op.drop_index("ix_repository_configs_repository_id", table_name="repository_configs")
    op.drop_table("repository_configs")

    op.drop_index("ix_code_symbols_name", table_name="code_symbols")
    op.drop_index("ix_code_symbols_file_path", table_name="code_symbols")
    op.drop_index("ix_code_symbols_repository_id", table_name="code_symbols")
    op.drop_table("code_symbols")

    op.drop_index("ix_repositories_analysis_id", table_name="repositories")
    op.drop_table("repositories")
