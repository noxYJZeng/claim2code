"""code_symbols: lexical search columns + GIN + trigram indexes

Revision ID: 0005_code_symbols_lexical
Revises: 0004_code_embeddings
Create Date: 2026-08-12 00:00:00.000000

Phase 5 (Hybrid Retrieval) adds the lexical search leg that runs
alongside Phase 4's semantic leg. The retriever's hybrid ranker
fuses three legs (semantic / lexical / structural) with
Reciprocal Rank Fusion; this migration supplies the storage for
the lexical leg.

The migration does three things:

  1. **Adds ``lex_text`` + ``lex_tsv`` generated columns** on
     ``code_symbols``. Both are ``GENERATED ALWAYS AS … STORED``
     so Postgres recomputes them automatically on INSERT / UPDATE
     — no application-side logic, no extra Python deps.
     ``lex_tsv`` uses ``setweight()`` so symbol name is weighted
     more than signature, which is weighted more than docstring,
     which is weighted more than file_path. The default
     ``english`` config handles lowercasing + Porter stemming +
     stopwords.
  2. **Adds a GIN index on ``lex_tsv``** — Postgres's full-text
     search index. Cheap to maintain; ``tsvector @@ tsquery``
     uses it to answer "which symbols contain at least one of the
     claim's tokens" in milliseconds.
  3. **Enables ``pg_trgm`` + adds a GIN trigram index on
     ``code_symbols.name``** — the same pattern the trigram
     extension uses for ``LIKE '%foo%'`` queries. The Phase 6
     ``ModelNameNormalizer`` will use this for fuzzy model-name
     matching; Phase 5 ships it now so we don't migrate twice.

Why generated columns
---------------------

A stored generated column is recomputed on every row write and
stored on disk. Queries read the stored value as if it were a
regular column. This means:

  - No application-side denormalisation (the indexer doesn't
    need to know about lex_text / lex_tsv).
  - The lex_text / lex_tsv always matches name + signature +
    docstring + file_path by construction — there is no
    possibility of drift.
  - The ORM can declare the column as ``Mapped[str | None]``
    (Postgres only; SQLite ignores it) and tests can still
    exercise the schema on SQLite.

Field weights
-------------

The ``tsvector`` carries weight labels (A/B/C/D) so the rank
function can favour name matches over file_path matches when
both happen to overlap with the claim's tokens. The defaults
follow Postgres conventions; users can override in
``Settings`` later if the eval shows field weights matter.

Why pg_trgm in 0005
-------------------

``pg_trgm`` is in the base ``pgvector/pgvector:pg16`` image (it
ships with Postgres 16), so enabling it is a no-op on
production. ``CREATE EXTENSION IF NOT EXISTS`` makes the
migration idempotent on any database where it was already
enabled. We do this in 0005 (not 0001) so 0001's scope stays
"the smallest viable extension set for Phase 1" — pg_trgm is
only needed once we have a fuzzy-name matching requirement.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_code_symbols_lexical"
down_revision: str | None = "0004_code_embeddings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Enable pg_trgm. Idempotent: a no-op if already enabled.
    # ``pg_trgm`` is in the base pgvector image; this just
    # makes it explicit + safe to re-run.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # 1. Add the lex_text generated column (raw concatenated text).
    #    ``STORED`` means Postgres writes it on disk; queries read
    #    it as a regular TEXT column. NULL inputs coalesce to ''
    #    so the column is never NULL.
    op.execute(
        "ALTER TABLE code_symbols ADD COLUMN lex_text TEXT "
        "GENERATED ALWAYS AS ("
        "  coalesce(name, '') || ' ' ||"
        "  coalesce(signature, '') || ' ' ||"
        "  coalesce(docstring, '') || ' ' ||"
        "  coalesce(file_path, '')"
        ") STORED"
    )

    # 2. Add the lex_tsv generated tsvector column with field
    #    weights A (name) > B (signature) > C (docstring) > D
    #    (file_path). ``setweight`` + ``to_tsvector('english', …)``
    #    + concat-with-``||`` is the canonical Postgres idiom.
    op.execute(
        "ALTER TABLE code_symbols ADD COLUMN lex_tsv tsvector "
        "GENERATED ALWAYS AS ("
        "  setweight(to_tsvector('english', coalesce(name, '')),       'A') ||"
        "  setweight(to_tsvector('english', coalesce(signature, '')),  'B') ||"
        "  setweight(to_tsvector('english', coalesce(docstring, '')),  'C') ||"
        "  setweight(to_tsvector('english', coalesce(file_path, '')),  'D')"
        ") STORED"
    )

    # 3. B-tree index on lex_text for LIKE-style debug queries.
    #    Cheap; the GIN index below handles the real workload.
    op.create_index(
        "ix_code_symbols_lex_text",
        "code_symbols",
        ["lex_text"],
    )

    # 4. GIN index on lex_tsv — the workhorse for ``@@`` queries.
    op.execute(
        "CREATE INDEX ix_code_symbols_lex_tsv_gin "
        "ON code_symbols USING GIN (lex_tsv)"
    )

    # 5. Trigram index on name for fuzzy match. Phase 6's
    #    ModelNameNormalizer will use this for "Qwen2.5-7B ↔
    #    Qwen/Qwen2.5-7B" style lookups. Phase 5 ships the index
    #    now so we don't migrate twice.
    op.execute(
        "CREATE INDEX ix_code_symbols_name_trgm "
        "ON code_symbols USING GIN (name gin_trgm_ops)"
    )


def downgrade() -> None:
    # Drop in reverse — indexes first, then columns.
    op.execute("DROP INDEX IF EXISTS ix_code_symbols_name_trgm")
    op.execute("DROP INDEX IF EXISTS ix_code_symbols_lex_tsv_gin")
    op.drop_index("ix_code_symbols_lex_text", table_name="code_symbols")
    op.drop_column("code_symbols", "lex_tsv")
    op.drop_column("code_symbols", "lex_text")
    # We do NOT drop pg_trgm — other tables / future migrations
    # may want it. Migration-level extensions are sticky.