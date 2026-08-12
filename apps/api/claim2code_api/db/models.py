"""ORM models. Phase 1 ships :class:`AnalysisState`; Phase 2 adds
:class:`Paper`; Phase 3 adds :class:`Repository` / :class:`CodeSymbol` /
:class:`RepositoryConfig` / :class:`RepositoryDependency`; Phase 4
adds :class:`CodeEmbedding`; Phase 6 will add :class:`ClaimCodeMapping`
and :class:`Verification` (see ``docs/specs/spec-summary.md`` §5).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CHAR, BigInteger, DateTime, Integer, String, Text, TypeDecorator, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.orm import Mapped, mapped_column

from claim2code_api.db.base import Base

# pgvector's SQLAlchemy type — adds ``VECTOR(1536)`` columns. The
# driver-side adapter is required for psycopg to round-trip the
# vector type; the Postgres extension itself is enabled by 0001_init.
try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover — keeps SQLite-only test envs importable
    Vector = None  # type: ignore[assignment,misc]

# JSONB → JSON on SQLite so the ORM compiles to an in-memory
# database for unit tests (integration tests run against the real
# Postgres JSONB and gain the same code path).
_JSONB = JSONB().with_variant(SQLITE_JSON(), "sqlite")


class _UuidDecorator(TypeDecorator):
    """A UUID type that uses Postgres' UUID on prod and CHAR(36) on SQLite.

    The CHAR(36) variant needs an explicit ``process_bind_param`` /
    ``process_result_value`` so SQLite can round-trip a Python
    ``uuid.UUID`` (which sqlite3 doesn't natively understand).
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):  # type: ignore[override]
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):  # type: ignore[override]
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return str(value)

    def process_result_value(self, value, dialect):  # type: ignore[override]
        if value is None:
            return None
        import uuid as _uuid

        if isinstance(value, _uuid.UUID):
            return value
        return _uuid.UUID(str(value))


_UUID = _UuidDecorator()


class AnalysisState(Base):
    """Persisted state for one paper + repository verification run.

    Mirrors the ``AnalysisState`` shape in ``docs/architecture/data-flow.md``
    §4. The wide JSONB columns (``claims``, ``symbols``, …) are deliberate
    for Phase 1 — the schema of those nested objects will only stabilise
    in Phase 2-6, and JSONB lets us evolve without DDL churn. We will
    split them into proper tables once the shapes are stable.
    """

    __tablename__ = "analysis_states"

    analysis_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    paper_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", server_default="pending"
    )
    repository_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", server_default="pending"
    )
    claims: Mapped[list[dict[str, Any]]] = mapped_column(
        _JSONB, nullable=False, default=list, server_default="[]"
    )
    symbols: Mapped[list[dict[str, Any]]] = mapped_column(
        _JSONB, nullable=False, default=list, server_default="[]"
    )
    mappings: Mapped[list[dict[str, Any]]] = mapped_column(
        _JSONB, nullable=False, default=list, server_default="[]"
    )
    verifications: Mapped[list[dict[str, Any]]] = mapped_column(
        _JSONB, nullable=False, default=list, server_default="[]"
    )
    errors: Mapped[list[dict[str, Any]]] = mapped_column(
        _JSONB, nullable=False, default=list, server_default="[]"
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        return (
            f"AnalysisState(analysis_id={self.analysis_id!s}, "
            f"paper_status={self.paper_status!r}, repository_status={self.repository_status!r})"
        )


class Paper(Base):
    """One uploaded PDF, keyed by sha256 for dedup.

    Phase 2 is 1:1 — one paper per analysis — but the
    ``analysis_id`` FK is left in place so Phase 3 (multi-PDF
    meta-analyses) can widen it to a list without DDL churn on this
    side. See ``migrations/versions/0002_papers.py`` for the schema
    definition and the ``parse_status`` lifecycle:

      pending → parsing → extracting → done
                              ↘ error

    The PDF bytes themselves live on disk at
    ``data/uploads/<sha256>.pdf``; this row carries only metadata.
    """

    __tablename__ = "papers"

    paper_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        # The FK + CASCADE are created by the migration; SQLAlchemy
        # is told about the constraint here so ORM-level cascades
        # match DB-level cascades.
        nullable=False,
    )
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    n_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    parse_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", server_default="pending"
    )
    parsed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        return (
            f"Paper(paper_id={self.paper_id!s}, analysis_id={self.analysis_id!s}, "
            f"sha256={self.sha256[:8]}..., parse_status={self.parse_status!r})"
        )


class Repository(Base):
    """One cloned GitHub repository bound to an analysis.

    Phase 3 is 1:1 — one repository per analysis (mirroring the
    1 paper-per-analysis Phase 2 constraint). ``analysis_id`` FK
    with ``ON DELETE CASCADE`` is set up by the migration; the
    ``(owner, name)`` UNIQUE constraint is a soft dedup — re-cloning
    the same repo against a different analysis is rejected at the
    migration / unique-constraint layer.

    Lifecycle (see :class:`Paper` for the parallel paper workflow):

      pending → cloning → parsing → done
                             ↘ error

    The clone itself lives on disk at
    ``data/repos/<owner>__<repo>/``; this row carries metadata only.
    See ``migrations/versions/0003_repositories.py`` for the schema.
    """

    __tablename__ = "repositories"

    repository_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        nullable=False,
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    owner: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    default_branch: Mapped[str] = mapped_column(
        String(128), nullable=False, default="main", server_default="main"
    )
    clone_path: Mapped[str] = mapped_column(Text, nullable=False)
    n_files: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_symbols: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cloned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    index_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", server_default="pending"
    )
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Phase 4 — which embedding model produced the rows in
    # ``code_embeddings`` for this repo. NULL = not yet indexed.
    # The retriever filters on this so a model upgrade doesn't
    # mix stale and fresh vectors in the same query result.
    embedding_model: Mapped[str | None] = mapped_column(String(64), nullable=True)

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        return (
            f"Repository(repository_id={self.repository_id!s}, "
            f"owner={self.owner!r}, name={self.name!r}, "
            f"index_status={self.index_status!r})"
        )


class CodeSymbol(Base):
    """One extracted code symbol (function, class, method, import, …).

    The ``parent_symbol_id`` self-FK captures class → method hierarchy
    — set during the workflow layer's two-pass walk. ``caller_symbol_id``
    is reserved for the Phase 3 v1 name-resolved call graph (currently
    always NULL; commit 5 wires it in).

    See ``migrations/versions/0003_repositories.py`` for the schema
    and ``claim2code_api.repos.schemas.CodeSymbol`` for the Pydantic
    representation the parser produces.
    """

    __tablename__ = "code_symbols"

    symbol_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    repository_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    language: Mapped[str] = mapped_column(String(32), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    start_col: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_col: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    signature: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    docstring: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_symbol_id: Mapped[uuid.UUID | None] = mapped_column(_UUID, nullable=True)

    # Phase 5 — lexical-search support columns. Generated columns
    # on Postgres (see migration 0005); on SQLite (unit tests)
    # they are plain Text columns and remain None unless explicitly
    # populated. The retriever reads them only on Postgres.
    #
    # We deliberately type them as ``Mapped[str | None]`` so the
    # ORM compile passes on both backends; the column exists in
    # the table but is populated by the database, not the
    # application. Tests that exercise the lexical scorer mock the
    # SQL layer and don't touch this field.
    lex_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    lex_tsv: Mapped[object | None] = mapped_column(  # type: ignore[assignment]
        # ``object`` because pgvector / tsvector have no generic
        # SQLAlchemy type; the migration handles the schema side.
        Text,
        nullable=True,
    )

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        return (
            f"CodeSymbol(symbol_id={self.symbol_id!s}, "
            f"kind={self.kind!r}, name={self.name!r}, file={self.file_path!r})"
        )


class RepositoryConfig(Base):
    """One parsed config file (YAML / JSON / TOML / INI).

    ``content_json`` holds the parsed tree as JSONB so Phase 4-5 can
    query without re-reading the file. Unparseable files still land
    here with ``content_json = {}`` and the parse error in ``error``.
    """

    __tablename__ = "repository_configs"

    config_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    repository_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    format: Mapped[str] = mapped_column(String(16), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    content_json: Mapped[dict[str, Any]] = mapped_column(
        _JSONB, nullable=False, default=dict, server_default="{}"
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        return (
            f"RepositoryConfig(config_id={self.config_id!s}, "
            f"file_path={self.file_path!r}, format={self.format!r})"
        )


class RepositoryDependency(Base):
    """One flattened dependency row from a lockfile / requirements file.

    Phase 6 normalises ``version_spec`` (``==2.1.0``,
    ``^1.0.0``, etc.) for cross-checking against paper text.
    """

    __tablename__ = "repository_dependencies"

    dep_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    repository_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        nullable=False,
    )
    ecosystem: Mapped[str] = mapped_column(String(16), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    version_spec: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    source_file: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        return (
            f"RepositoryDependency(dep_id={self.dep_id!s}, "
            f"ecosystem={self.ecosystem!r}, name={self.name!r}, "
            f"version_spec={self.version_spec!r})"
        )


class CodeEmbedding(Base):
    """One embedding vector for one ``CodeSymbol`` under one model.

    Phase 4 (Semantic Indexing) writes one row per symbol at
    index time. The composite UNIQUE on ``(symbol_id, model_name)``
    lets us upgrade the embedding model without destructive
    migrations: re-indexing writes new rows under the new model
    name; the old rows stay (re-buildable from the symbol row)
    but become unreferenced.

    The ``embedding`` column is ``VECTOR(1536)`` per
    ``Settings.openai_embed_dim``. Mismatch (e.g. swapping
    models with a different dim) raises during schema creation;
    the retriever does not silently coerces.

    ``analysis_id`` and ``repository_id`` are denormalised so the
    cosine query in :mod:`claim2code_api.indexing.retriever` is
    a one-table SELECT.

    See ``migrations/versions/0004_code_embeddings.py`` for the
    schema and the HNSW index.
    """

    __tablename__ = "code_embeddings"

    embedding_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    symbol_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        nullable=False,
    )
    repository_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        nullable=False,
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        _UUID,
        nullable=False,
    )
    # ``Vector`` is None when pgvector is not installed (the
    # SQLite-only test environment). The schema declaration is
    # in the migration; this is the runtime view. The repr does
    # not include the vector content to avoid megabytes of
    # output in logs.
    embedding: Mapped[object] = mapped_column(  # type: ignore[assignment]
        Vector(dim=1536) if Vector is not None else Text,  # type: ignore[arg-type]
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    embedded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        return (
            f"CodeEmbedding(embedding_id={self.embedding_id!s}, "
            f"symbol_id={self.symbol_id!s}, model_name={self.model_name!r})"
        )
