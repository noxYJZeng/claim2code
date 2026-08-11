"""ORM models. Phase 1 ships :class:`AnalysisState`; Phase 2 adds
:class:`Paper`; Phase 3 adds :class:`Repository` / :class:`CodeSymbol` /
:class:`RepositoryConfig` / :class:`RepositoryDependency`; Phase 6
will add :class:`ClaimCodeMapping` and :class:`Verification` (see
``docs/specs/spec-summary.md`` §5).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from claim2code_api.db.base import Base


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
        UUID(as_uuid=True),
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
        JSONB, nullable=False, default=list, server_default="[]"
    )
    symbols: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    mappings: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    verifications: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    errors: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
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
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
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
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
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
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    repository_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
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
    parent_symbol_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

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
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    repository_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    format: Mapped[str] = mapped_column(String(16), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    content_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
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
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    repository_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
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
