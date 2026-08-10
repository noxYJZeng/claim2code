"""ORM models. Phase 1 ships only :class:`AnalysisState`; later phases add
``Paper``, ``Repository``, ``CodeSymbol``, ``Claim``, ``ClaimCodeMapping``,
``Verification`` (see ``docs/specs/spec-summary.md`` §5).
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
