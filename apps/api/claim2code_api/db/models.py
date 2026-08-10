"""ORM models. Phase 1 ships only :class:`AnalysisState`; later phases add
``Paper``, ``Repository``, ``CodeSymbol``, ``Claim``, ``ClaimCodeMapping``,
``Verification`` (see ``docs/specs/spec-summary.md`` §5).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, func
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
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        return (
            f"AnalysisState(analysis_id={self.analysis_id!s}, "
            f"paper_status={self.paper_status!r}, repository_status={self.repository_status!r})"
        )
