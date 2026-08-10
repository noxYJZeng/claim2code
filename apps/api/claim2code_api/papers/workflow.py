"""Synchronous paper-ingestion workflow.

Implements the steps from ``docs/architecture/phase-2-design.md``
§2.5. The whole workflow runs inside one request:

  1. Validate upload (size, magic bytes)
  2. Compute sha256, write to disk
  3. INSERT Paper row (parse_status='parsing')
  4. UPDATE AnalysisState.paper_status='parsing'
  5. PyMuPDF → Paper
  6. UPDATE Paper.parse_status='extracting'
  7. UPDATE AnalysisState.paper_status='extracting'
  8. LLMClient.extract_claims(paper)
  9. UPDATE AnalysisState (claims=..., status='done', completed_at=now)
 10. UPDATE Paper (parse_status='done', parsed_at=now)
 11. Return summary

Every step's exception is caught at the workflow boundary, written
into ``analysis_states.errors`` as a structured entry, and re-raised
mapped (so the router can return the right HTTP status code).

Why synchronous and not BackgroundTasks?
-----------------------------------------
Phase 2 has no concurrent load. A 180 s sync request is acceptable
for "upload a paper and wait." Adding a worker pool + idempotency
keys + status polling would be premature — Phase 5+ revisits when
load demands it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from claim2code_api.db.models import AnalysisState, Paper
from claim2code_api.llm.client import LLMClient
from claim2code_api.papers import storage
from claim2code_api.papers.exceptions import (
    ClaimExtractionError,
    PdfInvalidError,
    PdfNoTextLayerError,
    PdfTooLargeError,
)
from claim2code_api.papers.extractor import extract_claims_from_paper
from claim2code_api.papers.parser import parse_pdf
from claim2code_api.papers.schemas import Claim

logger = logging.getLogger(__name__)

# Magic bytes that every PDF starts with.
_PDF_MAGIC = b"%PDF-"

# One error blob is appended to AnalysisState.errors per failure.
_ERROR_LIMIT_PER_WORKFLOW = 1


@dataclass(frozen=True)
class WorkflowResult:
    """The summary returned by :func:`run_paper_workflow`."""

    analysis_id: UUID
    paper_id: UUID
    parse_status: str
    n_claims: int


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def run_paper_workflow(
    *,
    db: Session,
    llm: LLMClient,
    upload_bytes: bytes,
    filename: str,
    max_size_bytes: int,
) -> WorkflowResult:
    """Execute the full parse + extract pipeline for one uploaded PDF.

    :param db: a SQLAlchemy session (caller owns the lifecycle).
    :param llm: the LLM client (testable seam).
    :param upload_bytes: the raw PDF bytes from the multipart upload.
    :param filename: original filename (preserved for display).
    :param max_size_bytes: ``Settings.max_pdf_size_mb * 1024 * 1024``.

    :raises PdfTooLargeError: if the upload exceeds ``max_size_bytes``.
    :raises PdfInvalidError: if the bytes don't start with ``%PDF-``.
    :raises PdfNoTextLayerError: if PyMuPDF yields < N total chars.
    :raises ClaimExtractionError: if the LLM call fails or returns
        an unusable response.
    """
    # Validate BEFORE any DB writes so a bad upload costs nothing.
    _validate_upload(upload_bytes, max_size_bytes)
    analysis_id = _create_analysis(db)
    paper_id: UUID | None = None
    try:
        sha = storage.sha256_bytes(upload_bytes)
        # ``write_pdf`` is atomic (exclusive create). A duplicate
        # upload is detected here and surfaces as FileExistsError,
        # which we map to PdfInvalidError.
        try:
            storage.write_pdf(sha, upload_bytes)
        except FileExistsError as exc:
            raise PdfInvalidError(
                "A paper with this sha256 was already uploaded; dedup not yet wired"
            ) from exc

        paper_id = _insert_paper_row(
            db,
            analysis_id=analysis_id,
            sha256=sha,
            filename=filename,
            size_bytes=len(upload_bytes),
        )
        _set_paper_status(db, analysis_id, "parsing")

        # 5. PyMuPDF parse.
        pdf_path = storage.path_for(sha)
        paper = parse_pdf(pdf_path, filename=filename)
        db.query(Paper).filter(Paper.paper_id == paper_id).update(
            {
                Paper.n_pages: paper.n_pages,
                Paper.n_chars: paper.n_chars,
                Paper.parse_status: "extracting",
            }
        )
        _set_paper_status(db, analysis_id, "extracting")

        # 8. LLM extract.
        claims: list[Claim] = extract_claims_from_paper(llm, paper)

        # 9-10. Persist claims + mark done.
        _persist_claims(db, analysis_id, claims)
        db.query(Paper).filter(Paper.paper_id == paper_id).update(
            {
                Paper.parse_status: "done",
                Paper.parsed_at: func.now(),
            }
        )
        _set_paper_status(db, analysis_id, "done", completed=True)
        db.commit()

        return WorkflowResult(
            analysis_id=analysis_id,
            paper_id=paper_id,
            parse_status="done",
            n_claims=len(claims),
        )
    except Exception as exc:
        db.rollback()
        # Persist the error against the analysis row so a future
        # GET /analyses/{id}/claims can surface it.
        _record_error(db, analysis_id, exc)
        if paper_id is not None and not isinstance(exc, ClaimExtractionError):
            # Parse errors: keep the paper row visible with parse_status='error'.
            db.query(Paper).filter(Paper.paper_id == paper_id).update(
                {Paper.parse_status: "error", Paper.error: str(exc)[:500]}
            )
            db.commit()
        raise


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _create_analysis(db: Session) -> UUID:
    """Insert a fresh AnalysisState row (status='pending') and return id."""
    state = AnalysisState()  # paper_status defaults to 'pending'
    db.add(state)
    db.flush()  # populate analysis_id without committing the transaction
    return state.analysis_id


def _validate_upload(upload_bytes: bytes, max_size_bytes: int) -> None:
    if len(upload_bytes) > max_size_bytes:
        raise PdfTooLargeError(f"Upload is {len(upload_bytes)} bytes; max is {max_size_bytes}")
    if not upload_bytes.startswith(_PDF_MAGIC):
        raise PdfInvalidError("Upload does not start with the PDF magic bytes")


def _insert_paper_row(
    db: Session,
    *,
    analysis_id: UUID,
    sha256: str,
    filename: str,
    size_bytes: int,
) -> UUID:
    paper = Paper(
        analysis_id=analysis_id,
        sha256=sha256,
        filename=filename,
        size_bytes=size_bytes,
    )
    db.add(paper)
    db.flush()
    return paper.paper_id


def _set_paper_status(
    db: Session, analysis_id: UUID, status: str, *, completed: bool = False
) -> None:
    update: dict[object, object] = {AnalysisState.paper_status: status}
    if completed:
        update[AnalysisState.completed_at] = datetime.now(UTC)
    db.query(AnalysisState).filter(AnalysisState.analysis_id == analysis_id).update(update)  # type: ignore[arg-type]


def _persist_claims(db: Session, analysis_id: UUID, claims: list[Claim]) -> None:
    """Write claims to ``analysis_states.claims`` as JSONB.

    We use ``claim.model_dump(mode='json')`` so the Pydantic model
    serialises to plain dicts that SQLAlchemy hands to psycopg as
    JSONB natively.
    """
    payload = [c.model_dump(mode="json") for c in claims]
    db.query(AnalysisState).filter(AnalysisState.analysis_id == analysis_id).update(
        {AnalysisState.claims: payload}
    )


def _record_error(db: Session, analysis_id: UUID, exc: Exception) -> None:
    """Append a structured error entry to AnalysisState.errors."""
    blob = {
        "at": datetime.now(UTC).isoformat(),
        "where": "paper_workflow",
        "type": exc.__class__.__name__,
        "message": str(exc)[:500],
    }
    # Re-open the transaction if the workflow rolled it back.
    db.query(AnalysisState).filter(AnalysisState.analysis_id == analysis_id).update(
        {AnalysisState.errors: [blob]}  # type: ignore[dict-item]
    )
    if isinstance(exc, PdfNoTextLayerError):
        status = "error"
    elif isinstance(exc, (PdfTooLargeError, PdfInvalidError)):
        status = "error"
    elif isinstance(exc, ClaimExtractionError):
        status = "error"
    else:  # pragma: no cover — defensive
        status = "error"
    db.query(AnalysisState).filter(AnalysisState.analysis_id == analysis_id).update(
        {AnalysisState.paper_status: status}
    )
    db.commit()


__all__ = [
    "WorkflowResult",
    "run_paper_workflow",
]


# Silence the "imported but unused" linter for Path — used in docstrings.
_ = Path
