"""HTTP layer for the paper workflow.

Three endpoints ship in Phase 2:

  * ``POST /papers`` (multipart) — upload a PDF, run parse + extract,
    return the ``WorkflowResult`` summary.
  * ``GET /papers/{paper_id}`` — fetch one paper's metadata row
    (filename, sha256, parse status, timing).
  * ``GET /analyses/{analysis_id}/claims`` — fetch the extracted
    claims as JSONB dicts (matches the shape persisted into
    ``analysis_states.claims``).

Exception → HTTP mapping (handled here, not in middleware, so the
response body matches the per-endpoint contract):

  * ``PdfTooLargeError`` → 413
  * ``PdfInvalidError`` → 422
  * ``PdfNoTextLayerError`` → 422
  * ``ClaimExtractionError`` → 500 (body points to GET /analyses/{id})
  * missing row on GET → 404
"""
from __future__ import annotations

import base64
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile  # noqa: B008
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from claim2code_api.app.config import Settings, get_settings
from claim2code_api.app.deps import get_db
from claim2code_api.db.models import AnalysisState, Paper
from claim2code_api.llm.client import LLMClient, get_llm_client
from claim2code_api.papers import storage
from claim2code_api.papers.exceptions import (
    ClaimExtractionError,
    PdfInvalidError,
    PdfNoTextLayerError,
    PdfTooLargeError,
)
from claim2code_api.papers.excerpt import render_excerpt
from claim2code_api.papers.schemas import (
    ClaimsResponse,
    ErrorResponse,
    PaperExcerptResponse,
    PaperMetadataResponse,
    PaperUploadResponse,
)
from claim2code_api.papers.workflow import WorkflowResult, run_paper_workflow

router = APIRouter(tags=["papers"])


# ----------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------


def get_llm() -> LLMClient:
    """FastAPI dep wrapper around :func:`get_llm_client`."""
    return get_llm_client()


# ----------------------------------------------------------------------
# POST /papers
# ----------------------------------------------------------------------


@router.post(
    "/papers",
    response_model=PaperUploadResponse,
    status_code=200,
    responses={
        413: {"model": ErrorResponse, "description": "PDF too large"},
        422: {"model": ErrorResponse, "description": "PDF invalid or image-only"},
        500: {"model": ErrorResponse, "description": "LLM extraction failed"},
    },
)
def upload_paper(
    file: UploadFile = File(..., description="The PDF to parse and extract claims from"),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
    db: Session = Depends(get_db),  # noqa: B008
    llm: LLMClient = Depends(get_llm),  # noqa: B008
) -> PaperUploadResponse:
    """Upload a PDF and run the synchronous parse + extract workflow."""
    upload_bytes = file.file.read()
    if not upload_bytes:
        raise HTTPException(status_code=422, detail="Empty upload")
    max_size_bytes = settings.max_pdf_size_mb * 1024 * 1024

    try:
        result: WorkflowResult = run_paper_workflow(
            db=db,
            llm=llm,
            upload_bytes=upload_bytes,
            filename=file.filename or "uploaded.pdf",
            max_size_bytes=max_size_bytes,
        )
    except PdfTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (PdfInvalidError, PdfNoTextLayerError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ClaimExtractionError as exc:
        # The analysis row already carries the structured error; we
        # surface a 500 with a hint so the client can fetch details.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PaperUploadResponse(
        analysis_id=result.analysis_id,
        paper_id=result.paper_id,
        parse_status=result.parse_status,
        n_claims=result.n_claims,
    )


# ----------------------------------------------------------------------
# GET /papers/{paper_id}
# ----------------------------------------------------------------------


@router.get(
    "/papers/{paper_id}",
    response_model=PaperMetadataResponse,
    responses={404: {"model": ErrorResponse, "description": "Paper not found"}},
)
def get_paper(
    paper_id: UUID,
    db: Session = Depends(get_db),  # noqa: B008
) -> PaperMetadataResponse:
    """Return metadata for one uploaded paper."""
    row = db.query(Paper).filter(Paper.paper_id == paper_id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"paper {paper_id} not found")
    return PaperMetadataResponse(
        paper_id=row.paper_id,
        analysis_id=row.analysis_id,
        filename=row.filename,
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        n_pages=row.n_pages,
        n_chars=row.n_chars,
        parse_status=row.parse_status,
        uploaded_at=row.uploaded_at,
        parsed_at=row.parsed_at,
        error=row.error,
    )


# ----------------------------------------------------------------------
# GET /analyses/{analysis_id}/claims
# ----------------------------------------------------------------------


@router.get(
    "/analyses/{analysis_id}/claims",
    response_model=ClaimsResponse,
    responses={404: {"model": ErrorResponse, "description": "Analysis not found"}},
)
def get_claims(
    analysis_id: UUID,
    db: Session = Depends(get_db),  # noqa: B008
) -> ClaimsResponse:
    """Return the extracted claims for one analysis run.

    The list of claims is persisted as JSONB on
    ``analysis_states.claims``; we hand it back verbatim so the
    frontend renders the same shape the LLM produced (plus the
    ``paper_evidence`` citation, which is computed server-side).
    """
    state = (
        db.query(AnalysisState)
        .filter(AnalysisState.analysis_id == analysis_id)
        .one_or_none()
    )
    if state is None:
        raise HTTPException(
            status_code=404, detail=f"analysis {analysis_id} not found"
        )
    return ClaimsResponse(
        analysis_id=state.analysis_id,
        paper_status=state.paper_status,
        claims=state.claims,
    )


# ----------------------------------------------------------------------
# GET /analyses/{analysis_id}/paper/pdf  and  /paper/excerpt
# ----------------------------------------------------------------------


def _paper_pdf(db: Session, analysis_id: UUID) -> tuple[Paper, Path]:
    row = db.query(Paper).filter(Paper.analysis_id == analysis_id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no paper for analysis {analysis_id}")
    path = storage.path_for(row.sha256)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="the uploaded PDF is no longer stored")
    return row, path


@router.get(
    "/analyses/{analysis_id}/paper/pdf",
    response_class=FileResponse,
    responses={404: {"model": ErrorResponse, "description": "Paper not found"}},
)
def get_paper_pdf(
    analysis_id: UUID,
    db: Session = Depends(get_db),  # noqa: B008
) -> FileResponse:
    """The uploaded PDF, served inline (append ``#page=N`` to jump)."""
    row, path = _paper_pdf(db, analysis_id)
    return FileResponse(
        path, media_type="application/pdf", filename=row.filename,
        content_disposition_type="inline",
    )


@router.get(
    "/analyses/{analysis_id}/paper/excerpt",
    response_model=PaperExcerptResponse,
    responses={404: {"model": ErrorResponse, "description": "Paper not found"}},
)
def get_paper_excerpt(
    analysis_id: UUID,
    page: int = Query(ge=1),
    quote: str = Query(min_length=1, max_length=4000),
    full: bool = False,
    db: Session = Depends(get_db),  # noqa: B008
) -> PaperExcerptResponse:
    """The part of the paper a claim cites, rendered with the quote highlighted."""
    _row, path = _paper_pdf(db, analysis_id)
    excerpt = render_excerpt(path, page=page, quote=quote, full_page=full)
    return PaperExcerptResponse(
        page=excerpt.page,
        n_pages=excerpt.n_pages,
        found=excerpt.found,
        image="data:image/png;base64," + base64.b64encode(excerpt.png).decode("ascii"),
        width=excerpt.width,
        height=excerpt.height,
    )


__all__ = [
    "router", "upload_paper", "get_paper", "get_claims", "get_paper_pdf", "get_paper_excerpt",
]