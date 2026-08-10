"""HTTP layer for the paper workflow.

Two endpoints ship in Phase 2:

  * ``POST /papers`` (multipart) — upload a PDF, run parse + extract,
    return the ``WorkflowResult`` summary.
  * ``GET /analyses/{analysis_id}/claims`` — fetch the extracted
    claims as JSONB dicts (matches the shape persisted into
    ``analysis_states.claims``). Added in commit 5.

Exception → HTTP mapping (handled here, not in middleware, so the
response body matches the per-endpoint contract):

  * ``PdfTooLargeError`` → 413
  * ``PdfInvalidError`` → 422
  * ``PdfNoTextLayerError`` → 422
  * ``ClaimExtractionError`` → 500 (body points to GET /analyses/{id})
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile  # noqa: B008
from sqlalchemy.orm import Session

from claim2code_api.app.config import Settings, get_settings
from claim2code_api.app.deps import get_db
from claim2code_api.llm.client import LLMClient, get_llm_client
from claim2code_api.papers.exceptions import (
    ClaimExtractionError,
    PdfInvalidError,
    PdfNoTextLayerError,
    PdfTooLargeError,
)
from claim2code_api.papers.schemas import (
    ErrorResponse,
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


__all__ = ["router", "upload_paper"]