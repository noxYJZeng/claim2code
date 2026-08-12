"""HTTP layer for the related-work (literature) workflow.

Two endpoints ship in Phase 7 (per phase-7-design.md §"Three live
decisions"):

  * ``POST /analyses/{analysis_id}/related-work`` — runs the full
    pipeline (SemanticScholarClient search → embedding ranker →
    per-paper LLM comparison) for ONE claim. Persists the
    :class:`RelatedWorkEntry` to
    ``AnalysisState.related_work: JSONB``.
  * ``GET /analyses/{analysis_id}/related-work?claim_id=…`` —
    returns the persisted list verbatim. ``claim_id`` query
    parameter is optional; omit to return all entries for the
    analysis.

Exception → HTTP mapping (matches ``verification/router.py`` and
``papers/router.py``):

  * missing analysis row → 404
  * :class:`LiteratureNotFoundError` (unknown ``claim_id``) → 422
  * :class:`LiteratureSearchError` (both S2 and OpenAlex failed) → 500
  * :class:`LiteratureConfigError` (malformed settings) → 500

The router deliberately does NOT raise :class:`LiteratureComparisonError`
— per-paper LLM errors are isolated by the workflow and reported
on ``RelatedWorkRunResponse.errors`` as a list of strings.

Cache-on-repeat-POST (mirrors Phase 6 verification):

  The workflow returns the cached :class:`RelatedWorkEntry` if
  ``claim_id`` already has one AND ``force_refresh=False``. The
  caller sees ``n_papers`` / ``n_with_comparison`` from the
  cache, and ``errors=[]`` (no work was done on this POST).
"""
from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException  # noqa: B008
from sqlalchemy.orm import Session

from claim2code_api.app.deps import get_db
from claim2code_api.db.models import AnalysisState
from claim2code_api.literature.exceptions import (
    LiteratureConfigError,
    LiteratureNotFoundError,
    LiteratureSearchError,
)
from claim2code_api.literature.schemas import (
    RelatedWorkEntry,
    RelatedWorkListResponse,
    RelatedWorkRunRequest,
    RelatedWorkRunResponse,
)
from claim2code_api.literature.workflow import (
    RelatedWorkWorkflowResult,
    run_related_work_workflow,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["related-work"])


# ----------------------------------------------------------------------
# POST /analyses/{analysis_id}/related-work
# ----------------------------------------------------------------------


@router.post(
    "/analyses/{analysis_id}/related-work",
    response_model=RelatedWorkRunResponse,
    status_code=200,
    responses={
        404: {"description": "Analysis not found"},
        422: {"description": "claim_id references unknown claim"},
        500: {"description": "Scholarly API failed (both S2 + OpenAlex)"},
    },
)
def run_related_work(
    analysis_id: UUID,
    body: RelatedWorkRunRequest,
    db: Session = Depends(get_db),  # noqa: B008
) -> RelatedWorkRunResponse:
    """Run the related-work pipeline for one claim.

    Pipeline (SEARCH → RANK → COMPARE → PERSIST):

      1. Search Semantic Scholar (or OpenAlex if S2 fails) for
         ``top_k * 4`` candidate papers.
      2. Embed the claim text + paper abstracts via Phase 4's
         :class:`EmbeddingClient`; run the 4-leg ranker.
      3. Truncate to ``top_k`` papers; LLM-compare each (one
         call per paper, per-paper isolation).
      4. Persist the resulting :class:`RelatedWorkEntry` to
         ``state.related_work`` (JSONB; replaces any existing
         entry for ``claim_id``).

    Per the plan's "no async" stance: synchronous, one paper at
    a time. A single LLM failure on one paper does NOT abort
    the rest — the workflow records the error on
    ``state.errors``, sets that paper's ``comparison=None``, and
    continues.

    :raises HTTPException(404): analysis row missing.
    :raises HTTPException(422): ``body.claim_id`` references an
        unknown claim.
    :raises HTTPException(500): both S2 and OpenAlex failed.
    """
    try:
        result: RelatedWorkWorkflowResult = run_related_work_workflow(
            db=db,
            analysis_id=analysis_id,
            claim_id=body.claim_id,
            top_k=body.top_k,
            force_refresh=body.force_refresh,
        )
    except LiteratureNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (LiteratureSearchError, LiteratureConfigError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info(
        "related-work analysis=%s claim=%s n_papers=%d n_with_comparison=%d errors=%d",
        analysis_id,
        body.claim_id,
        result.n_papers,
        result.n_with_comparison,
        len(result.errors),
    )

    return RelatedWorkRunResponse(
        analysis_id=result.analysis_id,
        entry=result.entry,
        n_papers=result.n_papers,
        n_with_comparison=result.n_with_comparison,
        errors=result.errors,
    )


# ----------------------------------------------------------------------
# GET /analyses/{analysis_id}/related-work
# ----------------------------------------------------------------------


@router.get(
    "/analyses/{analysis_id}/related-work",
    response_model=RelatedWorkListResponse,
    responses={404: {"description": "Analysis not found"}},
)
def get_related_work(
    analysis_id: UUID,
    claim_id: str | None = None,
    db: Session = Depends(get_db),  # noqa: B008
) -> RelatedWorkListResponse:
    """Return persisted related-work entries for one analysis.

    Reads :attr:`AnalysisState.related_work` JSONB verbatim and
    re-validates each entry through :class:`RelatedWorkEntry` so
    schema drift (a future migration that changes the JSONB
    shape) surfaces as a skipped row with a log warning (mirrors
    Phase 6's GET behavior).

    The optional ``claim_id`` query parameter filters to a single
    entry; omit to return all entries.

    Returns an empty list (NOT 404) when the analysis has no
    related-work entries yet — distinguishes "no related work"
    from "wrong analysis id".
    """
    state = (
        db.query(AnalysisState)
        .filter(AnalysisState.analysis_id == analysis_id)
        .one_or_none()
    )
    if state is None:
        raise HTTPException(
            status_code=404, detail=f"analysis {analysis_id} not found",
        )

    raw = state.related_work or []
    entries: list[RelatedWorkEntry] = []
    for raw_entry in raw:
        try:
            entry = RelatedWorkEntry.model_validate(raw_entry)
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning(
                "skip malformed related_work entry analysis=%s err=%s entry=%s",
                analysis_id, exc, raw_entry,
            )
            continue
        if claim_id is not None and entry.claim_id != claim_id:
            continue
        entries.append(entry)
    return RelatedWorkListResponse(
        analysis_id=state.analysis_id,
        entries=entries,
    )


__all__ = ["router", "run_related_work", "get_related_work"]