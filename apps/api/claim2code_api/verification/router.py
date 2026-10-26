"""HTTP layer for the verification workflow.

Two endpoints ship in Phase 6 (per phase-6-design.md §"Three live
decisions"):

  * ``POST /analyses/{analysis_id}/verifications`` — runs the
    full pipeline (hybrid_ranker → deterministic checkers →
    LLM verdict → compose) for one analysis and persists the
    results into ``AnalysisState.verifications`` (JSONB).
    Optional body: :class:`VerificationRunRequest` (defaults
    to ``top_k=8`` and ``claim_ids=None`` → verify all).
  * ``GET /analyses/{analysis_id}/verifications`` — the one
    named in the roadmap (``phase-roadmap.md:126``). Returns
    the persisted list verbatim.

Exception → HTTP mapping (matches the conventions in
``papers/router.py`` and ``repos/router.py``):

  * missing analysis row → 404
  * :class:`VerificationNotFoundError` (unknown ``claim_ids``) → 422
  * :class:`VerificationLLMError` (OpenAI failure) → 500
  * :class:`VerificationConfigError` (alias JSON malformed) → 500

The router deliberately does NOT expose per-claim run / per-claim
get endpoints — the plan defers async verification (one POST
runs the whole batch; the persisted JSONB is the canonical
read path).
"""
from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException  # noqa: B008
from sqlalchemy.orm import Session

from claim2code_api.app.deps import get_db
from claim2code_api.db.models import AnalysisState
from claim2code_api.verification.exceptions import (
    VerificationConfigError,
    VerificationLLMError,
    VerificationNotFoundError,
)
from claim2code_api.verification.schemas import (
    VerificationListResponse,
    VerificationRunRequest,
    VerificationRunResponse,
)
from claim2code_api.verification.workflow import (
    VerificationWorkflowResult,
    run_verification_workflow,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["verification"])


# ----------------------------------------------------------------------
# POST /analyses/{analysis_id}/verifications
# ----------------------------------------------------------------------


@router.post(
    "/analyses/{analysis_id}/verifications",
    response_model=VerificationRunResponse,
    status_code=200,
    responses={
        404: {"description": "Analysis not found"},
        422: {"description": "claim_ids references unknown claim_id"},
        500: {"description": "LLM call failed or alias config invalid"},
    },
)
def run_verification(
    analysis_id: UUID,
    body: VerificationRunRequest | None = None,
    db: Session = Depends(get_db),  # noqa: B008
) -> VerificationRunResponse:
    """Run the verification pipeline for one analysis.

    Iterates the analysis's extracted claims (Phase 2) one at a
    time, calling :func:`run_verification_workflow` (Phase 6).
    Per the plan's "no async" stance: synchronous, one claim at
    a time, sequential. A single LLM failure on one claim does
    NOT abort the whole run — the problematic claim gets an
    ``UNVERIFIED`` status with an empty ``code_evidence`` list
    (per-claim isolation in
    ``verification/workflow.py:128-141``).

    Persists the full :class:`Verification` list to
    ``AnalysisState.verifications`` (JSONB) so the matching
    ``GET`` endpoint can return it.

    :raises HTTPException(404): analysis row missing.
    :raises HTTPException(422): ``body.claim_ids`` references an
        unknown claim id.
    :raises HTTPException(500): LLM call failed, or alias JSON
        file is missing / malformed / empty.
    """
    body = body or VerificationRunRequest()

    try:
        result: VerificationWorkflowResult = run_verification_workflow(
            db=db,
            analysis_id=analysis_id,
            claim_ids=body.claim_ids,
            top_k=body.top_k,
        )
    except VerificationNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except VerificationLLMError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except VerificationConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info(
        "verification analysis=%s n=%d verified=%d partial=%d contradicted=%d unverified=%d errors=%d",
        analysis_id,
        len(result.verifications),
        result.n_verified,
        result.n_partially_verified,
        result.n_contradicted,
        result.n_unverified,
        len(result.errors),
    )

    return VerificationRunResponse(
        analysis_id=result.analysis_id,
        n_verified=result.n_verified,
        n_partially_verified=result.n_partially_verified,
        n_contradicted=result.n_contradicted,
        n_unverified=result.n_unverified,
        verifications=result.verifications,
    )


# ----------------------------------------------------------------------
# GET /analyses/{analysis_id}/verifications
# ----------------------------------------------------------------------


@router.get(
    "/analyses/{analysis_id}/verifications",
    response_model=VerificationListResponse,
    responses={404: {"description": "Analysis not found"}},
)
def get_verifications(
    analysis_id: UUID,
    db: Session = Depends(get_db),  # noqa: B008
) -> VerificationListResponse:
    """Return the persisted verification list for one analysis.

    Reads :attr:`AnalysisState.verifications` JSONB verbatim and
    re-validates each entry through :class:`Verification` so
    schema drift (a future migration that changes the JSONB
    shape) surfaces as 500 instead of a malformed response.

    Returns an empty list (NOT 404) when the analysis has no
    verifications yet — distinguishes "verified nothing" from
    "wrong analysis id".
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

    raw = state.verifications or []
    # Re-validate each entry; tolerate individual entry failures
    # so one bad row doesn't 500 the whole list.
    verifications = []
    for entry in raw:
        try:
            from claim2code_api.verification.schemas import Verification
            verifications.append(Verification.model_validate(entry))
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning(
                "skip malformed verification entry analysis=%s err=%s entry=%s",
                analysis_id, exc, entry,
            )
    return VerificationListResponse(
        analysis_id=state.analysis_id,
        verifications=verifications,
    )


__all__ = ["router", "run_verification", "get_verifications"]