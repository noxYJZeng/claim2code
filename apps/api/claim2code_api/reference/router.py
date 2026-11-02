"""HTTP layer for reference checks.

  * ``POST /analyses/{analysis_id}/reference-checks`` — run one check
    for one claim (body: :class:`ReferenceCheckRequest`) and persist it.
    403 unless ``REFERENCE_CHECK_ENABLED`` is set, since it executes
    repository code on this machine.
  * ``GET /analyses/{analysis_id}/reference-checks`` — the persisted
    checks, plus whether running new ones is enabled.
"""
from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException  # noqa: B008
from sqlalchemy.orm import Session

from claim2code_api.app.config import get_settings
from claim2code_api.app.deps import get_db
from claim2code_api.db.models import AnalysisState
from claim2code_api.reference.exceptions import ReferenceCheckNotFoundError
from claim2code_api.reference.schemas import (
    ReferenceCheck,
    ReferenceCheckListResponse,
    ReferenceCheckRequest,
)
from claim2code_api.reference.workflow import run_reference_check

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reference-checks"])

DISABLED_DETAIL = (
    "Reference checks run repository code on this machine and are off. "
    "Set REFERENCE_CHECK_ENABLED=true to turn them on."
)


@router.post(
    "/analyses/{analysis_id}/reference-checks",
    response_model=ReferenceCheck,
    responses={
        403: {"description": "Reference checks are disabled"},
        404: {"description": "Analysis, claim or symbol not found"},
    },
)
def create_reference_check(
    analysis_id: UUID,
    body: ReferenceCheckRequest,
    db: Session = Depends(get_db),  # noqa: B008
) -> ReferenceCheck:
    if not get_settings().reference_check_enabled:
        raise HTTPException(status_code=403, detail=DISABLED_DETAIL)
    try:
        check = run_reference_check(
            db=db, analysis_id=analysis_id, claim_id=body.claim_id, symbol_id=body.symbol_id,
        )
    except ReferenceCheckNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    db.commit()
    return check


@router.get(
    "/analyses/{analysis_id}/reference-checks",
    response_model=ReferenceCheckListResponse,
    responses={404: {"description": "Analysis not found"}},
)
def list_reference_checks(
    analysis_id: UUID,
    db: Session = Depends(get_db),  # noqa: B008
) -> ReferenceCheckListResponse:
    state = (
        db.query(AnalysisState)
        .filter(AnalysisState.analysis_id == analysis_id)
        .one_or_none()
    )
    if state is None:
        raise HTTPException(status_code=404, detail=f"analysis {analysis_id} not found")
    checks = []
    for entry in state.reference_checks or []:
        try:
            checks.append(ReferenceCheck.model_validate(entry))
        except Exception as exc:  # noqa: BLE001 — one bad row must not 500 the list
            logger.warning("skip malformed reference check analysis=%s err=%s", analysis_id, exc)
    return ReferenceCheckListResponse(
        analysis_id=state.analysis_id,
        enabled=get_settings().reference_check_enabled,
        checks=checks,
    )


__all__ = ["create_reference_check", "list_reference_checks", "router"]
