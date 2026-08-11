"""HTTP layer for the repository workflow.

Endpoints shipped in Phase 3:

  * ``POST /repositories`` (JSON body ``{url, analysis_id}``) — clone
    + index a public GitHub repo and bind it to an analysis; runs
    synchronously, capped by ``Settings.workflow_timeout_repo``.
  * ``GET /repositories/{repository_id}`` — fetch one repo's metadata
    row (status, n_files, n_symbols, timing).
  * ``GET /repositories/{repository_id}/symbols`` — list the extracted
    symbols (paginated client-side in v1).
  * ``GET /repositories/{repository_id}/configs`` — list parsed config
    files.
  * ``GET /repositories/{repository_id}/dependencies`` — flattened dep
    list.
  * ``DELETE /repositories/{repository_id}`` — remove the clone + the
    DB row. Idempotent — 204 even if the row is missing.

Exception → HTTP mapping:

  * ``RepoInvalidUrlError`` → 422
  * ``RepoTooLargeError`` → 413
  * ``RepoCloneError`` → 500
  * missing row on GET / DELETE → 404

Commit 5 ships only ``POST /repositories``. The GET / DELETE
endpoints land in commit 6.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException  # noqa: B008
from sqlalchemy.orm import Session

from claim2code_api.app.deps import get_db
from claim2code_api.db.models import AnalysisState, Repository
from claim2code_api.repos.exceptions import (
    RepoCloneError,
    RepoInvalidUrlError,
    RepoTooLargeError,
)
from claim2code_api.repos.schemas import (
    RepoCreateRequest,
    RepoCreateResponse,
    RepoErrorResponse,
)
from claim2code_api.repos.workflow import RepoWorkflowResult, run_repo_workflow

logger = logging.getLogger(__name__)

router = APIRouter(tags=["repositories"])


@router.post(
    "/repositories",
    response_model=RepoCreateResponse,
    status_code=200,
    responses={
        404: {"model": RepoErrorResponse, "description": "Analysis not found"},
        413: {"model": RepoErrorResponse, "description": "Repo too large"},
        422: {"model": RepoErrorResponse, "description": "Invalid URL"},
        500: {"model": RepoErrorResponse, "description": "Clone / index failed"},
    },
)
def create_repository(
    body: RepoCreateRequest,
    db: Session = Depends(get_db),  # noqa: B008
) -> RepoCreateResponse:
    """Clone a GitHub repo and bind it to an analysis.

    Runs the full clone + walk + parse + persist pipeline
    synchronously. Capped by ``Settings.workflow_timeout_repo`` at the
    ASGI middleware layer; an exceeded timeout surfaces as 504.
    """
    # Confirm the analysis row exists — otherwise we have no FK to
    # attach the repo to and the user has clearly pasted the wrong id.
    state = (
        db.query(AnalysisState).filter(AnalysisState.analysis_id == body.analysis_id).one_or_none()
    )
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"analysis {body.analysis_id} not found",
        )

    try:
        result: RepoWorkflowResult = run_repo_workflow(
            db=db,
            analysis_id=body.analysis_id,
            url=body.url,
        )
    except RepoInvalidUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RepoTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except RepoCloneError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Fetch the row back so we can return owner / name from the
    # canonical parsed values (the workflow already wrote them).
    row = db.query(Repository).filter(Repository.repository_id == result.repository_id).one()
    return RepoCreateResponse(
        repository_id=row.repository_id,
        analysis_id=row.analysis_id,
        url=row.url,
        owner=row.owner,
        name=row.name,
        n_files=row.n_files or 0,
        n_symbols=row.n_symbols or 0,
        n_configs=result.n_configs,
        n_dependencies=result.n_dependencies,
        index_status=row.index_status,
    )


__all__ = ["router", "create_repository"]
