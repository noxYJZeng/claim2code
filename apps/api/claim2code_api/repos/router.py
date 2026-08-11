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
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status  # noqa: B008
from sqlalchemy.orm import Session

from claim2code_api.app.deps import get_db
from claim2code_api.db.models import (
    AnalysisState,
    CodeSymbol,
    Repository,
    RepositoryConfig,
    RepositoryDependency,
)
from claim2code_api.repos import cloner
from claim2code_api.repos.exceptions import (
    RepoCloneError,
    RepoInvalidUrlError,
    RepoTooLargeError,
)
from claim2code_api.repos.schemas import (
    CodeLanguage,
    CodeSymbolKind,
    ConfigFormat,
    ConfigsResponse,
    DependenciesResponse,
    DependencyEcosystem,
    RepoCreateRequest,
    RepoCreateResponse,
    RepoErrorResponse,
    RepoMetadataResponse,
    SymbolsResponse,
)
from claim2code_api.repos.schemas import (
    CodeSymbol as CodeSymbolSchema,
)
from claim2code_api.repos.schemas import (
    RepositoryConfig as RepositoryConfigSchema,
)
from claim2code_api.repos.schemas import (
    RepositoryDependency as RepositoryDependencySchema,
)
from claim2code_api.repos.workflow import RepoWorkflowResult, run_repo_workflow

logger = logging.getLogger(__name__)

router = APIRouter(tags=["repositories"])


# ----------------------------------------------------------------------
# POST /repositories
# ----------------------------------------------------------------------


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


# ----------------------------------------------------------------------
# GET /repositories/{repository_id}
# ----------------------------------------------------------------------


@router.get(
    "/repositories/{repository_id}",
    response_model=RepoMetadataResponse,
    responses={404: {"model": RepoErrorResponse, "description": "Repository not found"}},
)
def get_repository(
    repository_id: UUID,
    db: Session = Depends(get_db),  # noqa: B008
) -> RepoMetadataResponse:
    """Return metadata for one cloned repository.

    The endpoint is read-only — the bytes on disk under
    ``data/repos/<owner>__<repo>/`` are not touched; only the
    metadata row is returned. Phase 4-5 will hit this endpoint to
    show the user the indexing status.
    """
    row = db.query(Repository).filter(Repository.repository_id == repository_id).one_or_none()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"repository {repository_id} not found",
        )
    return RepoMetadataResponse(
        repository_id=row.repository_id,
        analysis_id=row.analysis_id,
        url=row.url,
        owner=row.owner,
        name=row.name,
        default_branch=row.default_branch,
        clone_path=row.clone_path,
        n_files=row.n_files,
        n_symbols=row.n_symbols,
        index_status=row.index_status,
        cloned_at=row.cloned_at,
        indexed_at=row.indexed_at,
        error=row.error,
    )


# ----------------------------------------------------------------------
# GET /repositories/{repository_id}/symbols
# ----------------------------------------------------------------------


@router.get(
    "/repositories/{repository_id}/symbols",
    response_model=SymbolsResponse,
    responses={404: {"model": RepoErrorResponse, "description": "Repository not found"}},
)
def get_repository_symbols(
    repository_id: UUID,
    db: Session = Depends(get_db),  # noqa: B008
) -> SymbolsResponse:
    """Return all extracted code symbols for one repository.

    The list is flat (not grouped by file) in v1; the frontend does
    the grouping. Phase 5+ will add cursor-based pagination once a
    10k-symbol repo makes the payload uncomfortable.
    """
    repo = db.query(Repository).filter(Repository.repository_id == repository_id).one_or_none()
    if repo is None:
        raise HTTPException(
            status_code=404,
            detail=f"repository {repository_id} not found",
        )
    rows = (
        db.query(CodeSymbol)
        .filter(CodeSymbol.repository_id == repository_id)
        .order_by(CodeSymbol.file_path, CodeSymbol.start_line)
        .all()
    )
    symbols = [
        CodeSymbolSchema(
            kind=CodeSymbolKind(r.kind),
            language=CodeLanguage(r.language),
            file_path=r.file_path,
            start_line=r.start_line,
            start_col=r.start_col,
            end_line=r.end_line,
            end_col=r.end_col,
            name=r.name,
            signature=r.signature or "",
            docstring=r.docstring,
        )
        for r in rows
    ]
    return SymbolsResponse(repository_id=repository_id, symbols=symbols)


# ----------------------------------------------------------------------
# GET /repositories/{repository_id}/configs
# ----------------------------------------------------------------------


@router.get(
    "/repositories/{repository_id}/configs",
    response_model=ConfigsResponse,
    responses={404: {"model": RepoErrorResponse, "description": "Repository not found"}},
)
def get_repository_configs(
    repository_id: UUID,
    db: Session = Depends(get_db),  # noqa: B008
) -> ConfigsResponse:
    """Return parsed config files for one repository.

    The ``content_json`` field carries the parsed tree (dict for
    YAML/TOML/INI; whatever JSON returned for JSON). Unparseable
    files still appear here with ``error`` set and an empty
    ``content_json``.
    """
    repo = db.query(Repository).filter(Repository.repository_id == repository_id).one_or_none()
    if repo is None:
        raise HTTPException(
            status_code=404,
            detail=f"repository {repository_id} not found",
        )
    rows = (
        db.query(RepositoryConfig)
        .filter(RepositoryConfig.repository_id == repository_id)
        .order_by(RepositoryConfig.file_path)
        .all()
    )
    configs = [
        RepositoryConfigSchema(
            file_path=r.file_path,
            format=ConfigFormat(r.format),
            size_bytes=r.size_bytes,
            content_json=dict(r.content_json or {}),
            error=r.error,
        )
        for r in rows
    ]
    return ConfigsResponse(repository_id=repository_id, configs=configs)


# ----------------------------------------------------------------------
# GET /repositories/{repository_id}/dependencies
# ----------------------------------------------------------------------


@router.get(
    "/repositories/{repository_id}/dependencies",
    response_model=DependenciesResponse,
    responses={404: {"model": RepoErrorResponse, "description": "Repository not found"}},
)
def get_repository_dependencies(
    repository_id: UUID,
    db: Session = Depends(get_db),  # noqa: B008
) -> DependenciesResponse:
    """Return flattened dependency rows for one repository.

    Each row corresponds to one entry in a lockfile or requirements
    file. ``version_spec`` is kept verbatim (``==2.1.0``,
    ``^1.0.0``, etc.) — Phase 6 normalises for cross-checking
    against paper text.
    """
    repo = db.query(Repository).filter(Repository.repository_id == repository_id).one_or_none()
    if repo is None:
        raise HTTPException(
            status_code=404,
            detail=f"repository {repository_id} not found",
        )
    rows = (
        db.query(RepositoryDependency)
        .filter(RepositoryDependency.repository_id == repository_id)
        .order_by(RepositoryDependency.ecosystem, RepositoryDependency.name)
        .all()
    )
    deps = [
        RepositoryDependencySchema(
            ecosystem=DependencyEcosystem(r.ecosystem),
            name=r.name,
            version_spec=r.version_spec or "",
            source_file=r.source_file,
        )
        for r in rows
    ]
    return DependenciesResponse(repository_id=repository_id, dependencies=deps)


# ----------------------------------------------------------------------
# DELETE /repositories/{repository_id}
# ----------------------------------------------------------------------


@router.delete(
    "/repositories/{repository_id}",
    status_code=204,
    responses={
        204: {"description": "Repository deleted"},
        404: {"model": RepoErrorResponse, "description": "Repository not found"},
    },
)
def delete_repository(
    repository_id: UUID,
    db: Session = Depends(get_db),  # noqa: B008
) -> Response:
    """Remove a repository's DB rows and (best-effort) its clone directory.

    Cascade FKs wipe ``code_symbols`` / ``repository_configs`` /
    ``repository_dependencies`` automatically; the on-disk clone at
    ``data/repos/<owner>__<repo>/`` is removed via
    :func:`cloner.remove_clone`, which is itself idempotent.

    Returns 204 No Content on success. Returns 404 if the row is
    missing — the design doc says "idempotent" but the OpenAPI
    spec also wants 404 explicitly so callers can distinguish a
    bad id from a real delete.
    """
    row = db.query(Repository).filter(Repository.repository_id == repository_id).one_or_none()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"repository {repository_id} not found",
        )
    owner = row.owner
    name = row.name
    clone_path_str = row.clone_path
    db.delete(row)
    db.commit()
    # Best-effort disk cleanup. If this raises (e.g. permissions), the
    # DB row is already gone and the orphan clone will be reaped by
    # the next "data/repos" sweep; we log and move on.
    try:
        from pathlib import Path

        cloner.remove_clone(Path(clone_path_str))
    except OSError as exc:
        logger.warning(
            "DB row deleted but clone cleanup failed for %s/%s at %s: %s",
            owner,
            name,
            clone_path_str,
            exc,
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = [
    "router",
    "create_repository",
    "get_repository",
    "get_repository_symbols",
    "get_repository_configs",
    "get_repository_dependencies",
    "delete_repository",
]
