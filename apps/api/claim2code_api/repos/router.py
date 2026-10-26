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

Phase 4 (Semantic Indexing) adds:

  * ``POST /repositories/{repository_id}/retrieve`` — embed a claim
    string and return the top-``top_k`` matching symbols by cosine
    similarity. The index is built inline by ``POST /repositories``
    so the first retrieval call against a freshly-cloned repo is
    already served from the pgvector index.

Phase 5 (Hybrid Retrieval) extends the same endpoint with:

  * ``mode`` — ``"semantic"`` (Phase 4 behaviour), ``"lexical"``
    (Postgres tsvector + ts_rank_cd), or ``"hybrid"`` (RRF-fused
    default). The default is ``"hybrid"``; Phase 4 callers that
    omit ``mode`` continue to receive semantic-only scores when
    ``mode="semantic"`` is set explicitly.
  * ``weights`` — per-leg RRF weights for hybrid mode; defaults to
    ``{semantic: 0.55, lexical: 0.25, structural: 0.20}`` (the
    pre-declared triple in ``evaluation/README.md:30``).
  * ``breakdown`` — per-leg score breakdown in the response,
    exposed so the UI can show "cosine 0.91, BM25 0.62,
    struct 1.20".

Exception → HTTP mapping:

  * ``RepoInvalidUrlError`` → 422
  * ``RepoTooLargeError`` → 413
  * ``RepoCloneError`` → 500
  * ``EmbeddingError`` → 500 (semantic leg failure)
  * missing row on GET / DELETE → 404
  * missing repository on retrieve → 404
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
from claim2code_api.indexing.embedding_client import get_embedding_client
from claim2code_api.indexing.exceptions import EmbeddingError
from claim2code_api.indexing.lexical import lexical_search
from claim2code_api.indexing.retriever import retrieve_semantic
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
    RetrieveMatch,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveWeights,
    ScoreBreakdown,
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
from claim2code_api.retrieval.hybrid_ranker import HybridWeights, hybrid_ranker

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
            exclude_paths=body.exclude_paths,
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


# ----------------------------------------------------------------------
# POST /repositories/{repository_id}/retrieve  (Phase 4 + Phase 5)
# ----------------------------------------------------------------------


@router.post(
    "/repositories/{repository_id}/retrieve",
    response_model=RetrieveResponse,
    responses={
        404: {"model": RepoErrorResponse, "description": "Repository not found"},
        500: {"model": RepoErrorResponse, "description": "Embedding call failed"},
    },
)
def retrieve_repository(
    repository_id: UUID,
    body: RetrieveRequest,
    db: Session = Depends(get_db),  # noqa: B008
) -> RetrieveResponse:
    """Return the top-``top_k`` symbols matching ``claim``.

    Phase 5 dispatches on ``body.mode``:

      - ``"semantic"`` (Phase 4 behaviour) — embed the claim (1
        OpenAI call) and run the pgvector cosine query against
        ``code_embeddings``. Returns ``((symbol, similarity) * top_k)``
        in similarity-descending order.
      - ``"lexical"`` (Phase 5) — Postgres ``tsvector @@ query`` +
        ``ts_rank_cd`` against ``code_symbols.lex_tsv``.
      - ``"hybrid"`` (Phase 5 default) — RRF-fused over the three
        legs (semantic, lexical, structural). The per-leg
        ``breakdown`` is populated so the UI can show
        "cosine 0.91, BM25 0.62, struct 1.20".

    An empty list is returned when the repository has no
    embeddings (semantic / hybrid mode) or no lex_tsv matches
    (lexical mode). The user can retry by re-cloning or by
    hitting a future re-index endpoint.

    :raises HTTPException: 404 if the repository row is missing;
        500 if the OpenAI embedding call fails irrecoverably.
    """
    repo = db.query(Repository).filter(Repository.repository_id == repository_id).one_or_none()
    if repo is None:
        raise HTTPException(
            status_code=404,
            detail=f"repository {repository_id} not found",
        )

    # Resolve weights once; the same dict is forwarded to the
    # ranker and echoed in the response.
    weights = _resolve_weights(body.weights)

    # ---- Dispatch on mode ----
    if body.mode == "semantic":
        matches, scores, breakdown = _retrieve_semantic_path(
            db=db,
            repository_id=repository_id,
            claim=body.claim,
            top_k=body.top_k,
        )
    elif body.mode == "lexical":
        matches, scores, breakdown = _retrieve_lexical_path(
            db=db,
            repository_id=repository_id,
            claim=body.claim,
            top_k=body.top_k,
        )
    else:  # "hybrid" — Phase 5 default
        matches, scores, breakdown = _retrieve_hybrid_path(
            db=db,
            repository_id=repository_id,
            claim=body.claim,
            top_k=body.top_k,
            weights=weights,
        )

    return RetrieveResponse(
        repository_id=repository_id,
        claim=body.claim,
        symbols=matches,
        scores=scores,
        mode=body.mode,
        weights=weights,
        breakdown=breakdown,
    )


def _resolve_weights(raw: RetrieveWeights | None) -> RetrieveWeights:
    """Echo the requested weights or the defaults."""
    return raw if raw is not None else RetrieveWeights()


def _retrieve_semantic_path(
    *,
    db: Session,
    repository_id: UUID,
    claim: str,
    top_k: int,
) -> tuple[list[RetrieveMatch], list[float], list[ScoreBreakdown]]:
    """Run the semantic-only leg and serialise to HTTP shapes.

    Raises ``HTTPException(500)`` if the embedding call fails.
    """
    try:
        results = retrieve_semantic(
            db=db,
            repository_id=repository_id,
            claim=claim,
            top_k=top_k,
            embedding_client=get_embedding_client(),
        )
    except EmbeddingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    matches: list[RetrieveMatch] = []
    scores: list[float] = []
    for sym, similarity in results:
        matches.append(
            RetrieveMatch(
                symbol_id=sym.symbol_id,
                kind=CodeSymbolKind(sym.kind),
                language=CodeLanguage(sym.language),
                file_path=sym.file_path,
                start_line=sym.start_line,
                start_col=sym.start_col,
                end_line=sym.end_line,
                end_col=sym.end_col,
                name=sym.name,
                signature=sym.signature or "",
                docstring=sym.docstring,
                similarity=similarity,
            )
        )
        scores.append(similarity)
    return matches, scores, []


def _retrieve_lexical_path(
    *,
    db: Session,
    repository_id: UUID,
    claim: str,
    top_k: int,
) -> tuple[list[RetrieveMatch], list[float], list[ScoreBreakdown]]:
    """Run the lexical-only leg and serialise to HTTP shapes.

    The lexical leg doesn't hit OpenAI, so no EmbeddingError
    can surface here; missing rows just yield empty results.
    The scores are unbounded ``ts_rank_cd`` values (typically
    in ``[0, 1]``).
    """
    rows = lexical_search(
        db=db,
        repository_id=repository_id,
        claim=claim,
        top_k=top_k,
    )
    if not rows:
        return [], [], []

    symbol_ids = [sid for sid, _ in rows]
    symbols = (
        db.query(CodeSymbol)
        .filter(CodeSymbol.symbol_id.in_(symbol_ids))
        .all()
    )
    by_id = {s.symbol_id: s for s in symbols}

    matches: list[RetrieveMatch] = []
    scores: list[float] = []
    for sid, score in rows:
        sym = by_id.get(sid)
        if sym is None:
            continue
        matches.append(
            RetrieveMatch(
                symbol_id=sym.symbol_id,
                kind=CodeSymbolKind(sym.kind),
                language=CodeLanguage(sym.language),
                file_path=sym.file_path,
                start_line=sym.start_line,
                start_col=sym.start_col,
                end_line=sym.end_line,
                end_col=sym.end_col,
                name=sym.name,
                signature=sym.signature or "",
                docstring=sym.docstring,
                similarity=score,
            )
        )
        scores.append(score)
    return matches, scores, []


def _retrieve_hybrid_path(
    *,
    db: Session,
    repository_id: UUID,
    claim: str,
    top_k: int,
    weights: RetrieveWeights,
) -> tuple[list[RetrieveMatch], list[float], list[ScoreBreakdown]]:
    """Run the hybrid ranker and serialise to HTTP shapes.

    The hybrid ranker internally calls ``retrieve_semantic`` —
    we still surface ``EmbeddingError`` as 500.
    """
    ranker_weights = HybridWeights(
        semantic=weights.semantic,
        lexical=weights.lexical,
        structural=weights.structural,
    )
    try:
        hybrid_matches = hybrid_ranker(
            db=db,
            repository_id=repository_id,
            claim=claim,
            top_k=top_k,
            weights=ranker_weights,
            embedding_client=get_embedding_client(),
        )
    except EmbeddingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not hybrid_matches:
        return [], [], []

    # Bulk-fetch the CodeSymbol rows for the top-k matches.
    top_ids = [m.symbol_id for m in hybrid_matches]
    symbols = (
        db.query(CodeSymbol)
        .filter(CodeSymbol.symbol_id.in_(top_ids))
        .all()
    )
    by_id = {s.symbol_id: s for s in symbols}

    matches: list[RetrieveMatch] = []
    scores: list[float] = []
    breakdown: list[ScoreBreakdown] = []
    for hm in hybrid_matches:
        sym = by_id.get(hm.symbol_id)
        if sym is None:
            continue
        matches.append(
            RetrieveMatch(
                symbol_id=sym.symbol_id,
                kind=CodeSymbolKind(sym.kind),
                language=CodeLanguage(sym.language),
                file_path=sym.file_path,
                start_line=sym.start_line,
                start_col=sym.start_col,
                end_line=sym.end_line,
                end_col=sym.end_col,
                name=sym.name,
                signature=sym.signature or "",
                docstring=sym.docstring,
                similarity=hm.final_score,
            )
        )
        scores.append(hm.final_score)
        breakdown.append(
            ScoreBreakdown(
                symbol_id=hm.symbol_id,
                rank_semantic=hm.rank_semantic,
                score_semantic=hm.score_semantic,
                rank_lexical=hm.rank_lexical,
                score_lexical=hm.score_lexical,
                rank_structural=hm.rank_structural,
                struct_boost=hm.struct_boost,
            )
        )
    return matches, scores, breakdown


__all__ = [
    "router",
    "create_repository",
    "get_repository",
    "get_repository_symbols",
    "get_repository_configs",
    "get_repository_dependencies",
    "delete_repository",
    "retrieve_repository",
]
