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
import re
import subprocess
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query  # noqa: B008
from sqlalchemy.orm import Session

from claim2code_api.app.config import get_settings
from claim2code_api.app.deps import get_db
from claim2code_api.db.models import AnalysisState, Repository
from claim2code_api.repos.storage import resolve_repo_file
from claim2code_api.verification.exceptions import (
    VerificationConfigError,
    VerificationLLMError,
    VerificationNotFoundError,
)
from claim2code_api.verification.schemas import (
    CodeExcerpt,
    CodeLine,
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
    # The workflow only flushes; without a commit the session's close
    # rolls the persisted verdicts back and GET returns nothing.
    db.commit()

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


# ----------------------------------------------------------------------
# GET /analyses/{analysis_id}/code
# ----------------------------------------------------------------------

# Longest excerpt returned, context included.
_MAX_EXCERPT_LINES = 400


@router.get(
    "/analyses/{analysis_id}/code",
    response_model=CodeExcerpt,
    responses={404: {"description": "Analysis, repository or file not found"}},
)
def get_code_excerpt(
    analysis_id: UUID,
    file_path: str = Query(min_length=1),
    start_line: int = Query(ge=1),
    end_line: int = Query(ge=1),
    context: int = Query(default=3, ge=0, le=50),
    db: Session = Depends(get_db),  # noqa: B008
) -> CodeExcerpt:
    """Return the code a verdict cites, read from the cloned repository.

    The verdict's own ``quote`` is the LLM's copy of a fragment; this
    is the file itself, so the user can check the citation. Paths are
    resolved inside the clone only (``..`` and absolute paths that
    escape it are 404), and files over ``MAX_FILE_KB`` are refused.
    """
    repo = (
        db.query(Repository)
        .filter(Repository.analysis_id == analysis_id)
        .one_or_none()
    )
    if repo is None:
        raise HTTPException(status_code=404, detail="no repository bound to this analysis")
    root = Path(repo.clone_path).resolve()
    target = resolve_repo_file(root, file_path)
    if target is None:
        # Verdicts stored before citation paths were taken from the
        # candidates may carry a line suffix ("m.py:12-30").
        target = resolve_repo_file(root, re.sub(r":\d+(-\d+)?$", "", file_path))
    if target is None:
        raise HTTPException(status_code=404, detail=f"{file_path} not found in the repository")
    if target.stat().st_size > get_settings().max_file_kb * 1024:
        raise HTTPException(status_code=413, detail=f"{file_path} is too large to show")

    text = target.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(text)
    first = min(start_line, max(total, 1))
    last = min(max(end_line, first), max(total, 1))
    lo = max(1, first - context)
    hi = min(total, last + context, lo + _MAX_EXCERPT_LINES - 1)
    rel = target.relative_to(root).as_posix()
    return CodeExcerpt(
        file_path=rel,
        start_line=first,
        end_line=last,
        total_lines=total,
        lines=[CodeLine(number=n, text=text[n - 1]) for n in range(lo, hi + 1)],
        url=_forge_url(repo, root, rel, first, last),
    )


def _forge_url(repo: Repository, root: Path, rel: str, first: int, last: int) -> str | None:
    """GitHub-style link to the cited lines, pinned to the cloned commit."""
    if "github.com" not in repo.url:
        return None
    try:
        ref = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        ref = repo.default_branch
    base = repo.url.removesuffix(".git").rstrip("/")
    anchor = f"#L{first}" if first == last else f"#L{first}-L{last}"
    return f"{base}/blob/{ref}/{rel}{anchor}"


__all__ = ["router", "run_verification", "get_verifications", "get_code_excerpt"]
