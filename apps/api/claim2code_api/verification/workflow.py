"""Verification workflow — orchestration of the per-claim pipeline.

The pipeline for one claim:

  1. **Retrieve candidates** via Phase 5's ``hybrid_ranker``;
     convert each ``HybridMatch`` to a ``RetrieveMatch`` (the
     shape the deterministic checkers + LLM consume) by
     looking up the underlying ``CodeSymbol`` row.
  2. **Deterministic checks** (commits 3 + 4) — hyperparameter
     + 3 name normalizers.
  3. **Skip LLM if decisive** — when the deterministic
     checkers produced a HIGH-severity discrepancy, the
     verdict is already CONTRADICTED; skip the OpenAI call
     to save the round-trip + cost.
  4. **LLM verdict** (commit 5) — reuses the same
     ``LLMClient`` as Phase 2's ``extract_claims``.
  5. **Compose** the final ``Verification`` via
     :func:`claim2code_api.verification.verifier.compose_verification`.
  6. **Persist** to ``AnalysisState.verifications`` (JSONB).

The workflow is feature-local (does NOT use the empty
``claim2code_api/workflow/`` package) — mirroring the
``repos/workflow.py`` and ``papers/workflow.py`` convention.

Error handling
--------------

The Phase 6 plan (§"What we explicitly do NOT do") defers
async verification; one claim at a time, sequentially. A
single LLM failure does NOT abort the whole run — the
problematic claim gets an UNVERIFIED status with an empty
code_evidence list (and the workflow records the error on
``AnalysisState.errors``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from claim2code_api.db.models import AnalysisState, CodeSymbol, Repository
from claim2code_api.indexing.embedding_client import get_embedding_client
from claim2code_api.llm.client import get_llm_client
from claim2code_api.papers.schemas import Claim
from claim2code_api.repos.schemas import CodeLanguage, CodeSymbolKind, RetrieveMatch
from claim2code_api.retrieval.hybrid_ranker import HybridWeights, hybrid_ranker
from claim2code_api.verification.exceptions import VerificationNotFoundError
from claim2code_api.verification.schemas import (
    Verification,
    VerificationStatus,
)
from claim2code_api.verification.verifier import compose_verification

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationWorkflowResult:
    """Summary returned by :func:`run_verification_workflow`.

    ``verifications`` is the full list (sorted by ``claim_id``
    for determinism — same ordering on repeated runs). The
    four ``n_*`` counters mirror the response shape so the
    router can build :class:`VerificationRunResponse` without
    re-iterating the list. ``errors`` is a list of human-readable
    per-claim error strings (also appended to
    ``AnalysisState.errors``).
    """

    analysis_id: UUID
    verifications: list[Verification]
    n_verified: int = 0
    n_partially_verified: int = 0
    n_contradicted: int = 0
    n_unverified: int = 0
    errors: list[str] = field(default_factory=list)


def run_verification_workflow(
    *,
    db: Session,
    analysis_id: UUID,
    claim_ids: list[str] | None = None,
    top_k: int = 5,
) -> VerificationWorkflowResult:
    """Run the verify pipeline for one analysis.

    :param db: SQLAlchemy session (caller owns the lifecycle).
    :param analysis_id: the analysis to verify.
    :param claim_ids: optional subset of claim ids; ``None``
        means verify all extracted claims for the analysis.
    :param top_k: how many candidates to retrieve per claim
        (passed to Phase 5's ``hybrid_ranker``).

    :raises HTTPException(404): the analysis id is unknown.
    :raises VerificationNotFoundError: ``claim_ids`` references
        an unknown claim id.

    Persists the final verifications list to
    ``AnalysisState.verifications`` (JSONB) so the GET endpoint
    can serve them.
    """
    # Lazy import to keep workflow.py import-light.
    from fastapi import HTTPException

    state = (
        db.query(AnalysisState)
        .filter(AnalysisState.analysis_id == analysis_id)
        .one_or_none()
    )
    if state is None:
        raise HTTPException(
            status_code=404, detail=f"analysis {analysis_id} not found",
        )

    claims_dicts = _load_claims(state=state, claim_ids=claim_ids)
    repository_id = _repository_id_for(db, analysis_id)

    verifications: list[Verification] = []
    errors: list[str] = []

    for claim_dict in claims_dicts:
        claim = Claim.model_validate(claim_dict)
        try:
            verification, llm_error = _verify_one_claim(
                db=db,
                analysis_id=analysis_id,
                claim=claim,
                repository_id=repository_id,
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001 — per-claim isolation
            logger.exception(
                "verification failed claim=%s: %s", claim.claim_id, exc,
            )
            errors.append(f"{claim.claim_id}: {exc.__class__.__name__}: {exc}")
            verification = _error_verification(claim, exc)
        else:
            # LLM failure is isolated (the claim still gets a
            # UNVERIFIED verification), but we surface the error
            # on the workflow result so callers can see what went
            # wrong per claim.
            if llm_error is not None:
                errors.append(f"{claim.claim_id}: {llm_error}")
        verifications.append(verification)

    verifications.sort(key=lambda v: v.claim_id)

    # Persist to JSONB. The AnalysisState.verifications column was
    # pre-staged in Phase 1's 0001_init migration; Phase 6 only
    # writes to it.
    state.verifications = [v.model_dump(mode="json") for v in verifications]
    state.errors = (state.errors or []) + [
        {"stage": "verify", "message": e} for e in errors
    ]
    db.flush()

    # Tally status counts.
    n_verified = sum(1 for v in verifications if v.status == VerificationStatus.VERIFIED)
    n_partial = sum(
        1 for v in verifications if v.status == VerificationStatus.PARTIALLY_VERIFIED
    )
    n_contradicted = sum(
        1 for v in verifications if v.status == VerificationStatus.CONTRADICTED
    )
    n_unverified = sum(
        1 for v in verifications if v.status == VerificationStatus.UNVERIFIED
    )

    return VerificationWorkflowResult(
        analysis_id=analysis_id,
        verifications=verifications,
        n_verified=n_verified,
        n_partially_verified=n_partial,
        n_contradicted=n_contradicted,
        n_unverified=n_unverified,
        errors=errors,
    )


# ----------------------------------------------------------------------
# Per-claim pipeline
# ----------------------------------------------------------------------


def _verify_one_claim(
    *,
    db: Session,
    analysis_id: UUID,
    claim: Claim,
    repository_id: UUID | None,
    top_k: int,
) -> tuple[Verification, str | None]:
    """Run the per-claim pipeline.

    Pipeline (mirrors phase-6-design.md §2.6):

      1. Retrieve candidates via hybrid_ranker (Phase 5).
         Convert each HybridMatch → RetrieveMatch by looking
         up the underlying CodeSymbol row.

      2. Deterministic checkers produce discrepancies.

      3. Skip the LLM call if any HIGH-severity discrepancy
         was produced (the verdict is already CONTRADICTED).

      4. Otherwise call LLMClient.verdict_for_claim.

      5. Compose the final Verification.

    :returns: ``(verification, llm_error_or_none)``. The
        ``llm_error`` string is ``None`` unless the LLM call
        raised — the verification is still produced (UNVERIFIED
        by the structural guard), but the workflow records the
        error on its result so callers see what failed.
    """
    # 1. Retrieve candidates + convert to RetrieveMatch.
    retrieve_matches: list[RetrieveMatch] = []
    if repository_id is not None:
        try:
            hybrid_matches = hybrid_ranker(
                db=db,
                repository_id=repository_id,
                claim=claim.text,
                top_k=top_k,
                weights=HybridWeights(),
                embedding_client=get_embedding_client(),
            )
        except Exception:
            hybrid_matches = []
        retrieve_matches = _hybrid_to_retrieve_matches(db=db, matches=hybrid_matches)

    # 2 + 3. Deterministic checkers; skip LLM on HIGH-severity.
    from claim2code_api.verification.verifier import run_deterministic_checkers
    deterministic_discrepancies = run_deterministic_checkers(
        claim=claim, candidates=retrieve_matches,
    )
    has_high = any(d.severity == "HIGH" for d in deterministic_discrepancies)

    # 4. LLM verdict (or None if HIGH-severity already decided).
    llm_verdict: Verification | None = None
    llm_error: str | None = None
    if not has_high:
        # Phase 8: arm the safety violation callback so any secrets /
        # injection patterns detected in the claim / candidate
        # signatures are logged on ``state.errors``. The callback is
        # scoped to this single LLM call (cleared in the finally).
        from claim2code_api.llm.safety_bridge import make_violation_callback
        llm_client = get_llm_client()
        llm_client._on_violations = make_violation_callback(
            db, analysis_id=analysis_id,
        )
        try:
            llm_verdict = llm_client.verdict_for_claim(
                claim=claim,
                paper_evidence=claim.paper_evidence,
                candidates=retrieve_matches,
                deterministic_discrepancies=deterministic_discrepancies,
            )
        except Exception as exc:  # noqa: BLE001 — per-claim isolation
            logger.warning(
                "LLM verdict failed for claim=%s: %s", claim.claim_id, exc,
            )
            # Don't abort the workflow; the composition step will
            # treat llm_verdict=None and default to UNVERIFIED if
            # there are no code evidence or HIGH discrepancies.
            llm_error = f"{exc.__class__.__name__}: {exc}"
        finally:
            llm_client._on_violations = None

    # 5. Compose.
    verification = compose_verification(
        claim=claim,
        candidates=retrieve_matches,
        deterministic_discrepancies=deterministic_discrepancies,
        llm_verdict=llm_verdict,
    )
    return verification, llm_error


def _hybrid_to_retrieve_matches(
    *,
    db: Session,
    matches: list,
) -> list[RetrieveMatch]:
    """Convert Phase 5 ``HybridMatch`` objects to ``RetrieveMatch``.

    Looks up the underlying ``CodeSymbol`` row for each
    ``symbol_id`` to fetch the per-row fields (kind, language,
    file_path, line range, name, signature, docstring) that
    the deterministic checkers and the LLM need.

    Orphan matches (whose ``CodeSymbol`` row vanished between
    retrieval and lookup) are silently skipped — they're not
    actionable evidence.
    """
    if not matches:
        return []
    symbol_ids = [m.symbol_id for m in matches]
    symbols = (
        db.query(CodeSymbol)
        .filter(CodeSymbol.symbol_id.in_(symbol_ids))
        .all()
    )
    by_id = {s.symbol_id: s for s in symbols}
    out: list[RetrieveMatch] = []
    for m in matches:
        sym = by_id.get(m.symbol_id)
        if sym is None:
            continue
        out.append(
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
                similarity=m.final_score,
            )
        )
    return out


def _error_verification(claim: Claim, exc: Exception) -> Verification:
    """Build a UNVERIFIED ``Verification`` for a per-claim failure."""
    return Verification(
        claim_id=claim.claim_id,
        status=VerificationStatus.UNVERIFIED,
        confidence=0.0,
        explanation=f"Workflow error: {exc.__class__.__name__}: {exc}",
        paper_evidence=claim.paper_evidence,
        code_evidence=[],
        discrepancies=[],
        hyperparameters=[],
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _load_claims(
    *,
    state: AnalysisState,
    claim_ids: list[str] | None,
) -> list[dict]:
    """Return the claim dicts to verify, validating ``claim_ids`` if given."""
    if claim_ids is None:
        return list(state.claims or [])
    known = {c.get("claim_id") for c in (state.claims or [])}
    for cid in claim_ids:
        if cid not in known:
            raise VerificationNotFoundError(
                f"claim_id {cid!r} not found in analysis {state.analysis_id}",
            )
    wanted = set(claim_ids)
    return [c for c in (state.claims or []) if c.get("claim_id") in wanted]


def _repository_id_for(db: Session, analysis_id: UUID) -> UUID | None:
    """Return the analysis's bound repository id, or None if not bound."""
    repo = (
        db.query(Repository)
        .filter(Repository.analysis_id == analysis_id)
        .one_or_none()
    )
    return repo.repository_id if repo else None