"""Verification workflow — orchestration of the per-claim pipeline.

The pipeline for one claim:

  1. **Retrieve candidates** via Phase 5's ``hybrid_ranker``;
     convert each ``HybridMatch`` to a ``RetrieveMatch`` (the
     shape the deterministic checkers + LLM consume) by
     looking up the underlying ``CodeSymbol`` row.
  2. **Deterministic checks** (commits 3 + 4) — hyperparameter
     + 3 name normalizers.
  3. **Prepare located source** — retain bounded prefixes and
     claim-focused implementation windows with their real lines.
  4. **LLM verdict** — reuses the same
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

Observability
-------------

Phase 9: every run opens a root span ``verification_workflow.run``
(via the package's :func:`get_tracer`); each per-claim iteration
opens a child span ``verification.claim_verify`` with the
``claim.id`` and ``claim.type`` attributes so the trace tree is
drillable per claim in the trace UI.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from claim2code_api.app.config import get_settings
from claim2code_api.db.models import AnalysisState, CodeSymbol, Repository
from claim2code_api.indexing.embedding_client import get_embedding_client
from claim2code_api.indexing.indexer import _read_source_range
from claim2code_api.llm.client import get_llm_client
from claim2code_api.observability import get_tracer
from claim2code_api.papers.schemas import Claim
from claim2code_api.repos.schemas import CodeLanguage, CodeSymbolKind, RetrieveMatch, SourceSpan
from claim2code_api.retrieval.hybrid_ranker import HybridWeights, hybrid_ranker
from claim2code_api.retrieval.reranker import POOL_SIZE as RERANK_POOL_SIZE
from claim2code_api.retrieval.reranker import rerank
from claim2code_api.verification.context import prepare_candidates
from claim2code_api.verification.exceptions import VerificationNotFoundError
from claim2code_api.verification.schemas import (
    Discrepancy,
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
    top_k: int = 8,
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

    # Phase 9: open the root span for this verification run. The
    # per-claim loop below opens child spans so each claim's trace
    # is drillable independently in the trace UI.
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span(
        "verification_workflow.run",
        attributes={
            "verification.analysis_id": str(analysis_id),
            "verification.top_k": top_k,
            "verification.n_claims": len(claims_dicts),
        },
    ) as root_span:
        verifications: list[Verification] = []
        errors: list[str] = []

        for claim_dict in claims_dicts:
            claim = Claim.model_validate(claim_dict)
            # Child span per claim — disambiguates the per-claim
            # children in the trace UI and tags the final status.
            with tracer.start_as_current_span(
                "verification.claim_verify",
                attributes={
                    "claim.id": claim.claim_id,
                    "claim.type": claim.claim_type.value
                    if hasattr(claim.claim_type, "value")
                    else str(claim.claim_type),
                },
            ) as claim_span:
                try:
                    verification, llm_error = _verify_one_claim(
                        db=db,
                        analysis_id=analysis_id,
                        claim=claim,
                        repository_id=repository_id,
                        top_k=top_k,
                    )
                except Exception as exc:  # noqa: BLE001 — per-claim isolation
                    claim_span.record_exception(exc)
                    logger.exception(
                        "verification failed claim=%s: %s", claim.claim_id, exc,
                    )
                    errors.append(
                        f"{claim.claim_id}: {exc.__class__.__name__}: {exc}",
                    )
                    verification = _error_verification(claim, exc)
                else:
                    # LLM failure is isolated (the claim still gets a
                    # UNVERIFIED verification), but we surface the
                    # error on the workflow result so callers can see
                    # what failed per claim.
                    if llm_error is not None:
                        errors.append(f"{claim.claim_id}: {llm_error}")
                claim_span.set_attribute(
                    "verification.status",
                    verification.status.value
                    if hasattr(verification.status, "value")
                    else str(verification.status),
                )
            verifications.append(verification)

        verifications.sort(key=lambda v: v.claim_id)

        # Persist to JSONB. The AnalysisState.verifications column
        # was pre-staged in Phase 1's 0001_init migration; Phase 6
        # only writes to it. A subset run replaces only its own
        # claims' entries; earlier verdicts for other claims stay.
        rerun = {v.claim_id for v in verifications}
        kept = [e for e in (state.verifications or []) if e.get("claim_id") not in rerun]
        state.verifications = sorted(
            kept + [v.model_dump(mode="json") for v in verifications],
            key=lambda e: e["claim_id"],
        )
        state.errors = (state.errors or []) + [
            {"stage": "verify", "message": e} for e in errors
        ]
        db.flush()

        # Tally status counts.
        n_verified = sum(
            1 for v in verifications if v.status == VerificationStatus.VERIFIED
        )
        n_partial = sum(
            1 for v in verifications if v.status == VerificationStatus.PARTIALLY_VERIFIED
        )
        n_contradicted = sum(
            1 for v in verifications if v.status == VerificationStatus.CONTRADICTED
        )
        n_unverified = sum(
            1 for v in verifications if v.status == VerificationStatus.UNVERIFIED
        )

        root_span.set_attribute("verification.n_verified", n_verified)
        root_span.set_attribute("verification.n_partially_verified", n_partial)
        root_span.set_attribute("verification.n_contradicted", n_contradicted)
        root_span.set_attribute("verification.n_unverified", n_unverified)

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

      3. Prepare source windows shared by the LLM and validator.

      4. Call LLMClient.verdict_for_claim to obtain source citations.

      5. Compose the final Verification.

      6. If evidence is insufficient, expand retrieval once and ask
         again only when additional source symbols were found.

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
            hybrid_matches = _retrieve(db, repository_id, claim.text, top_k=top_k)
        except Exception:
            hybrid_matches = []
        retrieve_matches = _hybrid_to_retrieve_matches(
            db=db, matches=hybrid_matches, repository_id=repository_id,
        )

    retrieve_matches = prepare_candidates(retrieve_matches, claim.text)

    # A deterministic mismatch alone does not provide a source citation.
    # Still obtain a verdict: skipping here used to force a decisive
    # discrepancy through compose's no-evidence -> UNVERIFIED guard.
    from claim2code_api.verification.verifier import run_deterministic_checkers
    deterministic_discrepancies = run_deterministic_checkers(
        claim=claim, candidates=retrieve_matches,
    )
    llm_verdict, llm_error = _obtain_verdict(
        db, analysis_id, claim, retrieve_matches, deterministic_discrepancies,
    )

    # 5. Compose.
    verification = compose_verification(
        claim=claim,
        candidates=retrieve_matches,
        deterministic_discrepancies=deterministic_discrepancies,
        llm_verdict=llm_verdict,
    )
    if (
        verification.status == VerificationStatus.UNVERIFIED
        and repository_id is not None and llm_error is None and top_k < 24
    ):
        expanded = _expanded_candidates(
            db=db, repository_id=repository_id, claim=claim,
            previous=retrieve_matches, top_k=top_k,
        )
        if expanded:
            with get_tracer(__name__).start_as_current_span(
                "verification.evidence_retry",
                attributes={"verification.candidate_count": len(expanded)},
            ):
                retry_discrepancies = run_deterministic_checkers(claim=claim, candidates=expanded)
                retry_verdict, retry_error = _obtain_verdict(
                    db, analysis_id, claim, expanded, retry_discrepancies,
                )
                if retry_error is None:
                    verification = compose_verification(
                        claim=claim, candidates=expanded,
                        deterministic_discrepancies=retry_discrepancies,
                        llm_verdict=retry_verdict,
                    )
                else:
                    # Preserve the first assessment when the additional
                    # request fails, while exposing its model error.
                    llm_error = retry_error
    return verification, llm_error


def _retrieve(db: Session, repository_id: UUID, text: str, *, top_k: int) -> list:
    """Hybrid candidates, reranked by the chat model when enabled."""
    if not get_settings().retrieval_rerank:
        return hybrid_ranker(
            db=db, repository_id=repository_id, claim=text, top_k=top_k,
            weights=HybridWeights(), embedding_client=get_embedding_client(),
        )
    pool = hybrid_ranker(
        db=db, repository_id=repository_id, claim=text, top_k=max(top_k, RERANK_POOL_SIZE),
        weights=HybridWeights(), embedding_client=get_embedding_client(),
    )
    return rerank(db, repository_id, text, pool, ranker=get_llm_client())[:top_k]


def _expanded_candidates(
    *, db: Session, repository_id: UUID, claim: Claim,
    previous: list[RetrieveMatch], top_k: int,
) -> list[RetrieveMatch]:
    """One larger retrieval pass, with at most four new source candidates.

    Search uses the original claim; model explanations and benchmark labels
    never become retrieval instructions. An unchanged shortlist is not sent
    to the model again. This pass is used for insufficient evidence, not
    transport failures or repeated sampling until a preferred verdict.
    """
    try:
        matches = _retrieve(db, repository_id, claim.text, top_k=min(max(top_k * 2, 12), 24))
        expanded = _hybrid_to_retrieve_matches(
            db=db, matches=matches, repository_id=repository_id,
        )
    except Exception as exc:  # noqa: BLE001 — optional recovery pass
        logger.warning("Evidence expansion failed for claim=%s: %s", claim.claim_id, exc)
        return []
    known = {c.symbol_id for c in previous}
    additions = [c for c in expanded if c.symbol_id not in known and (c.source or c.context_spans)]
    if not additions:
        return []
    # Existing candidates are already bounded and remain in the same order.
    return prepare_candidates([*previous, *additions[:4]], claim.text)


def _obtain_verdict(
    db: Session, analysis_id: UUID, claim: Claim,
    candidates: list[RetrieveMatch], discrepancies: list[Discrepancy],
) -> tuple[Verification | None, str | None]:
    """Scope safety reporting and isolate a single model request failure."""
    from claim2code_api.llm.safety_bridge import make_violation_callback

    client = get_llm_client()
    client._on_violations = make_violation_callback(db, analysis_id=analysis_id)
    try:
        return client.verdict_for_claim(
            claim=claim, paper_evidence=claim.paper_evidence,
            candidates=candidates, deterministic_discrepancies=discrepancies,
        ), None
    except Exception as exc:  # noqa: BLE001 — per-claim isolation
        logger.warning("LLM verdict failed for claim=%s: %s", claim.claim_id, exc)
        return None, f"{exc.__class__.__name__}: {exc}"
    finally:
        client._on_violations = None


def _hybrid_to_retrieve_matches(
    *,
    db: Session,
    matches: list,
    repository_id: UUID | None = None,
) -> list[RetrieveMatch]:
    """Convert Phase 5 ``HybridMatch`` objects to ``RetrieveMatch``.

    Looks up the underlying ``CodeSymbol`` row for each
    ``symbol_id`` to fetch the per-row fields (kind, language,
    file_path, line range, name, signature, docstring) that
    the deterministic checkers and the LLM need.

    Orphan matches (whose ``CodeSymbol`` row vanished between
    retrieval and lookup) are silently skipped — they're not
    actionable evidence.

    When ``repository_id`` is given, each match also carries its
    source body read from the repo's clone (``""`` → ``None`` if the
    clone is gone), so the verifier sees the implementation.
    """
    if not matches:
        return []
    clone_path: Path | None = None
    if repository_id is not None:
        repo = db.get(Repository, repository_id)
        if repo is not None and repo.clone_path:
            clone_path = Path(repo.clone_path)
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
        source = (
            _read_source_range(
                clone_path=clone_path,
                rel_path=sym.file_path,
                start_line=sym.start_line,
                end_line=sym.end_line,
            )
            or None
            if clone_path is not None
            else None
        )
        class_spans = _class_spans(db, sym, clone_path) if clone_path is not None else []
        callee_spans = (
            _callee_spans(db, sym, source, clone_path, exclude=set(by_id))
            if clone_path is not None and source
            else []
        )
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
                source=source,
                class_context="\n\n".join(
                    f"# {s.name} (lines {s.start_line}-{s.end_line})\n{s.source}" for s in class_spans
                ) or None,
                callee_context="\n\n".join(
                    f"# {s.name} ({s.file_path}:{s.start_line}-{s.end_line})\n{s.source}"
                    for s in callee_spans
                ) or None,
                context_spans=class_spans + callee_spans,
            )
        )
    return out


# Methods that set a class up. A claim about initialisation or
# scaling ("B is initialised to 0", "scaled by alpha/r") is
# implemented here, not in the ``forward`` that retrieval tends to
# surface.
_SETUP_METHOD_NAMES = (
    "__init__",
    "reset_parameters",
    "_reset_parameters",
    "init_weights",
    "_init_weights",
)


def _span(sym: CodeSymbol, source: str) -> SourceSpan:
    return SourceSpan(
        symbol_id=sym.symbol_id, file_path=sym.file_path,
        start_line=sym.start_line, end_line=sym.end_line, name=sym.name, source=source,
    )


def _class_spans(db: Session, sym: CodeSymbol, clone_path: Path) -> list[SourceSpan]:
    """Return the source of ``sym``'s sibling setup methods, if any."""
    if sym.parent_symbol_id is None:
        return []
    siblings = (
        db.query(CodeSymbol)
        .filter(
            CodeSymbol.parent_symbol_id == sym.parent_symbol_id,
            CodeSymbol.name.in_(_SETUP_METHOD_NAMES),
            CodeSymbol.symbol_id != sym.symbol_id,
        )
        .order_by(CodeSymbol.start_line)
        .all()
    )
    parts: list[SourceSpan] = []
    for sib in siblings:
        body = _read_source_range(
            clone_path=clone_path,
            rel_path=sib.file_path,
            start_line=sib.start_line,
            end_line=sib.end_line,
        )
        if body:
            parts.append(_span(sib, body))
    return parts


# A call site: an identifier followed by ``(``. Language-agnostic
# enough for the Python/JS/Go/Rust/Java the parser indexes.
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
# A dispatch through a registry: ``loss_registry[config.model.type](...)``.
_DISPATCH_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\[[^\]\n]*\]\s*\(")
_IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_STRING_RE = re.compile(r"(\"[^\"\n]*\"|'[^'\n]*')")
_MAX_CALLEES = 4


def _callee_spans(
    db: Session,
    sym: CodeSymbol,
    source: str,
    clone_path: Path,
    *,
    exclude: set[UUID],
) -> list[SourceSpan]:
    """Return the source of repo functions/methods that ``sym`` calls.

    The detail a claim names often sits one call away from the
    retrieved symbol (``sample_image`` -> ``generalized_steps``,
    ``alfworld_run`` -> ``check_is_exhausted``). A name defined more
    than once resolves to the definition nearest ``sym``'s file; a tie
    is skipped as ambiguous. Symbols already among the candidates
    (``exclude``) are not repeated. A call through a module-level
    registry (``loss_registry[key](...)``) resolves to the functions
    the registry's literal names, listed before the direct callees.
    """
    body = source.split("\n", 1)[1] if "\n" in source else ""
    # Registry targets go first: a direct callee's name is at least
    # visible in ``source``, a dispatched one is not.
    registries = _DISPATCH_RE.findall(body)
    calls = _registry_members(db, sym, registries, clone_path)
    calls += _CALL_RE.findall(body)
    names = list(dict.fromkeys(n for n in calls if n != sym.name))
    if not names:
        return []
    defs = (
        db.query(CodeSymbol)
        .filter(
            CodeSymbol.repository_id == sym.repository_id,
            CodeSymbol.kind.in_(("function", "method")),
            CodeSymbol.name.in_(names),
            CodeSymbol.symbol_id != sym.symbol_id,
        )
        .all()
    )
    by_name: dict[str, list[CodeSymbol]] = {}
    for d in defs:
        by_name.setdefault(d.name, []).append(d)
    parts: list[SourceSpan] = []
    for name in names:
        callee = _nearest_definition(by_name.get(name, []), sym.file_path)
        if callee is None or callee.symbol_id in exclude:
            continue
        text = _read_source_range(
            clone_path=clone_path,
            rel_path=callee.file_path,
            start_line=callee.start_line,
            end_line=callee.end_line,
        )
        if text:
            parts.append(_span(callee, text))
        if len(parts) == _MAX_CALLEES:
            break
    # Show the dispatch table itself as well as its targets. Otherwise
    # the model can see a helper's body but cannot establish that a
    # selected configuration key actually calls it.
    parts.extend(_registry_spans(db, sym, registries, clone_path))
    return parts


def _registry_spans(
    db: Session, sym: CodeSymbol, registries: list[str], clone_path: Path,
) -> list[SourceSpan]:
    if not registries:
        return []
    consts = (
        db.query(CodeSymbol)
        .filter(
            CodeSymbol.repository_id == sym.repository_id,
            CodeSymbol.kind == "constant",
            CodeSymbol.name.in_(registries),
        )
        .all()
    )
    spans: list[SourceSpan] = []
    for name in dict.fromkeys(registries):
        const = _nearest_definition([c for c in consts if c.name == name], sym.file_path)
        if const is None:
            continue
        text = _read_source_range(
            clone_path=clone_path, rel_path=const.file_path,
            start_line=const.start_line, end_line=const.end_line,
        )
        if text:
            spans.append(_span(const, text))
    return spans


def _registry_members(
    db: Session,
    sym: CodeSymbol,
    registries: list[str],
    clone_path: Path,
) -> list[str]:
    """Identifiers named in the literal of each dispatched-through constant."""
    if not registries:
        return []
    consts = (
        db.query(CodeSymbol)
        .filter(
            CodeSymbol.repository_id == sym.repository_id,
            CodeSymbol.kind == "constant",
            CodeSymbol.name.in_(registries),
        )
        .all()
    )
    by_name: dict[str, list[CodeSymbol]] = {}
    for c in consts:
        by_name.setdefault(c.name, []).append(c)
    members: list[str] = []
    for name in dict.fromkeys(registries):
        const = _nearest_definition(by_name.get(name, []), sym.file_path)
        if const is None:
            continue
        text = _read_source_range(
            clone_path=clone_path,
            rel_path=const.file_path,
            start_line=const.start_line,
            end_line=const.end_line,
        )
        literal = text.split("=", 1)[1] if text and "=" in text else ""
        members += _IDENT_RE.findall(_STRING_RE.sub("", literal))
    return members


def _nearest_definition(defs: list[CodeSymbol], file_path: str) -> CodeSymbol | None:
    """Pick the definition sharing the longest path prefix with ``file_path``."""
    if not defs:
        return None
    target = file_path.split("/")

    def shared(d: CodeSymbol) -> int:
        n = 0
        for a, b in zip(d.file_path.split("/"), target, strict=False):
            if a != b:
                break
            n += 1
        return n

    ranked = sorted(defs, key=shared, reverse=True)
    if len(ranked) > 1 and shared(ranked[0]) == shared(ranked[1]):
        return None
    return ranked[0]


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
