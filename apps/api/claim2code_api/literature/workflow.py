"""Related-work workflow — orchestration of the per-claim pipeline.

Pipeline (mirrors ``verification/workflow.py`` shape, but at
smaller scale):

  1. **Load** the AnalysisState; resolve ``claim_id`` to a
     :class:`Claim` from ``state.claims`` (Phase 2 list of dicts).
  2. **Cache hit**: if ``force_refresh=False`` AND an entry for
     this ``claim_id`` already exists in ``state.related_work``,
     return the cached entry (idempotent POST semantics; same as
     Phase 6's verification cache).
  3. **Search**: ``SemanticScholarClient.search(query=claim.text)``;
     on ``LiteratureSearchError`` fall back to ``OpenAlexClient``.
     If both fail, persist an empty entry + record error on
     ``state.errors``.
  4. **Rank**: ``rank_papers(papers, sims=…, weights=…)``.
     Embeddings come from Phase 4's ``EmbeddingClient.embed``
     (over-fetch ~20 papers; ranker sorts; workflow truncates to
     ``top_k``).
  5. **Compare**: one ``LLMClient.extract_comparison`` per
     paper. Per-paper isolation: a single LLM failure records
     the error on ``state.errors`` and sets that paper's
     ``comparison`` to ``None`` (the structural guard mirrors
     Phase 6's "no code_evidence → UNVERIFIED").
  6. **Persist**: replace the existing entry (if any) in
     ``state.related_work`` with the new one; flush.

The workflow returns a :class:`RelatedWorkWorkflowResult`
mirroring :class:`VerificationWorkflowResult`. The router uses
it directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from claim2code_api.db.models import AnalysisState
from claim2code_api.indexing.embedding_client import EmbeddingClient, get_embedding_client
from claim2code_api.literature.exceptions import (
    LiteratureConfigError,
    LiteratureNotFoundError,
    LiteratureSearchError,
)
from claim2code_api.literature.ranker import DEFAULT_WEIGHTS, RankingWeights, rank_papers
from claim2code_api.literature.schemas import (
    RankedPaper,
    RelatedWorkEntry,
    ScholarPaper,
    ScholarSource,
)
from claim2code_api.literature.scholar_client import (
    OpenAlexClient,
    ScholarClient,
    SemanticScholarClient,
)
from claim2code_api.llm.client import LLMClient, get_llm_client
from claim2code_api.papers.schemas import Claim

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelatedWorkWorkflowResult:
    """Summary returned by :func:`run_related_work_workflow`.

    Mirrors :class:`VerificationWorkflowResult` shape. ``entry``
    is the new or updated :class:`RelatedWorkEntry`. The four
    counters let the router build
    :class:`RelatedWorkRunResponse` without re-iterating.
    """

    analysis_id: UUID
    entry: RelatedWorkEntry
    n_papers: int = 0
    n_with_comparison: int = 0
    errors: list[str] = field(default_factory=list)


def run_related_work_workflow(
    *,
    db: Session,
    analysis_id: UUID,
    claim_id: str,
    top_k: int = 5,
    force_refresh: bool = False,
    scholar_client: ScholarClient | None = None,
    fallback_client: ScholarClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    llm_client: LLMClient | None = None,
    weights: RankingWeights = DEFAULT_WEIGHTS,
    over_fetch_multiplier: int = 4,
) -> RelatedWorkWorkflowResult:
    """Run the related-work pipeline for one claim.

    :param db: SQLAlchemy session (caller owns the lifecycle).
    :param analysis_id: the analysis to query.
    :param claim_id: which claim from ``state.claims`` to expand.
    :param top_k: how many papers to keep (1–20; per request shape).
    :param force_refresh: if ``True``, bypass the cache and re-run
        even if an entry for ``claim_id`` exists.
    :param scholar_client: primary search client. Defaults to
        :class:`SemanticScholarClient` (or :class:`OpenAlexClient`
        if no Semantic Scholar key is configured).
    :param fallback_client: secondary search client tried on the
        primary's failure. Defaults to the other of S2 / OpenAlex.
    :param embedding_client: Phase 4 client for the cosine-sim leg
        of the ranker. Defaults to the singleton.
    :param llm_client: Phase 7 LLM client. Defaults to the
        singleton. Per-paper failures are isolated (see §5 above).
    :param weights: ranker weights dataclass.
    :param over_fetch_multiplier: fetch ``top_k * over_fetch_multiplier``
        candidates from the scholar client so the ranker has
        headroom to pick the best ``top_k``. Defaults to 4
        (= fetch 20 for top_k=5).
    :raises HTTPException(404): the analysis id is unknown.
    :raises LiteratureNotFoundError: ``claim_id`` is not in
        ``state.claims`` (router maps to 422).
    :raises LiteratureConfigError: malformed settings (router
        maps to 500).
    """
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

    claim = _resolve_claim(state=state, claim_id=claim_id)

    # Cache hit: return the existing entry if present and not force_refresh.
    if not force_refresh:
        cached = _find_existing_entry(state=state, claim_id=claim_id)
        if cached is not None:
            return RelatedWorkWorkflowResult(
                analysis_id=analysis_id,
                entry=cached,
                n_papers=len(cached.papers),
                n_with_comparison=sum(1 for p in cached.papers if p.comparison is not None),
                errors=[],
            )

    primary, fallback, source_used = _select_clients(
        scholar_client=scholar_client,
        fallback_client=fallback_client,
    )

    papers, errors = _search(
        primary=primary,
        fallback=fallback,
        query=claim.text,
        limit=top_k * over_fetch_multiplier,
    )

    # Even if both clients failed, we still persist an empty entry
    # + record the error so the UI can surface "no related work
    # available" rather than pretend the run never happened.
    if not papers:
        entry = _build_entry(
            claim_id=claim_id,
            ranked=[],
            source=source_used,
            papers=[],
        )
        _persist(
            state,
            entry=entry,
            new_errors=errors,
        )
        return RelatedWorkWorkflowResult(
            analysis_id=analysis_id,
            entry=entry,
            n_papers=0,
            n_with_comparison=0,
            errors=errors,
        )

    embed = embedding_client or get_embedding_client()
    ranked = _rank_papers(
        papers=papers,
        claim_text=claim.text,
        embed=embed,
        weights=weights,
    )

    llm = llm_client or get_llm_client()
    # Phase 8: arm the safety violation callback so any secrets /
    # injection patterns detected in the related paper's metadata
    # are logged on ``state.errors``. The callback is set once for
    # the whole comparison loop; ``_compare_papers`` does not
    # override it (the callback simply records violations on the
    # same analysis row regardless of which paper triggered it).
    from claim2code_api.llm.safety_bridge import make_violation_callback
    llm._on_violations = make_violation_callback(
        db, analysis_id=analysis_id,
    )
    try:
        ranked, llm_errors = _compare_papers(
            ranked=ranked,
            claim=claim,
            llm=llm,
        )
    finally:
        llm._on_violations = None
    errors = errors + llm_errors
    # Tally errors after the LLM step.
    n_with_comparison = sum(1 for p in ranked if p.comparison is not None)
    truncated = ranked[:top_k]

    entry = _build_entry(
        claim_id=claim_id,
        ranked=truncated,
        source=source_used,
        papers=papers,
    )

    _persist(state, entry=entry, new_errors=errors)
    return RelatedWorkWorkflowResult(
        analysis_id=analysis_id,
        entry=entry,
        n_papers=len(truncated),
        n_with_comparison=n_with_comparison,
        errors=errors,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_claim(*, state: AnalysisState, claim_id: str) -> Claim:
    """Find the :class:`Claim` for ``claim_id`` in ``state.claims``.

    :raises LiteratureNotFoundError: ``claim_id`` not in
        ``state.claims``.
    """
    raw_claims = state.claims or []
    for raw in raw_claims:
        try:
            claim = Claim.model_validate(raw)
        except Exception:  # noqa: BLE001 — defensive; skip malformed
            continue
        if claim.claim_id == claim_id:
            return claim
    raise LiteratureNotFoundError(
        f"claim_id {claim_id!r} not found in analysis {state.analysis_id}"
    )


def _find_existing_entry(
    *, state: AnalysisState, claim_id: str,
) -> RelatedWorkEntry | None:
    """Return the existing :class:`RelatedWorkEntry` for ``claim_id``."""
    for raw in state.related_work or []:
        try:
            entry = RelatedWorkEntry.model_validate(raw)
        except Exception:  # noqa: BLE001 — skip malformed
            continue
        if entry.claim_id == claim_id:
            return entry
    return None


def _select_clients(
    *,
    scholar_client: ScholarClient | None,
    fallback_client: ScholarClient | None,
) -> tuple[ScholarClient, ScholarClient | None, ScholarSource]:
    """Resolve primary / fallback clients + the :class:`ScholarSource`.

    Selection rules (in order):

      1. If both are injected, use them verbatim and report the
         primary's type as the source.
      2. If only primary is injected, default the fallback to the
         *other* class (S2 ↔ OpenAlex).
      3. If neither is injected, default to S2 primary + OpenAlex
         fallback.

    The :class:`ScholarSource` is inferred from the primary's
    concrete type — stored on the entry so Phase 10 eval can
    measure S2-vs-OpenAlex quality.
    """
    if scholar_client is not None:
        primary = scholar_client
        fallback = fallback_client
        source = _source_from_client(primary)
        return primary, fallback, source

    # Default: S2 primary, OpenAlex fallback. Imported lazily to
    # avoid pulling Settings at module import time (Settings
    # requires ``OPENAI_API_KEY`` to be defined; tests may want to
    # construct the workflow without that).
    from claim2code_api.app.config import get_settings

    settings = get_settings()
    s2 = SemanticScholarClient(api_key=settings.semantic_scholar_api_key)
    primary = s2
    fallback = OpenAlexClient()
    return primary, fallback, ScholarSource.SEMANTIC_SCHOLAR


def _source_from_client(client: ScholarClient) -> ScholarSource:
    """Best-effort source label from a client instance."""
    if isinstance(client, SemanticScholarClient):
        return ScholarSource.SEMANTIC_SCHOLAR
    if isinstance(client, OpenAlexClient):
        return ScholarSource.OPENALEX
    # Fallback for tests / mocks — return a generic S2 label.
    return ScholarSource.SEMANTIC_SCHOLAR


def _search(
    *,
    primary: ScholarClient,
    fallback: ScholarClient | None,
    query: str,
    limit: int,
) -> tuple[list[ScholarPaper], list[str]]:
    """Search primary; fall back to secondary on :class:`LiteratureSearchError`."""
    errors: list[str] = []
    try:
        return primary.search(query=query, limit=limit), errors
    except (LiteratureSearchError, LiteratureConfigError) as exc:
        errors.append(f"primary_search: {exc.__class__.__name__}: {exc}")

    if fallback is None:
        return [], errors

    try:
        return fallback.search(query=query, limit=limit), errors
    except (LiteratureSearchError, LiteratureConfigError) as exc:
        errors.append(f"fallback_search: {exc.__class__.__name__}: {exc}")

    return [], errors


def _rank_papers(
    *,
    papers: list[ScholarPaper],
    claim_text: str,
    embed: EmbeddingClient,
    weights: RankingWeights,
) -> list[RankedPaper]:
    """Embed (claim, paper abstracts) and run the ranker.

    On embedding failure (extremely rare — Phase 4 already raises
    :class:`EmbeddingError` on bad API key / network), we fall back
    to zero similarities so the ranker still produces a result
    ranked by recency + citation only.
    """
    try:
        query_vec = embed.embed([claim_text])[0]
        doc_vecs = embed.embed([p.abstract for p in papers])
    except Exception as exc:  # noqa: BLE001 — ranker fallback path
        logger.warning(
            "embedding failed during related-work rank; falling back to zero sims: %s",
            exc,
        )
        return rank_papers(
            papers,
            sims=[0.0] * len(papers),
            weights=weights,
        )

    sims = [_cosine(query_vec, doc_vec) for doc_vec in doc_vecs]
    return rank_papers(papers, sims=sims, weights=weights)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in ``[-1, 1]`` clamped to ``[0, 1]``.

    Vectors from ``text-embedding-3-small`` are unit-length and
    non-negative, so we clamp to ``[0, 1]`` for the ranker (the
    ranker's semantic leg expects ``[0, 1]``).
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return max(0.0, min(1.0, dot))


def _compare_papers(
    *,
    ranked: list[RankedPaper],
    claim: Claim,
    llm: LLMClient,
) -> tuple[list[RankedPaper], list[str]]:
    """Run :meth:`LLMClient.extract_comparison` for each paper.

    Per-paper isolation: a single LLM failure records the error
    on the workflow's ``errors`` list (caller passes back to
    ``AnalysisState.errors``) and sets that paper's
    ``comparison`` to ``None``. The other papers continue.

    Returns a tuple of (compared_ranked_papers, error_messages).
    """
    compared: list[RankedPaper] = []
    errors: list[str] = []
    for rp in ranked:
        try:
            comparison = llm.extract_comparison(claim=claim, paper=rp.paper)
        except Exception as exc:  # noqa: BLE001 — per-paper isolation
            logger.warning(
                "comparison failed paper=%s claim=%s: %s",
                rp.paper.paper_id,
                claim.claim_id,
                exc,
            )
            errors.append(
                f"{claim.claim_id}/{rp.paper.paper_id}: "
                f"{exc.__class__.__name__}: {exc}"
            )
            compared.append(rp.model_copy(update={"comparison": None}))
            continue
        compared.append(rp.model_copy(update={"comparison": comparison}))
    return compared, errors


def _build_entry(
    *,
    claim_id: str,
    ranked: list[RankedPaper],
    source: ScholarSource,
    papers: list[ScholarPaper],
) -> RelatedWorkEntry:
    """Build the :class:`RelatedWorkEntry` from ranked papers."""
    return RelatedWorkEntry(
        claim_id=claim_id,
        papers=ranked,
        generated_at=datetime.now(tz=UTC),
        source=source,
    )


def _persist(
    state: AnalysisState,
    *,
    entry: RelatedWorkEntry,
    new_errors: list[str],
) -> None:
    """Write the entry to ``state.related_work`` (replace by claim_id).

    Mirrors Phase 6's verifications write-back pattern:
    ``state.related_work = [entry if e.claim_id != cid else new_entry
    for e in state.related_work]`` — one in-place replace keeps
    ordering stable and avoids duplicate entries on repeated POSTs.
    """
    new_entry_dict = entry.model_dump(mode="json")
    existing = list(state.related_work or [])
    # Drop any existing entry for this claim_id, then append the new one.
    kept = [e for e in existing if e.get("claim_id") != entry.claim_id]
    kept.append(new_entry_dict)
    state.related_work = kept

    if new_errors:
        # Coerce error strings to dicts (state.errors is JSONB list of dicts).
        state.errors = (state.errors or []) + [
            {"stage": "related_work", "message": msg} for msg in new_errors
        ]


__all__ = ["RelatedWorkWorkflowResult", "run_related_work_workflow"]