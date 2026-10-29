"""Fuse semantic, lexical and identifier retrieval with repository structure.

Each leg contributes a bounded ranked candidate set. Semantic similarity
finds paraphrases; PostgreSQL full-text search finds prose in docstrings
and signatures; identifier search splits snake_case, CamelCase and acronym
names so undocumented implementations also enter the pool. Reciprocal
Rank Fusion uses default weights 0.40 / 0.40 / 0.20 and k=60, followed by
import/path relevance multipliers.

Declared vendored dependencies are searched both with and without their
paths. First-party code can enter even when a copied library fills the
unfiltered pool; copied code remains eligible because authors may adapt it.
The final cut retains each active leg's best implementation when space
allows. All signals come from the query and repository, never annotations.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, replace
from uuid import UUID

from sqlalchemy import or_
from sqlalchemy.orm import Session

from claim2code_api.db.models import CodeSymbol, RepositoryDependency
from claim2code_api.indexing.embedding_client import EmbeddingClient
from claim2code_api.indexing.lexical import identifier_search, lexical_search
from claim2code_api.indexing.retriever import retrieve_semantic
from claim2code_api.indexing.structural import structural_boost
from claim2code_api.observability import get_tracer
from claim2code_api.observability.meter import retriever_request_duration_seconds

logger = logging.getLogger(__name__)

# RRF constant. Cormack et al. 2009; default in ElasticSearch,
# OpenSearch, Vespa. Lower k → rank differences amplified;
# higher k → RRF approaches simple weighted sum.
_RRF_K = 60

# Score multiplier for symbols under a directory named after one of
# the repo's own dependencies (``examples/NLU/src/transformers`` in
# microsoft/LoRA): a vendored copy of a library, whose thousands of
# symbols otherwise drown the paper's code. Down-weighted rather than
# dropped, since authors do patch vendored files. When the vendored
# package sits in a sub-project with its own build manifest
# (``examples/NLU/setup.py``), the whole sub-project — the library's
# tests, examples and tooling too — is the vendored copy.
_VENDORED_FACTOR = 0.5
_MANIFESTS = ("setup.py", "setup.cfg", "pyproject.toml")


@dataclass(frozen=True)
class HybridWeights:
    """Per-leg RRF weights (see the module docstring for the defaults).

    Sum is not required to be 1 — the normaliser uses the
    *theoretical max* (all legs rank 1) which only equals 1
    if the weights sum to 1. If the caller passes non-default
    weights that don't sum to 1 the normalised score's range
    is documented as ``[0, sum(weights)]`` instead of
    ``[0, 1]``. Tests assert the default case.
    """

    semantic: float = 0.40
    lexical: float = 0.40
    structural: float = 0.20

    def as_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "lexical": self.lexical,
            "structural": self.structural,
        }


@dataclass(frozen=True)
class HybridMatch:
    """One hybrid-ranked symbol with full per-leg breakdown.

    ``rank_*`` is 1-indexed (rank 1 = best in that leg).
    ``None`` means the symbol wasn't in that leg's result.
    ``score_*`` mirrors the leg's raw score (``ts_rank_cd``
    for lexical, cosine similarity for semantic, etc.).
    ``struct_boost`` is the multiplier (default ``1.0``).
    ``final_score`` is the RRF score times the boost,
    normalised so the theoretical max (``all legs rank 1``)
    maps to ``sum(weights)``.
    """

    symbol_id: UUID
    final_score: float
    rank_semantic: int | None = None
    score_semantic: float | None = None
    rank_lexical: int | None = None
    score_lexical: float | None = None
    rank_structural: int | None = None
    struct_boost: float = 1.0


def hybrid_ranker(
    db: Session,
    repository_id: UUID,
    claim: str,
    *,
    top_k: int = 5,
    weights: HybridWeights | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> list[HybridMatch]:
    """Return top-``top_k`` semantic / lexical / identifier-fused symbols.

    :param db: a SQLAlchemy session (caller owns the lifecycle).
    :param repository_id: scope the search to this repository.
    :param claim: the natural-language claim string.
    :param top_k: number of results to return (default 5).
    :param weights: per-leg RRF weights; defaults to ``HybridWeights()``.
    :param embedding_client: optional override for the embedding
        client. Defaults to ``get_embedding_client()``; tests pass
        a stub.
    :returns: a list of ``HybridMatch`` ordered by ``final_score``
        descending. Empty claim → ``[]``.
    :raises EmbeddingError: the semantic leg's OpenAI call
        failed irrecoverably. (Lexical / structural failures
        don't raise — they degrade gracefully to no contribution.)
    """
    if not claim or not claim.strip():
        return []

    if weights is None:
        weights = HybridWeights()

    # Each leg returns up to ``top_k * 3`` candidates so the
    # union has enough variety. We pass top_k * 3 to the legs
    # but cap at 50 to mirror the Phase 4 endpoint cap.
    leg_top_k = min(max(top_k * 3, 15), 50)

    # Phase 9: open the orchestrator span up-front. The 3 legs
    # each open a child span with the ``retrieval.leg`` attribute
    # so per-leg latency is attributable in the trace UI.
    # ``retriever_request_duration_seconds`` records the same wall
    # time as a histogram observation per leg.
    tracer = get_tracer(__name__)
    retriever_duration = retriever_request_duration_seconds()
    with tracer.start_as_current_span(
        "retrieval.hybrid_ranker",
        attributes={
            "retrieval.top_k": top_k,
            "retrieval.leg_top_k": leg_top_k,
            "retrieval.weights": str(weights.as_dict()),
        },
    ) as root_span:
        # A vendored library can fill both legs' top ``leg_top_k`` on
        # its own (microsoft/LoRA: ~23k transformers symbols against
        # ~40 of loralib), so each leg also runs over the repo's own
        # code, and a symbol keeps the better of its two ranks.
        vendored = _vendored_pattern(db, repository_id)
        exclusions = [None] if vendored is None else [None, vendored]

        # ---- Leg 1: semantic ----
        leg_start = time.monotonic()
        with tracer.start_as_current_span(
            "retrieval.semantic",
            attributes={"retrieval.leg": "semantic"},
        ):
            sem_passes = [
                retrieve_semantic(
                    db,
                    repository_id,
                    claim,
                    top_k=leg_top_k,
                    embedding_client=embedding_client,
                    exclude_path_pattern=exclude,
                )
                for exclude in exclusions
            ]
        retriever_duration.record(
            time.monotonic() - leg_start, {"leg": "semantic"},
        )

        # ---- Leg 2: lexical ----
        leg_start = time.monotonic()
        with tracer.start_as_current_span(
            "retrieval.lexical",
            attributes={"retrieval.leg": "lexical"},
        ):
            lex_passes = [
                lexical_search(
                    db,
                    repository_id,
                    claim,
                    top_k=leg_top_k,
                    exclude_path_pattern=exclude,
                )
                for exclude in exclusions
            ]
        retriever_duration.record(
            time.monotonic() - leg_start, {"leg": "lexical"},
        )

        # Undocumented snake_case / CamelCase implementations may miss
        # both the embedding and raw-name tsvector legs. A bounded
        # identifier/path pass expands the pool independently of those
        # rankings, over both the full repo and first-party code.
        identifier_passes = [
            identifier_search(
                db, repository_id, claim, top_k=leg_top_k,
                exclude_path_pattern=exclude,
            )
            for exclude in exclusions
        ] if weights.structural > 0 else []

        # ---- Build rank + score maps per leg ----
        # Ranks are 1-indexed; not-in-leg → omit from sum.
        sem_ranks = _best_ranks(
            [[(sym.symbol_id, float(sim)) for sym, sim in results] for results in sem_passes]
        )
        lex_ranks = _best_ranks(
            [[(sid, float(score)) for sid, score in results] for results in lex_passes]
        )
        identifier_ranks = _best_ranks(identifier_passes)

        # Identifier matching independently expands the semantic / lexical pool.
        candidate_ids = list({*sem_ranks.keys(), *lex_ranks.keys(), *identifier_ranks.keys()})
        if not candidate_ids:
            root_span.set_attribute("retrieval.result_count", 0)
            return []

        # ---- Leg 3: structural (multiplier, not rank) ----
        leg_start = time.monotonic()
        with tracer.start_as_current_span(
            "retrieval.structural",
            attributes={"retrieval.leg": "structural"},
        ):
            struct_boosts = structural_boost(db, candidate_ids, claim)
        retriever_duration.record(
            time.monotonic() - leg_start, {"leg": "structural"},
        )

        # ---- RRF fusion ----
        raw_rrf: dict[UUID, float] = {sid: 0.0 for sid in candidate_ids}
        for sid in candidate_ids:
            if sid in sem_ranks:
                rank, _ = sem_ranks[sid]
                raw_rrf[sid] += weights.semantic / (_RRF_K + rank)
            if sid in lex_ranks:
                rank, _ = lex_ranks[sid]
                raw_rrf[sid] += weights.lexical / (_RRF_K + rank)
            if sid in identifier_ranks:
                rank, _ = identifier_ranks[sid]
                raw_rrf[sid] += weights.structural / (_RRF_K + rank)

        # ---- Apply structural boost (multiplicative) ----
        weight_sum = weights.semantic + weights.lexical + weights.structural
        # Normaliser: max theoretical RRF = sum(weights) / (k + 1)
        # (when every leg ranks d at position 1). We multiply so
        # the surface value lands in ``[0, sum(weights)]``.
        normaliser = (_RRF_K + 1)

        matches: list[HybridMatch] = []
        for sid in candidate_ids:
            raw = raw_rrf[sid]
            boost = struct_boosts.get(sid, 1.0)
            # raw * normaliser → max raw * normaliser = sum(weights)
            # at the theoretical-max RRF.
            normalised = raw * normaliser * boost
            # Clamp to [0, weight_sum * boost_max] so a future
            # structural cap change doesn't push us over 1.
            normalised = min(normalised, weight_sum * 2.2)

            matches.append(
                HybridMatch(
                    symbol_id=sid,
                    final_score=normalised,
                    rank_semantic=sem_ranks[sid][0] if sid in sem_ranks else None,
                    score_semantic=sem_ranks[sid][1] if sid in sem_ranks else None,
                    rank_lexical=lex_ranks[sid][0] if sid in lex_ranks else None,
                    score_lexical=lex_ranks[sid][1] if sid in lex_ranks else None,
                    rank_structural=identifier_ranks[sid][0] if sid in identifier_ranks else None,
                    struct_boost=boost,
                )
            )

        # Bulk-fetch the candidate rows so imports and orphaned hits can
        # be dropped before the final top_k cut.
        symbols = (
            db.query(CodeSymbol)
            .filter(CodeSymbol.symbol_id.in_([m.symbol_id for m in matches]))
            .all()
        )
        by_id = {s.symbol_id: s for s in symbols}

        if vendored is not None:
            matches = [
                replace(m, final_score=m.final_score * _VENDORED_FACTOR)
                if m.symbol_id in by_id
                and re.search(vendored, getattr(by_id[m.symbol_id], "file_path", ""), re.IGNORECASE)
                else m
                for m in matches
            ]

        # Sort by final_score desc, then by symbol_id (stable tie-break).
        matches.sort(key=lambda m: (-m.final_score, str(m.symbol_id)))

        # Filter out imports and any matches whose symbol row
        # vanished (orphan).
        eligible = [
            m
            for m in matches
            if m.symbol_id in by_id and getattr(by_id[m.symbol_id], "kind", None) != "import"
        ]
        survivors = _reserve_leg_tops(eligible, top_k)
        root_span.set_attribute("retrieval.result_count", len(survivors))
        return survivors


def _reserve_leg_tops(eligible: list[HybridMatch], top_k: int) -> list[HybridMatch]:
    """Cut ``eligible`` (score-sorted) to ``top_k``, keeping each leg's best.

    RRF favours symbols both legs found: in a small repo, where most
    symbols appear in both legs, a leg's rank-1 hit seen by that leg
    alone (``config/config.yaml`` for a claim listing the training
    hyperparameters) scores below a dozen mid-ranked ones. Each leg's
    top eligible symbol therefore displaces the lowest unreserved one.
    """
    top = eligible[:top_k]
    if len(eligible) <= top_k:
        return top
    reserved: list[HybridMatch] = []
    for leg_rank in ("rank_semantic", "rank_lexical", "rank_structural"):
        ranked = [m for m in eligible if getattr(m, leg_rank) is not None]
        if ranked:
            best = min(ranked, key=lambda m: getattr(m, leg_rank))
            if best not in reserved:
                reserved.append(best)
    for best in reserved:
        if best in top:
            continue
        drop = next((m for m in reversed(top) if m not in reserved), None)
        if drop is None:
            break
        top.remove(drop)
        top.append(best)
    top.sort(key=lambda m: (-m.final_score, str(m.symbol_id)))
    return top


def _best_ranks(passes: list[list[tuple[UUID, float]]]) -> dict[UUID, tuple[int, float]]:
    """Merge one leg's ranked passes: each symbol keeps its best (rank, score)."""
    out: dict[UUID, tuple[int, float]] = {}
    for results in passes:
        for i, (sid, score) in enumerate(results):
            if sid not in out or i + 1 < out[sid][0]:
                out[sid] = (i + 1, score)
    return out


def _vendored_pattern(db: Session, repository_id: UUID) -> str | None:
    """Regex matching the ``file_path`` of vendored code, or ``None``.

    Vendored code is any directory named after a declared dependency,
    plus — when that package sits in a sub-project with its own build
    manifest (``examples/NLU/setup.py`` next to
    ``examples/NLU/src/transformers``) — the whole sub-project. The
    pattern is valid both for Python ``re`` and Postgres ``~*``.
    """
    names = sorted(_vendored_dir_names(db, repository_id))
    if not names:
        return None
    alternatives = [r"(^|/)(" + "|".join(_package_regex(n) for n in names) + r")/"]
    roots = _vendored_roots(db, repository_id, names)
    if roots:
        alternatives.append("^(" + "|".join(re.escape(r) for r in sorted(roots)) + ")/")
    return "|".join(alternatives)


def _package_regex(normalised: str) -> str:
    """``py_yaml`` → ``py[-_.]yaml``: any PEP 503 spelling of the name."""
    return "[-_.]".join(re.escape(part) for part in normalised.split("_"))


def _vendored_dir_names(db: Session, repository_id: UUID) -> set[str]:
    """Normalised names of the repo's declared dependencies."""
    rows = (
        db.query(RepositoryDependency.name)
        .filter(RepositoryDependency.repository_id == repository_id)
        .all()
    )
    return {_normalise_package(row[0]) for row in rows if row[0]}


def _vendored_roots(db: Session, repository_id: UUID, dependency_names: list[str]) -> set[str]:
    """Sub-project directories that are a vendored copy of a dependency.

    A directory below the repo root holding a build manifest and a
    package (``<dir>/<dep>/`` or ``<dir>/src/<dep>/``) named after a
    declared dependency.
    """
    rows = (
        db.query(CodeSymbol.file_path)
        .filter(
            CodeSymbol.repository_id == repository_id,
            or_(*[CodeSymbol.file_path.like(f"%/{name}") for name in _MANIFESTS]),
        )
        .distinct()
        .all()
    )
    package = "(" + "|".join(_package_regex(n) for n in dependency_names) + ")"
    roots: set[str] = set()
    for (path,) in rows:
        root = path.rsplit("/", 1)[0]
        if root in roots:
            continue
        has_package = (
            db.query(CodeSymbol.symbol_id)
            .filter(
                CodeSymbol.repository_id == repository_id,
                CodeSymbol.file_path.op("~*")(f"^{re.escape(root)}/(src/)?{package}/"),
            )
            .first()
        )
        if has_package is not None:
            roots.add(root)
    return roots


def _normalise_package(name: str) -> str:
    """PEP 503-style: ``Py-YAML`` and ``py_yaml`` compare equal."""
    return re.sub(r"[-_.]+", "_", name).lower()


__all__ = ["hybrid_ranker", "HybridMatch", "HybridWeights"]
