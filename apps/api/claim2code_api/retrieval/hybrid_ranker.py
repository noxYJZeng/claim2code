"""Hybrid retrieval — Reciprocal Rank Fusion over (semantic, lexical, structural).

Phase 5 orchestrator. Given a paper claim, this is the **default**
read path. The endpoint (``POST /repositories/{id}/retrieve``)
dispatches to :func:`hybrid_ranker` when ``mode="hybrid"`` (the
default).

Algorithm
---------

Three legs each return their own top-``top_k * 3`` candidates
(more than ``top_k`` so each leg has room to surface its
long-tail picks):

  1. **Semantic** — :func:`retrieve_semantic` (Phase 4 pgvector
     cosine).
  2. **Lexical** — :func:`lexical_search` (Postgres tsvector +
     ts_rank_cd).
  3. **Structural** — :func:`structural_boost` (a multiplier,
     not a ranker; applied after fusion — see below).

The union of (semantic, lexical) candidate ``symbol_id``s
forms the candidate pool. For each candidate we compute the
Reciprocal Rank Fusion score:

    rrf(d) = Σ_i  w_i / (k + rank_i(d))

  - ``k = 60`` (Cormack et al. 2009 default; the standard
    ElasticSearch / OpenSearch RRF constant).
  - ``w_i ∈ {0.55, 0.25, 0.20}`` for ``{semantic, lexical,
    structural}``. The weights default to the pre-declared
    triple from ``evaluation/README.md:30`` but can be
    overridden via the ``weights`` parameter (Phase 5 eval uses
    this to study per-leg contribution).
  - ``rank_i(d)`` is the 1-indexed rank of ``d`` in leg ``i``'s
    result. Symbols **not** in leg ``i``'s result contribute
    ``0`` (we omit them from the sum).

Then:

  - Apply the structural multiplier (``>= 1.0``) to ``rrf(d)``.
  - Normalise to ``[0, 1]`` by multiplying by ``(k + 1)`` — the
    maximum theoretical RRF with all three legs ranking ``d`` at
    position 1 is ``Σ w_i / (k + 1) = 1 / (k + 1)`` (weights
    sum to 1); we multiply so the surface value lands in
    ``[0, 1]`` for the UI.

Why RRF over weighted-sum
-------------------------

Semantic returns ``0..1``, lexical returns unbounded ts_rank_cd,
structural returns ``1.0..2.2``. Weighted-sum requires score
normalisation before fusion, which is brittle (each leg would
need its own normaliser, calibrated against the same corpus).
RRF operates on **ranks** so the legs' score scales don't
matter. Standard multi-leg fusion pattern in Elasticsearch,
OpenSearch, Vespa, Weaviate.

Why structural is multiplicative, not a leg
--------------------------------------------

The structural scorer returns **multipliers**, not ranks —
it's not a retrieval leg, it's a re-ranker. We apply it after
RRF so the rank ordering from the ranker-fusion survives; only
scores within the top-k are nudged. (Multiplicative re-ranking
is the Elasticsearch ``function_score`` / Vespa pattern.)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from claim2code_api.db.models import CodeSymbol
from claim2code_api.indexing.embedding_client import EmbeddingClient
from claim2code_api.indexing.lexical import lexical_search
from claim2code_api.indexing.retriever import retrieve_semantic
from claim2code_api.indexing.structural import structural_boost
from claim2code_api.observability import get_tracer
from claim2code_api.observability.meter import retriever_request_duration_seconds

logger = logging.getLogger(__name__)

# RRF constant. Cormack et al. 2009; default in ElasticSearch,
# OpenSearch, Vespa. Lower k → rank differences amplified;
# higher k → RRF approaches simple weighted sum.
_RRF_K = 60


@dataclass(frozen=True)
class HybridWeights:
    """Per-leg RRF weights. Defaults match ``evaluation/README.md:30``.

    Sum is not required to be 1 — the normaliser uses the
    *theoretical max* (all legs rank 1) which only equals 1
    if the weights sum to 1. If the caller passes non-default
    weights that don't sum to 1 the normalised score's range
    is documented as ``[0, sum(weights)]`` instead of
    ``[0, 1]``. Tests assert the default case.
    """

    semantic: float = 0.55
    lexical: float = 0.25
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
    """Return top-``top_k`` symbols fused from semantic + lexical + structural.

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
        # ---- Leg 1: semantic ----
        leg_start = time.monotonic()
        with tracer.start_as_current_span(
            "retrieval.semantic",
            attributes={"retrieval.leg": "semantic"},
        ):
            sem_results = retrieve_semantic(
                db,
                repository_id,
                claim,
                top_k=leg_top_k,
                embedding_client=embedding_client,
            )
        retriever_duration.record(
            time.monotonic() - leg_start, {"leg": "semantic"},
        )

        # ---- Leg 2: lexical ----
        leg_start = time.monotonic()
        with tracer.start_as_current_span(
            "retrieval.lexical",
            attributes={"retrieval.leg": "lexical"},
        ):
            lex_results = lexical_search(
                db,
                repository_id,
                claim,
                top_k=leg_top_k,
            )
        retriever_duration.record(
            time.monotonic() - leg_start, {"leg": "lexical"},
        )

        # ---- Build rank + score maps per leg ----
        # Ranks are 1-indexed; not-in-leg → omit from sum.
        sem_ranks: dict[UUID, tuple[int, float]] = {
            sym.symbol_id: (i + 1, float(sim))
            for i, (sym, sim) in enumerate(sem_results)
        }
        lex_ranks: dict[UUID, tuple[int, float]] = {
            sid: (i + 1, float(score))
            for i, (sid, score) in enumerate(lex_results)
        }

        # Union of (sem,lex) candidates is the pool — structural
        # isn't a ranker, so it doesn't expand the candidate set.
        candidate_ids = list({*sem_ranks.keys(), *lex_ranks.keys()})
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
            # Structural is a multiplier, not a ranker; no RRF contribution.

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
                    rank_structural=None,  # structural isn't a ranker in v1
                    struct_boost=boost,
                )
            )

        # Sort by final_score desc, then by symbol_id (stable tie-break).
        matches.sort(key=lambda m: (-m.final_score, str(m.symbol_id)))

        # Bulk-fetch CodeSymbol rows for the top_k survivors so the
        # caller can render name / file_path / signature without a
        # second query.
        top = matches[:top_k]
        if not top:
            root_span.set_attribute("retrieval.result_count", 0)
            return []

        top_ids = [m.symbol_id for m in top]
        symbols = (
            db.query(CodeSymbol)
            .filter(CodeSymbol.symbol_id.in_(top_ids))
            .all()
        )
        by_id = {s.symbol_id: s for s in symbols}

        # Filter out any matches whose symbol row vanished (orphan).
        survivors = [m for m in top if m.symbol_id in by_id]
        root_span.set_attribute("retrieval.result_count", len(survivors))
        return survivors


__all__ = ["hybrid_ranker", "HybridMatch", "HybridWeights"]