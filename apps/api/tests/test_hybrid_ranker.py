"""Unit tests for :mod:`claim2code_api.retrieval.hybrid_ranker`.

Coverage:

  - Empty / whitespace claim short-circuits to ``[]``.
  - RRF formula: 1 candidate in 1 leg at rank 1 → ``final_score = w_i``.
  - RRF formula: 1 candidate in both legs at rank 1 →
    ``final_score = w_sem + w_lex``.
  - Symbol in only one leg → only that leg contributes.
  - Structural boost is multiplicative on RRF.
  - Top-k truncation.
  - Final order is ``final_score`` desc (stable tie-break).
  - Custom weights override defaults.
  - ``EmbeddingError`` from semantic leg re-raises.
  - Orphaned symbol rows (no CodeSymbol) are silently dropped.
  - Bulk-fetch queries the CodeSymbol table.

Both legs (``retrieve_semantic`` and ``lexical_search``) and
the structural scorer are heavy external dependencies —
tests stub them via ``unittest.mock.patch``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from claim2code_api.db.models import RepositoryDependency
from claim2code_api.indexing.exceptions import EmbeddingError
from claim2code_api.retrieval.hybrid_ranker import (
    HybridWeights,
    hybrid_ranker,
)

# ============================================================
# Helpers
# ============================================================


def _sem_result(score: float, symbol_id):
    """Build a single ``(symbol, similarity)`` tuple from retrieve_semantic."""
    sym = SimpleNamespace(symbol_id=symbol_id)
    return (sym, score)


def _lex_result(score: float, symbol_id):
    """Build a single ``(symbol_id, score)`` tuple from lexical_search."""
    return (symbol_id, score)


def _build_session(
    *, code_symbols: list[SimpleNamespace], dependencies: list[str] | None = None,
) -> MagicMock:
    """Build a MagicMock session that mimics the ranker's two queries.

    After RRF the hybrid_ranker bulk-fetches CodeSymbol rows by id
    (we return ``code_symbols``) and reads the repo's dependency
    names (``dependencies``, default none).
    """
    session = MagicMock(name="session")
    queue = [code_symbols]

    def _query(model):
        q = MagicMock()
        if model is RepositoryDependency.name:
            q.filter.return_value.all.return_value = [(n,) for n in dependencies or []]
        else:
            q.filter.return_value.all.return_value = queue.pop(0)
        return q

    session.query.side_effect = _query
    return session


# ============================================================
# Short-circuits
# ============================================================


def test_hybrid_ranker_empty_claim_returns_empty() -> None:
    """Empty claim → no leg calls, empty result."""
    db = _build_session(code_symbols=[])
    out = hybrid_ranker(db, uuid4(), "")
    assert out == []


def test_hybrid_ranker_whitespace_claim_returns_empty() -> None:
    """Whitespace-only claim → empty result."""
    db = _build_session(code_symbols=[])
    out = hybrid_ranker(db, uuid4(), "   \n  \t  ")
    assert out == []


# ============================================================
# RRF formula — single leg
# ============================================================


def test_hybrid_ranker_single_candidate_in_semantic_only() -> None:
    """1 candidate in semantic at rank 1 → final_score = 0.40.

    Formula: rrf = w / (k + rank) = 0.40 / 61 = ~ 0.00656.
    Normalised: rrf * 61 * boost = 0.40 * 1.0 = 0.40.
    """
    sid = uuid4()
    syms = [SimpleNamespace(symbol_id=sid)]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[_sem_result(0.9, sid)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid: 1.0},
        ),
    ):
        out = hybrid_ranker(_build_session(code_symbols=syms), uuid4(), "x")

    assert len(out) == 1
    assert out[0].symbol_id == sid
    assert out[0].final_score == pytest.approx(0.40, abs=1e-6)
    assert out[0].rank_semantic == 1
    assert out[0].score_semantic == pytest.approx(0.9)
    assert out[0].rank_lexical is None


def test_hybrid_ranker_single_candidate_in_lexical_only() -> None:
    """1 candidate in lexical at rank 1 → final_score = 0.40."""
    sid = uuid4()
    syms = [SimpleNamespace(symbol_id=sid)]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[_lex_result(0.7, sid)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid: 1.0},
        ),
    ):
        out = hybrid_ranker(_build_session(code_symbols=syms), uuid4(), "x")

    assert len(out) == 1
    assert out[0].final_score == pytest.approx(0.40, abs=1e-6)
    assert out[0].rank_lexical == 1
    assert out[0].rank_semantic is None


# ============================================================
# RRF formula — both legs
# ============================================================


def test_hybrid_ranker_candidate_in_both_legs_at_rank_one() -> None:
    """1 candidate in both legs at rank 1 → final_score = 0.40 + 0.40 = 0.80.

    (sum of weights at default configuration).
    """
    sid = uuid4()
    syms = [SimpleNamespace(symbol_id=sid)]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[_sem_result(0.9, sid)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[_lex_result(0.7, sid)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid: 1.0},
        ),
    ):
        out = hybrid_ranker(_build_session(code_symbols=syms), uuid4(), "x")

    assert len(out) == 1
    assert out[0].final_score == pytest.approx(0.80, abs=1e-6)


def test_hybrid_ranker_rank_two_contributes_less_than_rank_one() -> None:
    """Rank 2 contributes less than rank 1 (RRF denominator)."""
    sid_rank1 = uuid4()
    sid_rank2 = uuid4()
    syms = [
        SimpleNamespace(symbol_id=sid_rank1),
        SimpleNamespace(symbol_id=sid_rank2),
    ]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[
                _sem_result(0.9, sid_rank1),
                _sem_result(0.8, sid_rank2),
            ],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid_rank1: 1.0, sid_rank2: 1.0},
        ),
    ):
        out = hybrid_ranker(_build_session(code_symbols=syms), uuid4(), "x")

    # rank1: 0.40 / 61 * 61 = 0.40
    # rank2: 0.40 / 62 * 61 ≈ 0.3935
    assert out[0].symbol_id == sid_rank1
    assert out[0].final_score == pytest.approx(0.40, abs=1e-6)
    assert out[1].symbol_id == sid_rank2
    assert out[1].final_score == pytest.approx(0.3935, abs=1e-3)


# ============================================================
# Structural boost
# ============================================================


def test_hybrid_ranker_structural_boost_multiplies_score() -> None:
    """Structural boost of 2.0 → final_score doubled (compared to boost 1.0)."""
    sid = uuid4()
    syms = [SimpleNamespace(symbol_id=sid)]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[_sem_result(0.9, sid)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid: 2.0},
        ),
    ):
        out = hybrid_ranker(_build_session(code_symbols=syms), uuid4(), "x")

    # baseline: 0.40 → boosted: 0.80
    assert out[0].final_score == pytest.approx(0.80, abs=1e-6)
    assert out[0].struct_boost == 2.0


def test_hybrid_ranker_structural_boost_default_one() -> None:
    """A symbol missing from the boost dict defaults to 1.0."""
    sid = uuid4()
    syms = [SimpleNamespace(symbol_id=sid)]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[_sem_result(0.9, sid)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={},  # empty — default to 1.0
        ),
    ):
        out = hybrid_ranker(_build_session(code_symbols=syms), uuid4(), "x")

    assert out[0].struct_boost == 1.0


# ============================================================
# Top-k truncation
# ============================================================


def test_hybrid_ranker_truncates_to_top_k() -> None:
    """top_k=3 with 5 candidates → 3 results."""
    ids = [uuid4() for _ in range(5)]
    syms = [SimpleNamespace(symbol_id=sid) for sid in ids]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[_sem_result(1.0 - 0.1 * i, sid) for i, sid in enumerate(ids)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid: 1.0 for sid in ids},
        ),
    ):
        out = hybrid_ranker(
            _build_session(code_symbols=syms), uuid4(), "x", top_k=3,
        )

    assert len(out) == 3
    # The top 3 (highest sim) are ids[0], ids[1], ids[2].
    assert [m.symbol_id for m in out] == ids[:3]


def test_hybrid_ranker_default_top_k_is_5() -> None:
    """No top_k → default 5."""
    ids = [uuid4() for _ in range(10)]
    syms = [SimpleNamespace(symbol_id=sid) for sid in ids]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[_sem_result(1.0 - 0.05 * i, sid) for i, sid in enumerate(ids)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid: 1.0 for sid in ids},
        ),
    ):
        out = hybrid_ranker(_build_session(code_symbols=syms), uuid4(), "x")

    assert len(out) == 5


# ============================================================
# Order
# ============================================================


def test_hybrid_ranker_results_sorted_by_final_score_desc() -> None:
    """Results are returned sorted by final_score descending."""
    sid_high = uuid4()
    sid_mid = uuid4()
    sid_low = uuid4()
    syms = [
        SimpleNamespace(symbol_id=sid_low),
        SimpleNamespace(symbol_id=sid_mid),
        SimpleNamespace(symbol_id=sid_high),
    ]

    # semantic ranks high < mid < low; lexical ranks low < mid.
    # With k=60 + default weights {0.40, 0.40, 0.20}:
    #   high: rank1 sem + none lex → 0.40/61 = 0.006557 → 0.400 final
    #   mid:  rank2 sem + rank2 lex → 0.40/62 + 0.40/62 = 0.012903 → 0.787
    #   low:  rank3 sem + rank1 lex → 0.40/63 + 0.40/61 = 0.012906 → 0.787
    # Therefore the expected order is low > mid > high.
    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[
                _sem_result(0.9, sid_high),
                _sem_result(0.8, sid_mid),
                _sem_result(0.7, sid_low),
            ],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[
                _lex_result(0.9, sid_low),
                _lex_result(0.8, sid_mid),
            ],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid_high: 1.0, sid_mid: 1.0, sid_low: 1.0},
        ),
    ):
        out = hybrid_ranker(_build_session(code_symbols=syms), uuid4(), "x")

    assert [m.symbol_id for m in out] == [sid_low, sid_mid, sid_high]


def test_hybrid_ranker_drops_import_symbols() -> None:
    """An import line is not evidence; the next symbol takes its slot."""
    sid_import = uuid4()
    sid_func = uuid4()
    syms = [
        SimpleNamespace(symbol_id=sid_import, kind="import"),
        SimpleNamespace(symbol_id=sid_func, kind="function"),
    ]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[_sem_result(0.9, sid_import), _sem_result(0.8, sid_func)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid_import: 1.0, sid_func: 1.0},
        ),
    ):
        out = hybrid_ranker(_build_session(code_symbols=syms), uuid4(), "x", top_k=1)

    assert [m.symbol_id for m in out] == [sid_func]


def test_hybrid_ranker_reserves_a_slot_for_each_legs_top_hit() -> None:
    """A lexical rank-1 hit the semantic leg missed outranks nothing
    under RRF, but still gets a top-k slot."""
    both = [uuid4() for _ in range(3)]
    lex_only = uuid4()
    syms = [SimpleNamespace(symbol_id=s, kind="function") for s in [*both, lex_only]]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[_sem_result(0.9, s) for s in both],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[_lex_result(1.0, lex_only)] + [_lex_result(0.5, s) for s in both],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={s: 1.0 for s in [*both, lex_only]},
        ),
    ):
        out = hybrid_ranker(_build_session(code_symbols=syms), uuid4(), "x", top_k=2)

    assert [m.symbol_id for m in out] == [both[0], lex_only]


def test_hybrid_ranker_down_weights_vendored_dependency_dirs() -> None:
    """A symbol under a directory named after a declared dependency
    (a vendored copy of that library) is scored at half weight."""
    own, vendored = uuid4(), uuid4()
    syms = [
        SimpleNamespace(symbol_id=vendored, kind="function", file_path="src/transformers/models/bert.py"),
        SimpleNamespace(symbol_id=own, kind="function", file_path="loralib/layers.py"),
    ]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[_sem_result(0.9, vendored), _sem_result(0.8, own)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={vendored: 1.0, own: 1.0},
        ),
    ):
        db = _build_session(code_symbols=syms, dependencies=["Transformers", "torch"])
        out = hybrid_ranker(db, uuid4(), "x")

    assert [m.symbol_id for m in out] == [own, vendored]
    assert out[1].final_score == pytest.approx(0.40 / 61 * 61 * 0.5)


# ============================================================
# Custom weights
# ============================================================


def test_hybrid_ranker_custom_weights() -> None:
    """Custom weights override the defaults."""
    sid = uuid4()
    syms = [SimpleNamespace(symbol_id=sid)]

    custom = HybridWeights(semantic=1.0, lexical=0.0, structural=0.0)

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[_sem_result(0.9, sid)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid: 1.0},
        ),
    ):
        out = hybrid_ranker(
            _build_session(code_symbols=syms),
            uuid4(),
            "x",
            weights=custom,
        )

    # Pure semantic → final_score = 1.0 * (61/61) * 1.0 = 1.0
    assert out[0].final_score == pytest.approx(1.0, abs=1e-6)


def test_hybrid_ranker_weights_sum_to_one_normalisation() -> None:
    """Default weights sum to 1.0 → max final_score at theoretical-max RRF is 1.0."""
    sid = uuid4()
    syms = [SimpleNamespace(symbol_id=sid)]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[_sem_result(0.9, sid)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[_lex_result(0.7, sid)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid: 1.0},
        ),
    ):
        out = hybrid_ranker(_build_session(code_symbols=syms), uuid4(), "x")

    # rank 1 in both legs, default weights → 0.55 + 0.25 = 0.80
    assert out[0].final_score == pytest.approx(0.80, abs=1e-6)


# ============================================================
# Error propagation
# ============================================================


def test_hybrid_ranker_reraises_embedding_error() -> None:
    """EmbeddingError from the semantic leg propagates."""
    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            side_effect=EmbeddingError("openai is down"),
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[],
        ),
    ):
        with pytest.raises(EmbeddingError, match=r"openai is down"):
            hybrid_ranker(_build_session(code_symbols=[]), uuid4(), "x")


# ============================================================
# Orphan handling
# ============================================================


def test_hybrid_ranker_orphans_dropped_silently() -> None:
    """A match whose CodeSymbol row vanished is dropped from the output."""
    sid_real = uuid4()
    sid_orphan = uuid4()

    # semantic returns both; bulk-fetch returns only the real one.
    syms = [SimpleNamespace(symbol_id=sid_real)]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[
                _sem_result(0.9, sid_real),
                _sem_result(0.85, sid_orphan),
            ],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid_real: 1.0, sid_orphan: 1.0},
        ),
    ):
        out = hybrid_ranker(_build_session(code_symbols=syms), uuid4(), "x")

    assert len(out) == 1
    assert out[0].symbol_id == sid_real


# ============================================================
# Embedding client passthrough
# ============================================================


def test_hybrid_ranker_passes_through_embedding_client() -> None:
    """The optional embedding_client argument is forwarded to retrieve_semantic."""
    sentinel = object()

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[],
        ) as sem_mock,
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={},
        ),
    ):
        hybrid_ranker(
            _build_session(code_symbols=[]),
            uuid4(),
            "x",
            embedding_client=sentinel,  # type: ignore[arg-type]
        )

    # The keyword is forwarded (as embedding_client=...).
    assert sem_mock.call_args.kwargs.get("embedding_client") is sentinel


# ============================================================
# Structural boost passthrough
# ============================================================


def test_hybrid_ranker_passes_candidate_ids_to_structural_boost() -> None:
    """structural_boost receives the union of (sem, lex) candidate ids."""
    sid_a = uuid4()
    sid_b = uuid4()

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[_sem_result(0.9, sid_a)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[_lex_result(0.7, sid_b)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid_a: 1.0, sid_b: 1.0},
        ) as struct_mock,
    ):
        hybrid_ranker(_build_session(code_symbols=[]), uuid4(), "x")

    candidates = struct_mock.call_args.args[1]
    assert set(candidates) == {sid_a, sid_b}


def test_hybrid_ranker_passes_claim_to_structural_boost() -> None:
    """structural_boost receives the original claim string.

    Note: a candidate must exist (i.e. at least one of
    ``retrieve_semantic`` / ``lexical_search`` must return a
    non-empty list) for the ranker to call ``structural_boost``
    at all — otherwise the ranker short-circuits with ``[]``.
    """
    sid = uuid4()
    syms = [SimpleNamespace(symbol_id=sid)]

    with (
        patch(
            "claim2code_api.retrieval.hybrid_ranker.retrieve_semantic",
            return_value=[_sem_result(0.9, sid)],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.lexical_search",
            return_value=[],
        ),
        patch(
            "claim2code_api.retrieval.hybrid_ranker.structural_boost",
            return_value={sid: 1.0},
        ) as struct_mock,
    ):
        hybrid_ranker(_build_session(code_symbols=syms), uuid4(), "the claim text")

    assert struct_mock.call_args.args[2] == "the claim text"