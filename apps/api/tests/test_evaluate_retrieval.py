"""Unit tests for ``evaluation/evaluate_retrieval.py``.

The eval harness is a thin orchestrator — the heavy lifting
lives in the per-leg scorers (covered by
``tests/test_retriever.py``, ``tests/test_lexical.py``,
``tests/test_hybrid_ranker.py``). These tests verify the
harness-side contracts:

  - ``ClaimResult`` correctly computes Recall@1 / Recall@5 / MRR
  - ``evaluate_baseline_on_dataset`` aggregates per-claim results
  - ``_render_report_md`` produces a valid Markdown table
  - ``_render_results_json`` aggregates by baseline
  - Per-baseline failures don't crash the loop (one bad row is
    marked ``error`` and the next claim still runs)

The baseline ``run()`` functions and the DB session are
mocked; these tests don't touch Postgres or OpenAI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Add the repo root to sys.path so ``evaluation/`` resolves
# (``evaluation/`` lives at the repo root, one level above
# ``apps/api``). Also add ``apps/api`` so the eval runner's
# ``import claim2code_api`` works in case pytest strips cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(_REPO_ROOT), str(_REPO_ROOT / "apps" / "api")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import AFTER sys.path setup.
from evaluation.evaluate_retrieval import (  # noqa: E402
    BaselineMetrics,
    ClaimResult,
    _render_report_md,
    _render_results_json,
    evaluate_baseline_on_dataset,
)

# ============================================================
# ClaimResult metric computation
# ============================================================


def test_claim_result_recall_at_1_hit() -> None:
    """The first retrieved symbol is in the relevant set → R@1 = 1.0."""
    cr = ClaimResult(
        claim_id="c1",
        relevant_symbol_ids={"a", "b"},
        retrieved_symbol_ids=["a", "x", "y", "z", "w"],
    )
    assert cr.recall_at_1 == 1.0


def test_claim_result_recall_at_1_miss() -> None:
    """The first retrieved symbol is NOT in the relevant set → R@1 = 0.0."""
    cr = ClaimResult(
        claim_id="c1",
        relevant_symbol_ids={"a"},
        retrieved_symbol_ids=["x", "a", "y", "z", "w"],
    )
    assert cr.recall_at_1 == 0.0


def test_claim_result_recall_at_5_hit() -> None:
    """A relevant symbol in the top 5 → R@5 = 1.0."""
    cr = ClaimResult(
        claim_id="c1",
        relevant_symbol_ids={"a"},
        retrieved_symbol_ids=["x", "y", "a", "z", "w"],
    )
    assert cr.recall_at_5 == 1.0


def test_claim_result_recall_at_5_miss_outside_top5() -> None:
    """A relevant symbol only in position 6+ → R@5 = 0.0."""
    cr = ClaimResult(
        claim_id="c1",
        relevant_symbol_ids={"a"},
        retrieved_symbol_ids=["x", "y", "z", "w", "v", "a"],
    )
    assert cr.recall_at_5 == 0.0


def test_claim_result_mrr_first_position() -> None:
    """Relevant at position 1 → MRR = 1.0."""
    cr = ClaimResult(
        claim_id="c1",
        relevant_symbol_ids={"a"},
        retrieved_symbol_ids=["a", "x", "y"],
    )
    assert cr.mrr == 1.0


def test_claim_result_mrr_third_position() -> None:
    """Relevant at position 3 → MRR = 1/3."""
    cr = ClaimResult(
        claim_id="c1",
        relevant_symbol_ids={"a"},
        retrieved_symbol_ids=["x", "y", "a"],
    )
    assert cr.mrr == pytest.approx(1.0 / 3.0)


def test_claim_result_mrr_no_match() -> None:
    """No relevant retrieved → MRR = 0.0."""
    cr = ClaimResult(
        claim_id="c1",
        relevant_symbol_ids={"a"},
        retrieved_symbol_ids=["x", "y", "z"],
    )
    assert cr.mrr == 0.0


def test_claim_result_multiple_relevant_uses_union() -> None:
    """Multiple relevant symbols → any hit counts."""
    cr = ClaimResult(
        claim_id="c1",
        relevant_symbol_ids={"a", "b", "c"},
        retrieved_symbol_ids=["x", "b", "y"],
    )
    assert cr.recall_at_1 == 0.0  # first one is "x", miss
    assert cr.recall_at_5 == 1.0  # "b" is in the top 5
    assert cr.mrr == pytest.approx(0.5)  # "b" at position 2


def test_claim_result_empty_retrieved_does_not_crash() -> None:
    """Empty retrieved list → all 0.0 (no IndexError on recall_at_1)."""
    cr = ClaimResult(
        claim_id="c1",
        relevant_symbol_ids={"a"},
        retrieved_symbol_ids=[],
    )
    assert cr.recall_at_1 == 0.0
    assert cr.recall_at_5 == 0.0
    assert cr.mrr == 0.0


# ============================================================
# evaluate_baseline_on_dataset — aggregation
# ============================================================


def _fake_baseline_module(rows_by_claim: dict[str, list[tuple[str, float]]]):
    """Build a baseline module stub with ``NAME`` + ``run``."""
    mod = MagicMock()
    mod.NAME = "test_baseline"

    def _run(_db, _repo_id, claim: str, *, top_k: int = 5):
        return rows_by_claim.get(claim, [])

    mod.run.side_effect = _run
    return mod


def test_evaluate_baseline_aggregates_metrics() -> None:
    """Per-claim results are aggregated into a BaselineMetrics row."""
    annotation = {
        "pair_id": "test_pair",
        "claims": [
            {"claim_id": "c1", "text": "claim 1", "relevant_symbols": []},
            {"claim_id": "c2", "text": "claim 2", "relevant_symbols": []},
        ],
    }

    # Both claims return no rows → R@1 = 0, R@5 = 0, MRR = 0.
    baseline = _fake_baseline_module({"claim 1": [], "claim 2": []})

    db = MagicMock()
    db.query.return_value.filter_by.return_value.all.return_value = []

    m = evaluate_baseline_on_dataset(
        db=db,
        repository_id=MagicMock(),
        annotation=annotation,
        baseline_mod=baseline,
    )
    assert isinstance(m, BaselineMetrics)
    assert m.baseline == "test_baseline"
    assert m.dataset == "test_pair"
    assert m.n_claims == 2
    assert m.recall_at_1 == 0.0
    assert m.recall_at_5 == 0.0
    assert m.mrr == 0.0
    assert m.error is None


def test_evaluate_baseline_handles_per_claim_failure() -> None:
    """A single failing claim is captured; the loop continues."""
    annotation = {
        "pair_id": "test_pair",
        "claims": [
            {"claim_id": "c1", "text": "claim 1", "relevant_symbols": []},
            {"claim_id": "c2", "text": "claim 2", "relevant_symbols": []},
            {"claim_id": "c3", "text": "claim 3", "relevant_symbols": []},
        ],
    }

    # c2 raises; c1, c3 succeed.
    rows = {"claim 1": [("a", 1.0)], "claim 3": [("b", 1.0)]}

    def _run(_db, _repo_id, claim: str, *, top_k: int = 5):
        if claim == "claim 2":
            raise RuntimeError("simulated failure")
        return rows.get(claim, [])

    baseline = MagicMock()
    baseline.NAME = "test_baseline"
    baseline.run.side_effect = _run

    db = MagicMock()
    db.query.return_value.filter_by.return_value.all.return_value = []

    m = evaluate_baseline_on_dataset(
        db=db,
        repository_id=MagicMock(),
        annotation=annotation,
        baseline_mod=baseline,
    )
    assert m.n_claims == 3
    # c2's failure contributes empty retrieval → R@1=0; the
    # others return ["a"] / ["b"] with no relevant symbols → 0.
    assert m.recall_at_1 == 0.0
    assert m.error is None


def test_evaluate_baseline_resolves_relevant_symbols() -> None:
    """``relevant_symbols`` (file_path, name, kind) resolve to symbol_ids."""
    annotation = {
        "pair_id": "test_pair",
        "claims": [
            {
                "claim_id": "c1",
                "text": "claim 1",
                "relevant_symbols": [
                    {"file_path": "pkg/foo.py", "name": "bar", "kind": "function"},
                ],
            },
        ],
    }
    baseline = _fake_baseline_module({"claim 1": [("resolved-id", 1.0)]})

    # First call resolves the relevant symbol, second call is
    # the baseline.run (we mock it; not a real query).
    db = MagicMock()
    db.query.return_value.filter_by.return_value.all.return_value = [
        SimpleNamespace(symbol_id="resolved-id"),
    ]

    m = evaluate_baseline_on_dataset(
        db=db,
        repository_id=MagicMock(),
        annotation=annotation,
        baseline_mod=baseline,
    )
    assert m.n_claims == 1
    # The resolved symbol_id matches what the baseline
    # returned → R@1 = 1.0, R@5 = 1.0, MRR = 1.0.
    assert m.recall_at_1 == 1.0
    assert m.recall_at_5 == 1.0
    assert m.mrr == 1.0


# ============================================================
# Report / JSON rendering
# ============================================================


def _metric(*, baseline: str, dataset: str, r1: float, r5: float, mrr: float) -> BaselineMetrics:
    return BaselineMetrics(
        baseline=baseline,
        dataset=dataset,
        n_claims=10,
        recall_at_1=r1,
        recall_at_5=r5,
        mrr=mrr,
    )


def test_render_report_md_contains_all_datasets() -> None:
    """The rendered report has a section per dataset + an Aggregate."""
    metrics = [
        _metric(baseline="embedding_only", dataset="bert", r1=0.5, r5=0.8, mrr=0.6),
        _metric(baseline="lexical_only", dataset="bert", r1=0.4, r5=0.7, mrr=0.5),
        _metric(baseline="hybrid_claim2code", dataset="bert", r1=0.6, r5=0.9, mrr=0.7),
    ]
    md = _render_report_md(metrics)
    assert "# Retrieval Evaluation Report" in md
    assert "## bert" in md
    assert "## Aggregate" in md
    assert "embedding_only" in md
    assert "lexical_only" in md
    assert "hybrid_claim2code" in md


def test_render_report_md_handles_missing_baseline() -> None:
    """A dataset with only some baselines → "(no data)" rows for the rest."""
    metrics = [
        _metric(baseline="embedding_only", dataset="bert", r1=0.5, r5=0.8, mrr=0.6),
    ]
    md = _render_report_md(metrics)
    assert "(no data)" in md


def test_render_results_json_structure() -> None:
    """``results.json`` has per_dataset + aggregate_by_baseline keys."""
    metrics = [
        _metric(baseline="embedding_only", dataset="bert", r1=0.5, r5=0.8, mrr=0.6),
        _metric(baseline="hybrid_claim2code", dataset="bert", r1=0.6, r5=0.9, mrr=0.7),
        _metric(baseline="embedding_only", dataset="lora", r1=0.4, r5=0.7, mrr=0.5),
        _metric(baseline="hybrid_claim2code", dataset="lora", r1=0.5, r5=0.8, mrr=0.6),
    ]
    out = _render_results_json(metrics)
    assert "per_dataset" in out
    assert "aggregate_by_baseline" in out
    assert len(out["per_dataset"]) == 4
    # aggregate: avg_recall_at_5 = (0.8 + 0.7) / 2 = 0.75 for embedding_only
    assert out["aggregate_by_baseline"]["embedding_only"]["avg_recall_at_5"] == pytest.approx(0.75)


def test_render_results_json_is_json_serialisable() -> None:
    """The dict round-trips through json.dumps/json.loads."""
    metrics = [
        _metric(baseline="embedding_only", dataset="bert", r1=0.5, r5=0.8, mrr=0.6),
    ]
    out = _render_results_json(metrics)
    blob = json.dumps(out)
    reloaded = json.loads(blob)
    assert reloaded == out


# ============================================================
# Annotation file loading (sanity check the JSON is valid)
# ============================================================


def test_annotation_files_load_with_labelled_claims() -> None:
    """Every annotation file loads, with at least 8 labelled claims."""
    from evaluation.evaluate_retrieval import _load_annotations

    annotations = _load_annotations()
    by_id = {a["pair_id"]: a for a in annotations}
    assert set(by_id) == {
        "attention_is_all_you_need", "bert", "lora", "mae", "ddim", "reflexion", "dpo",
    }
    for ann in annotations:
        assert len(ann["claims"]) >= 8, f"{ann['pair_id']} should have >= 8 claims"
        # Every claim has text + relevant_symbols.
        for c in ann["claims"]:
            assert "text" in c
            assert "claim_id" in c
            assert "relevant_symbols" in c
            assert len(c["relevant_symbols"]) >= 1