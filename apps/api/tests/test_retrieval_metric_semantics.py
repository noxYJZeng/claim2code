"""Retrieval progress must retain gold denominators and historical definitions."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from claim2code_api.db.models import Base, CodeSymbol

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import evaluate_retrieval, report  # noqa: E402
from evaluation.evaluate_retrieval import (  # noqa: E402
    BaselineMetrics,
    ClaimResult,
    _render_results_json,
    evaluate_baseline_on_dataset,
)


def test_unresolved_targets_and_class_alternatives_have_fixed_denominators() -> None:
    result = ClaimResult(
        claim_id="partial-recall", relevant_symbol_ids={"class", "method"},
        retrieved_symbol_ids=["method", "class"],
        relevant_symbol_groups=(frozenset({"class", "method"}), frozenset()),
        relevant_symbols=({"name": "Layer"}, {"name": "missing_function"}),
    )
    assert result.recall_at_1 == 0.5
    assert result.recall_at_5 == 0.5
    assert result.hit_at_5 == 1.0
    assert result.unresolved_symbols == [{"name": "missing_function"}]


def test_parent_disambiguation_and_class_resolution_use_real_database_queries() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    repo_id = uuid4()

    def symbol(name, kind, line, *, repo=repo_id, parent=None):
        return CodeSymbol(
            symbol_id=uuid4(), repository_id=repo, kind=kind, language="python",
            file_path="model.py", name=name, signature="", start_line=line,
            end_line=line + 1, start_col=0, end_col=4, parent_symbol_id=parent,
        )

    try:
        with Session(engine) as db:
            first = symbol("TextEncoder", "class", 1)
            other = symbol("ImageEncoder", "class", 10)
            foreign = symbol("TextEncoder", "class", 1, repo=uuid4())
            forward = symbol("forward", "method", 2, parent=first.symbol_id)
            other_forward = symbol("forward", "method", 11, parent=other.symbol_id)
            foreign_forward = symbol("forward", "method", 2, repo=foreign.repository_id,
                                     parent=foreign.symbol_id)
            db.add_all([first, other, foreign, forward, other_forward, foreign_forward])
            db.flush()
            targets = [
                {"file_path": "model.py", "name": "TextEncoder", "kind": "class"},
                {"file_path": "model.py", "name": "forward", "kind": "method", "parent_name": "TextEncoder"},
                {"file_path": "missing.py", "name": "missing", "kind": "function"},
            ]
            groups = evaluate_retrieval._resolve_symbol_groups(db, repo_id, targets)
            assert groups == (frozenset({str(first.symbol_id), str(forward.symbol_id)}),
                              frozenset({str(forward.symbol_id)}), frozenset())
    finally:
        engine.dispose()


def test_rankings_resolution_failures_and_baseline_errors_are_saved(monkeypatch) -> None:
    symbols = [{"file_path": "m.py", "name": "first", "kind": "function"},
               {"file_path": "m.py", "name": "missing", "kind": "function"}]
    annotations = {"pair_id": "p", "claims": [
        {"claim_id": "ok", "text": "ok", "relevant_symbols": symbols},
        {"claim_id": "error", "text": "error", "relevant_symbols": symbols},
    ]}
    monkeypatch.setattr(evaluate_retrieval, "_resolve_symbol_groups",
                        lambda *_args: (frozenset({"a"}), frozenset()))

    def run(_db, _repo, claim, *, top_k):
        if claim == "error":
            raise RuntimeError("offline")
        return [("a", 0.9), ("unrelated", 0.7)]

    metrics = evaluate_baseline_on_dataset(
        db=MagicMock(), repository_id=uuid4(), annotation=annotations,
        baseline_mod=SimpleNamespace(NAME="embedding_only", run=run),
    )
    data = _render_results_json([metrics])
    assert data["schema_version"] == 2
    assert data["metric_semantics"] == "symbol_recall_v2"
    assert data["aggregate_by_baseline"]["embedding_only"]["avg_recall_at_5"] == 0.25
    assert data["aggregate_by_baseline"]["embedding_only"]["avg_hit_at_5"] == 0.5
    assert data["per_dataset"][0]["n_unresolved_symbols"] == 2
    assert data["per_dataset"][0]["n_claim_errors"] == 1
    success, failure = data["per_claim"]
    assert success["resolved_symbol_groups"] == [["a"], []]
    assert success["retrieved_symbol_ids"] == ["a", "unrelated"]
    assert success["retrieved_scores"] == [0.9, 0.7]
    assert failure["error"] == "RuntimeError: offline"
    assert failure["recall_at_5"] == 0.0


def test_legacy_report_preserves_hit_rates_without_claiming_symbol_recall() -> None:
    archived = {"per_dataset": [{
        "baseline": "hybrid_claim2code", "dataset": "old", "n_claims": 79,
        "recall_at_1": 0.40, "recall_at_5": 0.74, "mrr": 0.529,
    }]}
    text = "\n".join(report.render_retrieval(archived))
    assert "not symbol recall" in text
    assert "Symbol Recall@1/5 is unavailable" in text
    assert "Avg Hit@5" in text
    assert "0.74" in text
    assert "Avg R@5" not in text
    assert "metric_semantics" not in archived


def test_new_report_separates_recall_and_hit_rate() -> None:
    metrics = BaselineMetrics(
        baseline="hybrid_claim2code", dataset="new", n_claims=1,
        recall_at_1=0.5, recall_at_5=0.5, mrr=1.0, hit_at_1=1.0, hit_at_5=1.0,
    )
    text = "\n".join(report.render_retrieval(_render_results_json([metrics])))
    assert "fraction of annotated relevant symbols" in text
    assert "Avg R@5" in text and "Avg Hit@5" in text
    assert "0.50" in text and "1.00" in text


def test_run_report_outputs_do_not_replace_archived_artifacts(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    archived_json = archive / "results.json"
    archived_md = archive / "report.md"
    archived_json.write_text('{"historical": true}')
    archived_md.write_text("historical report")
    monkeypatch.setattr(report, "RESULTS_JSON", archived_json)
    monkeypatch.setattr(report, "REPORT_MD", archived_md)
    run = tmp_path / "run"
    results = run / "results"
    results.mkdir(parents=True)
    metrics = BaselineMetrics("hybrid_claim2code", "p", 1, 0.5, 0.5, 1.0,
                              hit_at_1=1.0, hit_at_5=1.0)
    (results / "retrieval.json").write_text(json.dumps(_render_results_json([metrics])))
    assert report.main(["--results-dir", str(results), "--output-dir", str(run)]) == 0
    assert archived_json.read_text() == '{"historical": true}'
    assert archived_md.read_text() == "historical report"
    assert json.loads((run / "results.json").read_text())["retrieval"]["schema_version"] == 2


def test_unknown_pair_stops_before_database_or_model_calls(monkeypatch) -> None:
    monkeypatch.setattr(evaluate_retrieval, "_load_annotations", lambda: [{"pair_id": "known"}])
    monkeypatch.setattr(evaluate_retrieval, "_build_session_factory",
                        lambda: pytest.fail("unknown pairs must fail before opening a database"))
    assert evaluate_retrieval.main(["--pairs", "typo"]) == 2
