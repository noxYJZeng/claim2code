"""Measurement identity must survive rendering, resuming and regrading."""
from __future__ import annotations

import copy
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import evaluate_verification, provenance, report  # noqa: E402
from evaluation.evaluate_verification import _read_clone_file, _regrade, score  # noqa: E402


def record(expected: str, predicted: str, claim_id: str = "c") -> dict:
    return {
        "pair_id": "p", "claim_id": claim_id, "expected": expected, "predicted": predicted,
        "n_evidence": 0, "n_grounded": 0, "evidence_hit": False, "error": None,
    }


def metadata(*, source: str = "old-source", model: str = "measured-model") -> dict:
    return {
        "schema_version": 1, "run_id": "measured-run", "started_at": "2026-01-01T00:00:00Z",
        "code": {"git_commit": "a" * 40, "source_sha256": source, "dirty": False},
        "models": {"chat_model": model, "embedding_model": "measured-embedding"},
        "parameters": {"top_k": 8}, "reproducibility_issues": [],
    }


def verification(*, meta: dict | None = None) -> dict:
    records = [record("VERIFIED", "VERIFIED"), record("PARTIALLY_VERIFIED", "UNVERIFIED")]
    result = {"top_k": 8, "aggregate": score(records), "per_pair": {}, "claims": records}
    if meta is not None:
        result["provenance"] = meta
    return result


def test_score_exposes_majority_baseline_and_absent_labels() -> None:
    records = [record("VERIFIED", "VERIFIED") for _ in range(7)]
    records += [record("PARTIALLY_VERIFIED", "UNVERIFIED")]
    result = score(records)
    assert result["majority_class_baseline"] == {"label": "VERIFIED", "status_accuracy": 7 / 8}
    assert result["missing_gold_labels"] == ["CONTRADICTED", "UNVERIFIED"]
    assert result["per_class"]["UNVERIFIED"]["support"] == 0
    assert result["per_class"]["UNVERIFIED"]["recall"] is None
    assert result["per_class"]["PARTIALLY_VERIFIED"]["f1"] == 0
    assert result["macro_f1_present_labels"] == 0.5
    assert result["balanced_accuracy_present_labels"] == 0.5


def test_discrepancy_recall_counts_contradicted_gold() -> None:
    result = score([record("CONTRADICTED", "CONTRADICTED"), record("CONTRADICTED", "VERIFIED")])
    assert result["n_expected_discrepancy"] == 2
    assert result["discrepancy_recall"] == 0.5


def test_rendering_uses_saved_identity_without_reading_current_settings(tmp_path, monkeypatch) -> None:
    from claim2code_api.app import config

    def forbidden():
        raise AssertionError("rendering must not read current model settings")

    monkeypatch.setattr(config, "get_settings", forbidden)
    results = tmp_path / "results"
    results.mkdir()
    saved = verification(meta=metadata())
    (results / "verification.json").write_text(json.dumps(saved))
    monkeypatch.setattr(report, "RESULTS_DIR", results)
    combined, md = report.build()
    assert combined["verification"] == saved
    assert combined["provenance_warnings"] == []
    assert "measured-model" in md and "a" * 40 in md
    assert "git_commit" not in combined["environment"]
    assert "does not measure PDF extraction accuracy" in md


def test_legacy_outputs_stay_unknown_and_strict_report_refuses_them(tmp_path, monkeypatch) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "verification.json").write_text(json.dumps(verification()))
    header = {"git_commit": "old-header", "chat_model": "old-header-model"}
    (tmp_path / "results.json").write_text(json.dumps({"environment": header}))
    monkeypatch.setattr(report, "RESULTS_DIR", results)
    combined, md = report.build()
    assert combined["legacy_reported_environment"] == header
    assert "provenance" not in combined["verification"]
    assert "inference provenance unknown" in md
    assert "not a new measurement of this checkout" in md
    with pytest.raises(ValueError, match="no per-run measurement provenance"):
        report.build(require_provenance=True)


def test_comparison_reports_mixed_source_or_models() -> None:
    retrieval = {"provenance": metadata(source="source-a")}
    verification_data = {"provenance": metadata(source="source-b", model="different-model")}
    warnings = provenance.comparison_warnings({"retrieval": retrieval, "verification": verification_data})
    assert any("different source trees" in w for w in warnings)
    assert any("different models" in w for w in warnings)


def test_annotation_hash_ignores_machine_path_but_tracks_label_edits() -> None:
    annotation: dict = {
        "pair_id": "p", "_path": "/machine-a/p.json",
        "repo": {"commit": "a" * 40}, "paper": {"sha256": "b" * 64},
        "claims": [{"claim_id": "c", "expected_verification": "VERIFIED"}],
    }
    moved = copy.deepcopy(annotation)
    moved["_path"] = "/machine-b/p.json"
    assert provenance.annotation_snapshot([annotation]) == provenance.annotation_snapshot([moved])
    moved["claims"][0]["expected_verification"] = "UNVERIFIED"
    assert provenance.annotation_snapshot([annotation]) != provenance.annotation_snapshot([moved])


def test_unfrozen_inputs_have_an_explicit_reproducibility_gate() -> None:
    dataset = provenance.annotation_snapshot([{
        "pair_id": "p", "repo": {"commit": "main"}, "paper": {"sha256": "placeholder"},
    }])
    assert len(dataset["issues"]) == 2
    with pytest.raises(ValueError, match="full immutable commit"):
        provenance.require_frozen({"reproducibility_issues": dataset["issues"]})


def test_pinned_clone_mismatch_is_rejected_even_in_development(monkeypatch, tmp_path) -> None:
    db = SimpleNamespace(get=lambda *_args: SimpleNamespace(
        url="https://github.com/acme/repo", clone_path=str(tmp_path),
    ))
    monkeypatch.setattr(provenance, "_git", lambda _path, *args: "b" * 40 if args[0] == "rev-parse" else "")
    with pytest.raises(ValueError, match="pinned annotation revision"):
        provenance.repository_snapshot(db, [{
            "pair_id": "p", "repo": {"url": "https://github.com/acme/repo", "commit": "a" * 40},
        }], {"p": str(uuid4())})


def test_capture_records_only_allowlisted_nonsecret_settings(monkeypatch, tmp_path) -> None:
    from claim2code_api.app import config
    from claim2code_api.llm import providers

    settings = SimpleNamespace(
        openai_model_embed="embedding", openai_embed_dim=768,
        llm_request_timeout_seconds=60, embed_batch_size=128, embed_max_retries=3,
        openai_api_key="DO_NOT_STORE_THIS_KEY",
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(providers, "active", lambda _s: SimpleNamespace(
        preset_id="custom", model="local-model", api_key="DO_NOT_STORE_THIS_KEY",
    ))
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    monkeypatch.setattr(provenance, "_git", lambda *_args: "")
    meta = provenance.capture("verification", models=True)
    assert meta["models"]["chat_model"] == "local-model"
    assert meta["models"]["embedding_dimensions"] == 768
    assert meta["models"]["chat_model_revision"] is None
    assert "DO_NOT_STORE_THIS_KEY" not in json.dumps(meta)


def test_resume_identity_changes_with_model_code_labels_and_source_revision() -> None:
    original = metadata()
    for field, change in (
        ("models", {"chat_model": "new-model"}),
        ("code", {"source_sha256": "edited-code"}),
        ("dataset", {"annotations_sha256": "edited-labels"}),
        ("repositories", {"p": {"commit": "different-repo"}}),
    ):
        changed = copy.deepcopy(original)
        changed[field] = change
        assert provenance.measurement_identity(original) != provenance.measurement_identity(changed)
    timestamp_change = {**original, "started_at": "later", "run_id": "another-run"}
    assert provenance.measurement_identity(original) == provenance.measurement_identity(timestamp_change)


def test_nested_baseline_edits_change_identity_with_the_same_dirty_head(tmp_path, monkeypatch) -> None:
    baseline = tmp_path / "evaluation" / "baselines" / "hybrid_claim2code.py"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("def run():\n    return [('symbol', 0.25)]\n")
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    monkeypatch.setattr(
        provenance, "_git",
        lambda _path, *args: "a" * 40 if args[0] == "rev-parse" else " M evaluation/baselines/hybrid_claim2code.py",
    )

    before = provenance.capture("retrieval")
    baseline.write_text("def run():\n    return [('symbol', 0.75)]\n")
    after = provenance.capture("retrieval")

    assert before["code"]["git_commit"] == after["code"]["git_commit"]
    assert before["code"]["dirty"] is after["code"]["dirty"] is True
    assert before["code"]["source_sha256"] != after["code"]["source_sha256"]
    assert provenance.measurement_identity(before) != provenance.measurement_identity(after)


def test_resume_refuses_a_changed_model_without_modifying_records(tmp_path, monkeypatch) -> None:
    previous = metadata(model="original-model")
    current = metadata(model="new-model")
    annotations = [{"pair_id": "p", "claims": [{"claim_id": "c"}]}]
    db = SimpleNamespace()
    run_file = tmp_path / "verification_run.json"
    jsonl = tmp_path / "verification_claims.jsonl"
    run_file.write_text(json.dumps(previous))
    saved_records = json.dumps(record("VERIFIED", "VERIFIED")) + "\n"
    jsonl.write_text(saved_records)
    monkeypatch.setattr(evaluate_verification, "_load_annotations", lambda: annotations)
    monkeypatch.setattr(evaluate_verification, "_build_session_factory", lambda: lambda: nullcontext(db))
    monkeypatch.setattr(evaluate_verification, "resolve_repo_ids", lambda *_args: {"p": str(uuid4())})
    monkeypatch.setattr(evaluate_verification, "repository_snapshot", lambda *_args: {})
    monkeypatch.setattr(evaluate_verification, "capture", lambda *_args, **_kwargs: current)
    monkeypatch.setattr(evaluate_verification, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(evaluate_verification, "RUN_JSON", run_file)
    monkeypatch.setattr(evaluate_verification, "CLAIMS_JSONL", jsonl)
    assert evaluate_verification.main(["--resume"]) == 2
    assert jsonl.read_text() == saved_records
    assert json.loads(run_file.read_text()) == previous


def test_regrading_preserves_original_inference_and_scopes_labels_by_pair(tmp_path) -> None:
    (tmp_path / "m.py").write_text("x = 1\n")
    db = SimpleNamespace(get=lambda *_args: SimpleNamespace(clone_path=str(tmp_path)))
    original = {
        **record("PARTIALLY_VERIFIED", "VERIFIED"), "inference_run_id": "old-run",
        "evidence": [{"file_path": "m.py", "start_line": 1, "end_line": 1, "quote": "x = 1"}],
    }
    annotations = [
        {"pair_id": "p", "claims": [{"claim_id": "c", "expected_verification": "VERIFIED"}]},
        {"pair_id": "other", "claims": [{"claim_id": "c", "expected_verification": "CONTRADICTED"}]},
    ]
    [graded] = _regrade(lambda: nullcontext(db), {"p": str(uuid4())}, [original], annotations)
    assert graded["expected"] == "VERIFIED"
    assert graded["inference_expected"] == "PARTIALLY_VERIFIED"
    assert graded["inference_run_id"] == "old-run"
    assert original["expected"] == "PARTIALLY_VERIFIED"
    assert "inference_expected" not in original


def test_grading_does_not_read_outside_the_clone(tmp_path) -> None:
    clone = tmp_path / "clone"
    clone.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("private_test_data = 1\n")
    (clone / "linked.py").symlink_to(outside)
    assert _read_clone_file(clone, "../outside.py") is None
    assert _read_clone_file(clone, "linked.py") is None
