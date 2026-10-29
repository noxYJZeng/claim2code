"""Target gates preserve the submitted thresholds and the original regression set."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.provenance import annotation_snapshot  # noqa: E402
from evaluation.targets import MANIFEST, assess, main  # noqa: E402


@pytest.fixture
def measurement(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    directory = tmp_path / "annotations"
    directory.mkdir()
    annotations = []
    for pid in manifest["pair_ids"]:
        ann = json.loads((ROOT / "evaluation" / "annotations" / f"{pid}.json").read_text())
        ann["repo"]["commit"] = "a" * 40
        ann["paper"]["sha256"] = "b" * 64
        (directory / f"{pid}.json").write_text(json.dumps(ann))
        annotations.append(ann)
    meta = {"run_id": "test-run", "code": {"source_sha256": "current"},
            "dataset": annotation_snapshot(annotations), "reproducibility_issues": [],
            "models": {"chat_model": "test-chat", "embedding_model": "test-embed"},
            "settings": {}, "repositories": {pid: {"commit": "a" * 40, "dirty": False}
                                               for pid in manifest["pair_ids"]}}
    verification_rows, retrieval_rows, summaries = [], [], []
    for ann in annotations:
        pid = ann["pair_id"]
        for claim in ann["claims"]:
            verification_rows.append({"pair_id": pid, "claim_id": claim["claim_id"],
                "expected": claim["expected_verification"], "predicted": claim["expected_verification"],
                "n_evidence": 1, "n_grounded": 1, "n_line_accurate": 1,
                "evidence_hit": True, "error": None})
            refs = claim["relevant_symbols"]
            groups = [[f"{claim['claim_id']}-{i}"] for i in range(len(refs))]
            for baseline in ("hybrid_claim2code", "embedding_only"):
                retrieval_rows.append({"dataset": pid, "baseline": baseline,
                    "claim_id": claim["claim_id"], "relevant_symbols": refs,
                    "resolved_symbol_groups": groups,
                    "retrieved_symbol_ids": [g[0] for g in groups],
                    "recall_at_5": 1.0, "hit_at_5": 1.0, "error": None})
        for baseline in ("hybrid_claim2code", "embedding_only"):
            summaries.append({"dataset": pid, "baseline": baseline, "n_claims": len(ann["claims"]),
                "n_unresolved_symbols": 0, "n_claim_errors": 0, "error": None})
    sections = {
        "retrieval": {"schema_version": 2, "metric_semantics": "symbol_recall_v2",
            "per_claim": retrieval_rows, "per_dataset": summaries, "provenance": copy.deepcopy(meta)},
        "verification": {"claims": verification_rows, "provenance": copy.deepcopy(meta)},
    }
    coverage = {"schema_version": 1, "provenance": copy.deepcopy(meta),
        "pairs": {pid: {"framework": "pytorch", "source_checked": True,
            "indexed": True, "n_symbols": 20, "n_embeddings": 20, "commit": "a" * 40}
            for pid in manifest["pytorch_pair_ids"]}}
    return {"sections": sections, "manifest": manifest, "coverage": coverage,
            "annotations_dir": directory, "current_code": {"source_sha256": "current"}}


def test_targets_recompute_records_and_keep_baseline_claim_separate(measurement):
    report = assess(**measurement)
    assert report["quantitative_targets_met"] is True
    assert report["regression"]["n_claims"] == 79
    assert report["all_resume_claims_supported"] is False  # measured baseline 100%, not claimed 74%
    assert report["reported_baseline_comparison"]["status"] == "unmet"


def test_added_positive_claims_cannot_dilute_regression_errors(measurement):
    rows = measurement["sections"]["verification"]["claims"]
    regression = set(measurement["manifest"]["regression_pair_ids"])
    original = [r for r in rows if r["pair_id"] in regression]
    for row in original[:14]:
        row["predicted"] = "UNVERIFIED"
    assert sum(r["expected"] == r["predicted"] for r in rows) / len(rows) >= 0.87
    report = assess(**measurement)
    assert report["requirements"]["status_accuracy"]["actual"] == pytest.approx(65 / 79)
    assert report["requirements"]["status_accuracy"]["status"] == "unmet"


@pytest.mark.parametrize("change", ["missing", "duplicate", "changed_label"])
def test_incomplete_or_changed_regression_cannot_pass(measurement, change):
    rows = measurement["sections"]["verification"]["claims"]
    if change == "missing":
        rows.pop(0)
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows[0]["expected"] = "UNVERIFIED"
    assert assess(**measurement)["requirements"]["status_accuracy"]["status"] == "unmeasured"


@pytest.mark.parametrize("field,value", [("text", "replacement claim"), ("expected_verification", "UNVERIFIED"),
                                        ("relevant_symbols", [])])
def test_frozen_original_claim_content_cannot_change(measurement, field, value):
    pid = measurement["manifest"]["regression_pair_ids"][0]
    path = measurement["annotations_dir"] / f"{pid}.json"
    data = json.loads(path.read_text())
    data["claims"][0][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Frozen regression"):
        assess(**measurement)


def test_legacy_hit_metric_cannot_satisfy_standard_recall(measurement):
    measurement["sections"]["retrieval"]["schema_version"] = 1
    report = assess(**measurement)
    assert report["requirements"]["recall_at_5"]["status"] == "unmeasured"
    assert report["retrieval"]["standard_recall_at_5"] is None


def test_recall_is_recomputed_from_all_target_groups(measurement):
    rows = measurement["sections"]["retrieval"]["per_claim"]
    row = next(r for r in rows if r["dataset"] in measurement["manifest"]["regression_pair_ids"]
               and r["baseline"] == "hybrid_claim2code" and len(r["resolved_symbol_groups"]) > 1)
    row["retrieved_symbol_ids"] = row["retrieved_symbol_ids"][:1]
    report = assess(**measurement)
    assert report["retrieval"]["hit_at_5"] == 1.0
    assert report["retrieval"]["standard_recall_at_5"] < 1.0
    assert report["requirements"]["recall_at_5"]["status"] == "unmeasured"  # stored 1.0 is inconsistent


@pytest.mark.parametrize("broken", ["source", "model", "annotations", "skipped", "request_error"])
def test_mixed_or_incomplete_measurements_cannot_pass(measurement, broken):
    data = measurement["sections"]["verification"]
    if broken == "source":
        data["provenance"]["code"]["source_sha256"] = "old"
    elif broken == "model":
        data["provenance"]["models"]["chat_model"] = "different"
    elif broken == "annotations":
        data["provenance"]["dataset"]["annotations_sha256"] = "different"
    elif broken == "skipped":
        data["skipped_pairs"] = ["lora"]
    else:
        data["claims"][0]["error"] = "model unavailable"
    assert assess(**measurement)["requirements"]["status_accuracy"]["status"] == "unmeasured"


def test_missing_pdf_hash_is_advisory_for_curated_claim_metrics(measurement):
    for section in measurement["sections"].values():
        section["provenance"]["reproducibility_issues"] = ["lora: paper SHA-256 is missing or a placeholder"]
    report = assess(**measurement)
    assert report["requirements"]["status_accuracy"]["status"] == "met"
    assert report["reproducibility_advisories"]


def test_failed_new_coverage_claim_does_not_establish_complete_suite(measurement):
    rows = measurement["sections"]["verification"]["claims"]
    row = next(r for r in rows if r["pair_id"] not in measurement["manifest"]["regression_pair_ids"])
    row["error"] = "model failed on a newly covered repository"
    report = assess(**measurement)
    assert report["regression"]["metrics"]["status_accuracy"] == 1.0
    assert report["requirements"]["status_accuracy"]["status"] == "unmeasured"


def test_metadata_alone_cannot_establish_twelve_repo_coverage(measurement):
    measurement["coverage"] = None
    report = assess(**measurement)
    assert report["requirements"]["pytorch_repositories"]["status"] == "unmeasured"
    assert report["coverage"]["source_checked_pairs"] == []


def test_partial_or_stale_source_coverage_cannot_pass(measurement):
    pid = measurement["manifest"]["pytorch_pair_ids"][0]
    measurement["coverage"]["pairs"][pid]["source_checked"] = False
    assert assess(**measurement)["requirements"]["pytorch_repositories"]["status"] == "unmet"
    measurement["coverage"]["provenance"]["code"]["source_sha256"] = "old"
    assert assess(**measurement)["requirements"]["pytorch_repositories"]["status"] == "unmeasured"


def test_thresholds_cannot_be_lowered(measurement):
    measurement["manifest"]["targets"]["status_accuracy"] = 0.1
    with pytest.raises(ValueError, match="cannot be lowered"):
        assess(**measurement)


def test_cli_missing_coverage_is_unmeasured_and_still_writes_report(tmp_path):
    results, output = tmp_path / "results", tmp_path / "output"
    results.mkdir()
    assert main(["--results-dir", str(results), "--coverage", str(tmp_path / "absent.json"),
                 "--output-dir", str(output)]) == 1
    report = json.loads((output / "targets.json").read_text())
    assert report["requirements"]["pytorch_repositories"]["status"] == "unmeasured"
    assert (output / "targets.md").exists()
