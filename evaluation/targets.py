#!/usr/bin/env python3
"""Check resume targets against complete, current measurements without running models."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "apps" / "api"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluation.evaluate_verification import score
from evaluation.provenance import annotation_snapshot, capture, comparison_warnings

MANIFEST = ROOT / "evaluation" / "suites" / "resume-targets.json"
TARGETS = {"pytorch_repositories": 12, "recall_at_5": 0.91,
           "status_accuracy": 0.87, "grounding_rate": 0.97}


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return row.get("pair_id", row.get("dataset", "")), row["claim_id"]


def _target(value: float | int | None, threshold: float | int, issues: list[str]) -> dict[str, Any]:
    return {"status": "unmeasured" if value is None or issues else
            ("met" if value >= threshold else "unmet"),
            "actual": value, "required": threshold, "issues": sorted(set(issues))}


def _regression(
    manifest: dict[str, Any], annotations: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    expected = {_identity(c): c for c in manifest["regression_claims"]}
    if len(expected) != len(manifest["regression_claims"]) or len(expected) != 79:
        raise ValueError("Regression manifest must contain the original 79 unique claim identities")
    actual = {(a["pair_id"], c["claim_id"]): c for a in annotations
              if a["pair_id"] in manifest["regression_pair_ids"] for c in a["claims"]}
    if set(actual) != set(expected):
        raise ValueError("The original regression claim identities have changed")
    for key, old in expected.items():
        now = actual[key]
        if now["expected_verification"] != old["expected"] or (
            hashlib.sha256(now["text"].encode()).hexdigest() != old["text_sha256"]
        ) or now["relevant_symbols"] != old["relevant_symbols"]:
            raise ValueError(f"Frozen regression text, label or relevant symbols changed: {key}")
    return expected


def _retrieval(
    data: dict[str, Any] | None, expected: dict[tuple[str, str], dict[str, Any]], baseline: str,
) -> tuple[float | None, float | None, list[str]]:
    if not data or data.get("schema_version") != 2 or data.get("metric_semantics") != "symbol_recall_v2":
        return None, None, ["A new symbol_recall_v2 measurement is required; legacy values are hit rates"]
    rows = [r for r in data.get("per_claim", []) if r.get("baseline") == baseline
            and r.get("dataset") in {key[0] for key in expected}]
    if len(rows) != len(expected) or {_identity(r) for r in rows} != set(expected):
        return None, None, [f"{baseline}: original regression retrieval records are incomplete or duplicated"]
    recalls: dict[str, list[float]] = defaultdict(list)
    hits: dict[str, list[float]] = defaultdict(list)
    issues = []
    for row in rows:
        groups = row.get("resolved_symbol_groups", [])
        retrieved = set(row.get("retrieved_symbol_ids", [])[:5])
        if len(groups) != len(expected[_identity(row)]["relevant_symbols"]) or (
            row.get("relevant_symbols") != expected[_identity(row)]["relevant_symbols"]
        ):
            issues.append(f"{baseline}: annotated target groups changed in {row['claim_id']}")
        if row.get("error") or not groups or any(not group for group in groups):
            issues.append(f"{baseline}: failed or unresolved retrieval target in {row['claim_id']}")
        recall = sum(bool(set(g) & retrieved) for g in groups) / len(groups) if groups else 0.0
        hit = float(any(set(g) & retrieved for g in groups))
        if not math.isclose(row.get("recall_at_5", -1), recall, abs_tol=1e-9):
            issues.append(f"{baseline}: stored recall disagrees with target groups in {row['claim_id']}")
        recalls[row["dataset"]].append(recall)
        hits[row["dataset"]].append(hit)
    summaries = [r for r in data.get("per_dataset", []) if r.get("baseline") == baseline
                 and r.get("dataset") in recalls]
    if len(summaries) != len(recalls) or {r["dataset"] for r in summaries} != set(recalls):
        issues.append(f"{baseline}: per-dataset summaries are incomplete or duplicated")
    for row in summaries:
        if row["baseline"] == baseline and row["dataset"] in recalls:
            if row.get("error") or row.get("n_claim_errors") or row.get("n_unresolved_symbols"):
                issues.append(f"{baseline}: errors or unresolved symbols in {row['dataset']}")
            if row.get("n_claims") != len(recalls[row["dataset"]]):
                issues.append(f"{baseline}: per-dataset claim count differs in {row['dataset']}")
    return (sum(sum(v) / len(v) for v in recalls.values()) / len(recalls),
            sum(sum(v) / len(v) for v in hits.values()) / len(hits), issues)


def assess(
    sections: dict[str, dict[str, Any] | None], manifest: dict[str, Any], *,
    coverage: dict[str, Any] | None = None, annotations_dir: Path | None = None,
    current_code: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep the original regression denominator separate from expanded coverage."""
    if manifest["targets"] != TARGETS:
        raise ValueError("The submitted resume targets cannot be lowered in the manifest")
    if any(len(manifest[key]) != len(set(manifest[key]))
           for key in ("pair_ids", "pytorch_pair_ids", "regression_pair_ids")):
        raise ValueError("Suite pair identities must be unique")
    directory = annotations_dir or ROOT / "evaluation" / "annotations"
    annotations = [json.loads((directory / f"{pid}.json").read_text()) for pid in manifest["pair_ids"]]
    expected = _regression(manifest, annotations)
    suite_claims = {(a["pair_id"], c["claim_id"]) for a in annotations for c in a["claims"]}
    snapshot = annotation_snapshot(annotations)
    current = current_code if current_code is not None else capture("targets")["code"]
    warnings = comparison_warnings(sections)
    # Curated claim-to-code measurements do not consume PDFs. Keep
    # freezing limitations visible without discarding valid development scores.
    advisories = [w for w in warnings if "paper SHA-256 is missing or a placeholder" in w
                  or "repository revision is not a full immutable commit SHA" in w]
    issues = [w for w in warnings if w not in advisories]
    for name in ("retrieval", "verification"):
        data = sections.get(name)
        meta = (data or {}).get("provenance", {})
        if not data:
            issues.append(f"{name}: measurement missing")
            continue
        if data.get("skipped_pairs"):
            issues.append(f"{name}: skipped pairs make the suite incomplete")
        if name == "verification":
            rows = data.get("claims", [])
        else:
            rows = [r for r in data.get("per_claim", []) if r.get("baseline") == "hybrid_claim2code"]
        if len(rows) != len(suite_claims) or {_identity(r) for r in rows} != suite_claims:
            issues.append(f"{name}: complete measurements for all suite claims are required")
        if any(r.get("error") for r in rows):
            issues.append(f"{name}: failed requests make the suite incomplete")
        if meta.get("code", {}).get("source_sha256") != current.get("source_sha256"):
            issues.append(f"{name}: measurement does not match the current source tree")
        if meta.get("dataset", {}).get("annotations_sha256") != snapshot["annotations_sha256"]:
            issues.append(f"{name}: measurement annotations differ from the current suite")
        for ann in annotations:
            repo = meta.get("repositories", {}).get(ann["pair_id"], {})
            actual, declared = str(repo.get("commit") or ""), str(ann["repo"].get("commit") or "")
            matches = (re.fullmatch(r"[0-9a-f]{40}", actual) and
                       (not re.fullmatch(r"[0-9a-f]{7,40}", declared) or actual.startswith(declared)))
            if not matches or repo.get("dirty") is not False:
                issues.append(f"{name}: indexed clone does not match clean pinned {ann['pair_id']}")
        if not meta.get("models", {}).get("chat_model") or not meta.get("models", {}).get("embedding_model"):
            issues.append(f"{name}: model identities are missing")
    rmeta = (sections.get("retrieval") or {}).get("provenance", {})
    vmeta = (sections.get("verification") or {}).get("provenance", {})
    if rmeta.get("settings") != vmeta.get("settings"):
        issues.append("Retrieval and verification have different settings provenance")

    verification = sections.get("verification") or {}
    records = [r for r in verification.get("claims", []) if r.get("pair_id") in manifest["regression_pair_ids"]]
    verification_issues = list(issues)
    valid = len(records) == len(expected) and {_identity(r) for r in records} == set(expected)
    if not valid:
        verification_issues.append("Original regression verification records are incomplete or duplicated")
    if any(r.get("expected") != expected.get(_identity(r), {}).get("expected") for r in records):
        verification_issues.append("A measured regression label differs from the frozen label")
    if any(r.get("error") for r in records):
        verification_issues.append("Verification contains failed model requests")
    metrics = score(records) if valid else {}
    recall, hit, retrieval_issues = _retrieval(sections.get("retrieval"), expected, "hybrid_claim2code")
    baseline, baseline_hit, baseline_issues = _retrieval(sections.get("retrieval"), expected, "embedding_only")

    proof = (coverage or {}).get("pairs", {})
    covered: list[str] = []
    embedded: list[str] = []
    missing: list[str] = []
    for pid in manifest["pytorch_pair_ids"]:
        row = proof.get(pid, {})
        ann = next(a for a in annotations if a["pair_id"] == pid)
        checked = (row.get("framework") == "pytorch" and row.get("source_checked") is True
                   and row.get("indexed") is True and row.get("n_symbols", 0) > 0
                   and row.get("commit") == ann["repo"]["commit"])
        (covered if checked else missing).append(pid)
        if checked and row.get("n_embeddings", 0) > 0:
            embedded.append(pid)
    coverage_issues = [] if coverage else ["Source/index coverage has not been measured"]
    if coverage and (coverage.get("provenance", {}).get("code", {}).get("source_sha256") != current.get("source_sha256")
                     or coverage.get("provenance", {}).get("dataset", {}).get("annotations_sha256") != snapshot["annotations_sha256"]):
        coverage_issues.append("Coverage proof does not match the current source and annotations")
    thresholds = manifest["targets"]
    requirements = {
        "pytorch_repositories": _target(len(covered) if coverage else None, thresholds["pytorch_repositories"], coverage_issues),
        "recall_at_5": _target(recall, thresholds["recall_at_5"], issues + retrieval_issues),
        "status_accuracy": _target(metrics.get("status_accuracy"), thresholds["status_accuracy"], verification_issues),
        "grounding_rate": _target(metrics.get("grounding_rate"), thresholds["grounding_rate"], verification_issues),
    }
    quantitative = all(r["status"] == "met" for r in requirements.values())
    baseline_status = "unmeasured" if baseline is None or issues or baseline_issues else (
        "met" if abs(baseline - manifest["reported_baseline"]["value"]) < 0.005
        and requirements["recall_at_5"]["status"] == "met" else "unmet")
    return {
        "schema_version": 1, "suite_id": manifest["suite_id"], "requirements": requirements,
        "quantitative_targets_met": quantitative,
        "all_resume_claims_supported": quantitative and baseline_status == "met",
        "regression": {"n_claims": len(expected), "metrics": metrics},
        "retrieval": {"standard_recall_at_5": recall, "hit_at_5": hit,
                      "embedding_only_standard_recall_at_5": baseline, "embedding_only_hit_at_5": baseline_hit},
        "reported_baseline_comparison": {"status": baseline_status, "claimed_baseline": 0.74,
            "actual_paired_standard_baseline": baseline,
            "note": manifest["reported_baseline"]["note"]},
        "coverage": {"source_checked_pairs": covered, "missing_pairs": missing,
                     "embedded_pairs": embedded, "model_verification_pairs": sorted(verification.get("per_pair", {}))},
        "provenance_issues": sorted(set(issues)),
        "reproducibility_advisories": sorted(set(advisories)),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Resume target check", "", "| Requirement | Status | Actual | Required |",
             "|---|---|---|---|"]
    for name, row in report["requirements"].items():
        value = row["actual"]
        actual = "unmeasured" if value is None else (str(value) if name == "pytorch_repositories" else f"{value:.1%}")
        required = str(row["required"]) if name == "pytorch_repositories" else f"{row['required']:.0%}"
        lines.append(f"| {name} | {row['status']} | {actual} | {required} |")
    lines += ["", "Accuracy and retrieval use the unchanged original 79-claim regression set.",
              "New positive coverage claims do not change that denominator.", "",
              "Baseline comparison: " + report["reported_baseline_comparison"]["status"] + ". "
              + report["reported_baseline_comparison"]["note"], "",
              f"All submitted resume claims supported: {report['all_resume_claims_supported']}."]
    warnings = sorted({message for row in report["requirements"].values() for message in row["issues"]})
    if warnings:
        lines += ["", "Measurement issues:", ""] + [f"- {issue}" for issue in warnings]
    if report["reproducibility_advisories"]:
        lines += ["", "Input-freezing advisories (development metrics remain in scope):", ""]
        lines += [f"- {issue}" for issue in report["reproducibility_advisories"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "evaluation" / "results")
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--annotations-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        sections = {name: json.loads(path.read_text()) if path.exists() else None
                    for name in ("retrieval", "verification")
                    for path in [args.results_dir / f"{name}.json"]}
        manifest = json.loads(args.manifest.read_text())
        coverage = json.loads(args.coverage.read_text()) if args.coverage and args.coverage.exists() else None
        report = assess(sections, manifest, coverage=coverage, annotations_dir=args.annotations_dir)
        directory = args.output_dir or args.results_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "targets.json").write_text(json.dumps(report, indent=2) + "\n")
        (directory / "targets.md").write_text(markdown(report))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Invalid target measurement: {exc}", file=sys.stderr)
        return 2
    print(markdown(report))
    return 0 if report["quantitative_targets_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
