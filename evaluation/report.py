#!/usr/bin/env python3
"""Fold the per-evaluator results into ``results.json`` + ``report.md``.

Reads whichever of ``evaluation/results/{retrieval,verification,safety}.json``
exist (an evaluator that has not run yet is reported as missing) and
writes:

  - ``evaluation/results.json`` — every evaluator's output under one
    object, plus the git commit and model names it was measured with.
  - ``evaluation/report.md`` — the human-readable report.
"""
# Imports below bootstrap the repository-root CLI entry point.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "apps" / "api"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluation.evaluate_retrieval import (
    BASELINES,
    BaselineMetrics,
    _render_report_md,
)
from evaluation.evaluate_verification import classification_metrics
from evaluation.provenance import comparison_warnings

EVAL_DIR = _THIS.parent
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_JSON = EVAL_DIR / "results.json"
REPORT_MD = EVAL_DIR / "report.md"
SECTIONS = ("retrieval", "verification", "safety")


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _frac(block: dict[str, Any]) -> str:
    return f"{block['passed']}/{block['n']} ({_pct(block['rate'])})"


def render_retrieval(data: dict[str, Any]) -> list[str]:
    metrics = [
        BaselineMetrics(
            baseline=row["baseline"],
            dataset=row["dataset"],
            n_claims=row["n_claims"],
            recall_at_1=row["recall_at_1"],
            recall_at_5=row["recall_at_5"],
            mrr=row["mrr"],
            error=row.get("error"),
        )
        for row in data["per_dataset"]
    ]
    body = _render_report_md(metrics).splitlines()
    # Demote the evaluator's own headings one level under "## Retrieval".
    lines = ["## Retrieval"]
    lines += [("#" + ln) if ln.startswith("## ") else ln for ln in body[1:]]
    names = [b.NAME for b in BASELINES]
    lines += [
        "",
        "Aggregates are macro averages over datasets. Baselines: " + ", ".join(names) + ".",
    ]
    return lines + [""]


def render_verification(data: dict[str, Any]) -> list[str]:
    agg = {**data["aggregate"], **classification_metrics(data["aggregate"]["confusion"])}
    majority = agg["majority_class_baseline"]
    lines = [
        "## Verification",
        "",
        (f"{agg['n_claims']} curated annotated claims, claim-to-code pipeline (hybrid retrieval top_k="
        f"{data['top_k']} → deterministic checkers → LLM verdict)."),
        "",
        ("The input is an annotation's claim text, also used as its paper quote. This does not "
         "measure PDF extraction accuracy or verify the paper citation against a PDF."),
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Status accuracy (exact, on observed labels) | {_pct(agg['status_accuracy'])} |",
        (f"| Majority-class baseline (always {majority['label']}) "
         f"| {_pct(majority['status_accuracy'])} |"),
        f"| Macro F1 (gold-present labels only) | {_pct(agg['macro_f1_present_labels'])} |",
        (f"| Balanced accuracy (gold-present labels only) "
         f"| {_pct(agg['balanced_accuracy_present_labels'])} |"),
        f"| Support rate (VERIFIED or PARTIALLY_VERIFIED) | {_pct(agg['support_rate'])} |",
        (f"| Discrepancy precision | {_pct(agg['discrepancy_precision'])} "
        f"({agg['n_flagged']} flagged) |"),
        (f"| Discrepancy recall | {_pct(agg['discrepancy_recall'])} "
        f"({agg['n_expected_discrepancy']} annotated) |"),
        (f"| Grounding rate (cited code exists in the cited file) | {_pct(agg['grounding_rate'])} "
        f"({agg['n_code_evidence']} citations) |"),
        f"| Line accuracy (cited code at the cited lines) | {_pct(agg.get('line_accuracy'))} |",
        (f"| Evidence hit rate (supported verdicts citing an annotated symbol) "
        f"| {_pct(agg['evidence_hit_rate'])} |"),
        f"| LLM errors | {agg['n_llm_errors']} |",
        "",
        "### Per pair",
        "",
        "| Pair | Claims | Accuracy | Support | Disc. P | Disc. R | Grounding |",
        "|---|---|---|---|---|---|---|",
    ]
    for pid, m in data["per_pair"].items():
        lines.append(
            f"| {pid} | {m['n_claims']} | {_pct(m['status_accuracy'])} | {_pct(m['support_rate'])} "
            f"| {_pct(m['discrepancy_precision'])} | {_pct(m['discrepancy_recall'])} "
            f"| {_pct(m['grounding_rate'])} |"
        )
    statuses = ("VERIFIED", "PARTIALLY_VERIFIED", "CONTRADICTED", "UNVERIFIED")
    lines += [
        "", "### Label coverage", "",
        "| Gold label | Count | Precision | Recall | F1 |", "|---|---|---|---|---|",
    ]
    for label, metrics in agg["per_class"].items():
        lines.append(
            f"| {label} | {metrics['support']} | {_pct(metrics['precision'])} "
            f"| {_pct(metrics['recall'])} | {_pct(metrics['f1'])} |"
        )
    if agg["missing_gold_labels"]:
        lines += ["", "No gold examples for: " + ", ".join(agg["missing_gold_labels"])
                  + ". These results do not establish four-way classification performance."]
    lines += [
        "", "Code grounding measures whether a quote exists, not whether the verdict is correct.",
    ]
    lines += [
        "",
        "### Confusion (rows = annotated, columns = predicted)",
        "",
        "| | " + " | ".join(statuses) + " |",
        "|---|" + "---|" * len(statuses),
    ]
    for exp, row in agg["confusion"].items():
        lines.append(f"| {exp} | " + " | ".join(str(row[s]) for s in statuses) + " |")
    if data.get("skipped_pairs"):
        lines += ["", "Skipped (repository not indexed): " + ", ".join(data["skipped_pairs"]) + "."]
    return lines + [""]


def render_safety(data: dict[str, Any]) -> list[str]:
    m = data["metrics"]
    sp, inj, fp = m["secret_protection"], m["injection_detection"], m["false_positive"]
    lines = [
        "## Safety",
        "",
        ("`covered` cases use the forms the Phase 8 patterns target; `held_out` cases are "
        "common real-world forms written without reference to them."),
        "",
        "| Check | Covered | Held-out | All |",
        "|---|---|---|---|",
        f"| Secrets protected | {_frac(sp['covered'])} | {_frac(sp['held_out'])} | {_frac(sp['all'])} |",
        (f"| Injections detected | {_frac(inj['covered'])} | {_frac(inj['held_out'])} "
        f"| {_frac(inj['all'])} |"),
        (f"| Clean text flagged (false positive) | | | {fp['flagged']}/{fp['n']} "
        f"({_pct(fp['rate'])}) |"),
        "",
    ]
    failures = [c for c in data["cases"] if not c["passed"]]
    if failures:
        lines += ["| Failed case | Category | Subset | Patterns hit |", "|---|---|---|---|"]
        for c in failures:
            lines.append(
                f"| {c['id']} | {c['category']} | {c['subset']} | {', '.join(c['patterns']) or '—'} |"
            )
        lines.append("")
    return lines


RENDERERS = {
    "retrieval": render_retrieval,
    "verification": render_verification,
    "safety": render_safety,
}


def build(*, require_provenance: bool = False) -> tuple[dict[str, Any], str]:
    """Load the per-evaluator results and render the combined outputs."""
    env = {"report_generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
    combined: dict[str, Any] = {"environment": env}
    lines = [
        "# Claim2Code — Evaluation Report",
        "",
        f"Report rendered {env['report_generated_at']}. Rendering does not run the models.",
        "Measurement code and model identities below come only from stored evaluator provenance.",
        "",
    ]
    # Preserve the old header as an unverified historical declaration,
    # never upgrade it into missing per-evaluator inference provenance.
    previous = RESULTS_DIR.parent / "results.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        legacy = old.get("legacy_reported_environment")
        if legacy is None and old.get("environment", {}).get("git_commit"):
            legacy = old["environment"]
        if legacy:
            combined["legacy_reported_environment"] = legacy
            lines += [
                ("Historical combined-report header (not verified per evaluator): "
                f"commit `{legacy.get('git_commit', 'unknown')}`, "
                f"chat `{legacy.get('chat_model', 'unknown')}`, "
                f"embeddings `{legacy.get('embedding_model', 'unknown')}`."),
                "These archived numbers predate subsequent fixes and are not a new measurement of this checkout.",
                "",
            ]
    for name in SECTIONS:
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            combined[name] = None
            lines += [f"## {name.capitalize()}", "", f"Not run (`{path.name}` missing).", ""]
            continue
        data = json.loads(path.read_text())
        combined[name] = data
        meta = data.get("provenance")
        if meta:
            lines += [
                (f"Measurement `{name}`: run `{meta.get('run_id', 'unknown')}`, "
                f"code `{meta.get('code', {}).get('git_commit', 'unknown')}`, "
                f"started `{meta.get('started_at', 'unknown')}`."),
                "",
            ]
            if meta.get("models"):
                model = meta["models"]
                lines += [(f"Recorded chat `{model.get('chat_model', 'unknown')}`; "
                           f"embeddings `{model.get('embedding_model', 'unknown')}`."), ""]
        else:
            lines += [f"Measurement `{name}`: legacy artifact; inference provenance unknown.", ""]
        lines += RENDERERS[name](data)
    warnings = comparison_warnings({name: combined[name] for name in SECTIONS})
    combined["provenance_warnings"] = warnings
    if require_provenance and warnings:
        raise ValueError("Cannot render a comparison with complete source/input provenance: "
                         + "; ".join(warnings))
    if warnings:
        lines += ["## Provenance and reproducibility warnings", ""]
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    return combined, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--require-provenance", action="store_true",
                        help="require compatible recorded source/input metadata; model weights are not verified")
    args = parser.parse_args(argv)
    try:
        combined, md = build(require_provenance=args.require_provenance)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    RESULTS_JSON.write_text(json.dumps(combined, indent=2))
    REPORT_MD.write_text(md)
    print(f"wrote {RESULTS_JSON} and {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
