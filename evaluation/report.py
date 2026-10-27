#!/usr/bin/env python3
"""Fold the per-evaluator results into ``results.json`` + ``report.md``.

Reads whichever of ``evaluation/results/{retrieval,verification,safety}.json``
exist (an evaluator that has not run yet is reported as missing) and
writes:

  - ``evaluation/results.json`` — every evaluator's output under one
    object, plus the git commit and model names it was measured with.
  - ``evaluation/report.md`` — the human-readable report.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
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

EVAL_DIR = _THIS.parent
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_JSON = EVAL_DIR / "results.json"
REPORT_MD = EVAL_DIR / "report.md"
SECTIONS = ("retrieval", "verification", "safety")


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _frac(block: dict[str, Any]) -> str:
    return f"{block['passed']}/{block['n']} ({_pct(block['rate'])})"


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {"generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        env["git_commit"] = None
    try:
        from claim2code_api.app.config import get_settings

        s = get_settings()
        env["chat_model"] = s.openai_model_chat
        env["embedding_model"] = s.openai_model_embed
    except Exception as exc:  # noqa: BLE001 — settings are informative only
        env["settings_error"] = str(exc)
    return env


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
    agg = data["aggregate"]
    lines = [
        "## Verification",
        "",
        (f"{agg['n_claims']} annotated claims, production pipeline (hybrid retrieval top_k="
        f"{data['top_k']} → deterministic checkers → LLM verdict)."),
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Status accuracy (exact, 4-way) | {_pct(agg['status_accuracy'])} |",
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


def build() -> tuple[dict[str, Any], str]:
    """Load the per-evaluator results and render the combined outputs."""
    env = _environment()
    combined: dict[str, Any] = {"environment": env}
    lines = [
        "# Claim2Code — Evaluation Report",
        "",
        f"Generated {env['generated_at']} at commit `{env.get('git_commit')}`"
        + (
            f" with chat model `{env['chat_model']}` and embeddings `{env['embedding_model']}`."
            if "chat_model" in env else "."
        ),
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
        lines += RENDERERS[name](data)
    return combined, "\n".join(lines)


def main() -> int:
    combined, md = build()
    RESULTS_JSON.write_text(json.dumps(combined, indent=2))
    REPORT_MD.write_text(md)
    print(f"wrote {RESULTS_JSON} and {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
