#!/usr/bin/env python3
"""Safety evaluator — secret redaction, injection detection, false positives.

Run with ``make eval-safety`` from the repo root. Each case in
``evaluation/datasets/safety_corpus.json`` is wrapped as an untrusted
user message and passed through :class:`SafetyGate.wrap_messages`, the
same call every LLM request goes through. No LLM or database is needed.

Metrics (each split into ``covered`` — the forms the Phase 8 patterns
target — and ``held_out`` — common real-world forms written without
reference to the patterns):

  - **secret_protection_rate** — a secret is protected when no
    12-character window of it survives in the gated message, so a
    partial redaction that leaks the tail still counts as a leak.
  - **injection_detection_rate** — the gate records at least one
    injection violation.
  - **false_positive_rate** — clean paper/code text that triggers any
    violation, or whose text the gate changes.

Writes ``evaluation/results/safety.json``; ``evaluation/report.py``
renders it.
"""
# Imports below bootstrap the repository-root CLI entry point.
# ruff: noqa: E402
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "apps" / "api"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluation.provenance import capture, finish

logger = logging.getLogger("claim2code.eval.safety")

EVAL_DIR = _THIS.parent
CORPUS = EVAL_DIR / "datasets" / "safety_corpus.json"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "safety.json"

_LEAK_WINDOW = 12


def leaks(secret: str, text: str) -> bool:
    """True when any ``_LEAK_WINDOW``-char window of ``secret`` is in ``text``."""
    if len(secret) <= _LEAK_WINDOW:
        return secret in text
    return any(
        secret[i:i + _LEAK_WINDOW] in text
        for i in range(len(secret) - _LEAK_WINDOW + 1)
    )


def run_case(gate: Any, case: dict[str, Any]) -> dict[str, Any]:
    """Gate one case and record what happened to it."""
    result = gate.wrap_messages(
        [{"role": "user", "content": case["text"]}], kind="eval_safety",
    )
    out_text = result.messages[0]["content"]
    kinds = sorted({v.kind for v in result.violations})
    labels = sorted({v.pattern_label for v in result.violations})
    record = {
        "id": case["id"],
        "category": case["category"],
        "subset": case["subset"],
        "violation_kinds": kinds,
        "patterns": labels,
    }
    if case["category"] == "secret":
        record["passed"] = not leaks(case["secret"], out_text)
    elif case["category"] == "injection":
        record["passed"] = "injection" in kinds
    else:
        record["passed"] = not kinds and out_text == case["text"]
    return record


def score(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pass rates per (category, subset)."""
    def rate(category: str, subset: str | None = None) -> dict[str, Any]:
        rs = [
            r for r in records
            if r["category"] == category and (subset is None or r["subset"] == subset)
        ]
        passed = sum(r["passed"] for r in rs)
        return {"n": len(rs), "passed": passed, "rate": passed / len(rs) if rs else None}

    clean = rate("clean")
    return {
        "secret_protection": {
            "covered": rate("secret", "covered"),
            "held_out": rate("secret", "held_out"),
            "all": rate("secret"),
        },
        "injection_detection": {
            "covered": rate("injection", "covered"),
            "held_out": rate("injection", "held_out"),
            "all": rate("injection"),
        },
        "false_positive": {
            "n": clean["n"],
            "flagged": clean["n"] - clean["passed"],
            "rate": (clean["n"] - clean["passed"]) / clean["n"] if clean["n"] else None,
        },
        "failures": [r["id"] for r in records if not r["passed"]],
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from claim2code_api.safety.gate import SafetyGate

    corpus = json.loads(CORPUS.read_text())
    meta = capture("safety", corpus=corpus, parameters={"leak_window": _LEAK_WINDOW})
    cases = corpus["cases"]
    gate = SafetyGate()
    records = [run_case(gate, c) for c in cases]
    out: dict[str, Any] = {"metrics": score(records), "cases": records, "provenance": finish(meta)}
    RESULTS_DIR.mkdir(exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(out, indent=2))
    m = out["metrics"]
    logger.info(
        "secrets %d/%d  injections %d/%d  false positives %d/%d",
        m["secret_protection"]["all"]["passed"], m["secret_protection"]["all"]["n"],
        m["injection_detection"]["all"]["passed"], m["injection_detection"]["all"]["n"],
        m["false_positive"]["flagged"], m["false_positive"]["n"],
    )
    logger.info("wrote %s", RESULTS_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
