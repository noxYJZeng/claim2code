"""Unit tests for the verification + safety evaluators and the report.

Only the pure parts are covered here: scoring, citation grounding,
leak detection, the safety corpus run (gate only, no LLM) and report
rendering. The pipeline run itself needs a live DB + LLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(_REPO_ROOT), str(_REPO_ROOT / "apps" / "api")):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation import evaluate_safety, report  # noqa: E402
from evaluation.evaluate_verification import (  # noqa: E402
    _claim_from_annotation,
    quote_is_grounded,
    score,
)


def _record(expected: str, predicted: str, *, n_evidence: int = 1, n_grounded: int = 1,
            hit: bool = True, error: str | None = None) -> dict:
    return {
        "pair_id": "p", "claim_id": "c", "expected": expected, "predicted": predicted,
        "n_evidence": n_evidence, "n_grounded": n_grounded, "evidence_hit": hit, "error": error,
    }


# ============================================================
# Verification scoring
# ============================================================


def test_score_counts_accuracy_support_and_discrepancy_detection() -> None:
    records = [
        _record("VERIFIED", "VERIFIED"),
        _record("VERIFIED", "PARTIALLY_VERIFIED"),  # false discrepancy
        _record("PARTIALLY_VERIFIED", "CONTRADICTED"),  # caught (severity overcalled)
        _record("PARTIALLY_VERIFIED", "VERIFIED"),  # missed
        _record("VERIFIED", "UNVERIFIED", n_evidence=0, n_grounded=0, hit=False),
    ]
    m = score(records)
    assert m["n_claims"] == 5
    assert m["status_accuracy"] == pytest.approx(1 / 5)
    assert m["support_rate"] == pytest.approx(3 / 5)
    assert m["discrepancy_precision"] == pytest.approx(1 / 2)
    assert m["discrepancy_recall"] == pytest.approx(1 / 2)
    assert m["confusion"]["PARTIALLY_VERIFIED"]["CONTRADICTED"] == 1
    assert "CONTRADICTED" not in m["confusion"]  # no annotated row → dropped


def test_score_grounding_and_evidence_hit_use_the_right_denominators() -> None:
    records = [
        {**_record("VERIFIED", "VERIFIED", n_evidence=3, n_grounded=2, hit=True), "n_line_accurate": 1},
        _record("VERIFIED", "PARTIALLY_VERIFIED", n_evidence=1, n_grounded=1, hit=False),
        _record("VERIFIED", "UNVERIFIED", n_evidence=0, n_grounded=0, hit=False, error="boom"),
    ]
    m = score(records)
    assert m["grounding_rate"] == pytest.approx(3 / 4)
    assert m["line_accuracy"] == pytest.approx(1 / 4)
    assert m["evidence_hit_rate"] == pytest.approx(1 / 2)  # over supported verdicts only
    assert m["n_llm_errors"] == 1


def test_score_reports_none_when_a_rate_has_no_denominator() -> None:
    m = score([_record("VERIFIED", "UNVERIFIED", n_evidence=0, n_grounded=0, hit=False)])
    assert m["discrepancy_precision"] is None
    assert m["discrepancy_recall"] is None
    assert m["grounding_rate"] is None
    assert m["evidence_hit_rate"] is None
    assert score([]) == {"n_claims": 0}


def test_quote_is_grounded_tolerates_whitespace_and_line_slack() -> None:
    text = "def f(x):\n    y = x  *  2\n    return y\n" + "\n" * 10 + "z = 1\n"
    assert quote_is_grounded("y = x * 2", text, 2, 2)
    assert quote_is_grounded("return y", text, 1, 1)  # off by two lines
    assert not quote_is_grounded("return z", text, 1, 3)
    assert not quote_is_grounded("   ", text, 1, 3)
    assert not quote_is_grounded("y", None, 1, 1)
    # Real code, wrong lines: grounded in the file, not at the citation.
    assert quote_is_grounded("z = 1", text)
    assert not quote_is_grounded("z = 1", text, 1, 3)


def test_claim_from_annotation_builds_a_valid_claim() -> None:
    claim = _claim_from_annotation({
        "claim_id": "x_c1",
        "text": "The loss is MSE.",
        "evidence_span": {"section": "3", "page": 4, "char_start": 0, "char_end": 0},
    })
    assert claim.claim_id == "x_c1"
    assert claim.paper_evidence.quote == "The loss is MSE."
    assert claim.paper_evidence.char_end > claim.paper_evidence.char_start


# ============================================================
# Safety evaluator
# ============================================================


def test_leaks_catches_partial_redaction() -> None:
    secret = "sk-ant-api03-AbCdEfGhIjKlMnOp_QrStUvWxYz0123456789"
    assert evaluate_safety.leaks(secret, f"key={secret}")
    # Prefix redacted but the tail survives → still a leak.
    assert evaluate_safety.leaks(secret, "[REDACTED:anthropic_key]_QrStUvWxYz0123456789")
    assert not evaluate_safety.leaks(secret, "[REDACTED:anthropic_key]")
    assert evaluate_safety.leaks("short", "a short one")


def test_safety_corpus_covered_cases_all_pass() -> None:
    """The forms the Phase 8 patterns target must stay fully handled."""
    from claim2code_api.safety.gate import SafetyGate

    cases = json.loads(evaluate_safety.CORPUS.read_text())["cases"]
    assert {c["category"] for c in cases} == {"secret", "injection", "clean"}
    gate = SafetyGate()
    covered = [evaluate_safety.run_case(gate, c) for c in cases if c["subset"] == "covered"]
    assert covered and all(r["passed"] for r in covered), [r["id"] for r in covered if not r["passed"]]


def test_safety_score_splits_subsets() -> None:
    records = [
        {"id": "s1", "category": "secret", "subset": "covered", "passed": True},
        {"id": "s2", "category": "secret", "subset": "held_out", "passed": False},
        {"id": "i1", "category": "injection", "subset": "covered", "passed": True},
        {"id": "c1", "category": "clean", "subset": "clean", "passed": False},
        {"id": "c2", "category": "clean", "subset": "clean", "passed": True},
    ]
    m = evaluate_safety.score(records)
    assert m["secret_protection"]["covered"]["rate"] == 1.0
    assert m["secret_protection"]["held_out"]["rate"] == 0.0
    assert m["injection_detection"]["held_out"]["rate"] is None
    assert m["false_positive"] == {"n": 2, "flagged": 1, "rate": 0.5}
    assert m["failures"] == ["s2", "c1"]


# ============================================================
# Report
# ============================================================


def test_report_renders_available_sections_and_marks_missing(tmp_path, monkeypatch) -> None:
    results = tmp_path / "results"
    results.mkdir()
    agg = score([_record("VERIFIED", "VERIFIED"), _record("PARTIALLY_VERIFIED", "CONTRADICTED")])
    (results / "verification.json").write_text(json.dumps({
        "top_k": 8, "aggregate": agg, "per_pair": {"p": agg}, "skipped_pairs": ["bert"],
        "claims": [],
    }))
    (results / "safety.json").write_text(json.dumps({
        "metrics": evaluate_safety.score([
            {"id": "s1", "category": "secret", "subset": "covered", "passed": True},
            {"id": "c1", "category": "clean", "subset": "clean", "passed": False, "patterns": []},
        ]),
        "cases": [{"id": "c1", "category": "clean", "subset": "clean", "passed": False,
                   "patterns": ["role_injection"]}],
    }))
    monkeypatch.setattr(report, "RESULTS_DIR", results)
    combined, md = report.build()
    assert combined["retrieval"] is None
    assert "Not run (`retrieval.json` missing)" in md
    assert "| Status accuracy (exact, 4-way) | 50.0% |" in md
    assert "Skipped (repository not indexed): bert." in md
    assert "| c1 | clean | clean | role_injection |" in md
    json.dumps(combined)  # serialisable
