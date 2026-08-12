"""Unit tests for :mod:`claim2code_api.verification.hyperparameter_checker`.

Coverage:

  - Regex extract: paper- and code-side name/value pairs.
  - Aliases: ``lr`` / ``η`` / ``learning_rate`` all map to ``learning_rate``.
  - Regime: additive (batch_size / num_epochs / hidden_size) tolerates
    5 % relative diff, fails on > 5 %.
  - Regime: multiplicative (lr / temperature / beta / gamma) tolerates
    10× ratio, fails on > 10×.
  - Python signature form (``lr: float = 5e-5``) is recognised.
  - Empty text → empty dict.
  - When both sides have a name and the values disagree, one
    HIGH-severity discrepancy is emitted.
  - When only one side has the name, no discrepancy (LLM residual
    handles asymmetry).
  - Discrepancy detail string names both values.
"""
from __future__ import annotations

from uuid import uuid4

from claim2code_api.repos.schemas import CodeLanguage, CodeSymbolKind, RetrieveMatch
from claim2code_api.verification.hyperparameter_checker import HyperparameterChecker
from claim2code_api.verification.schemas import (
    DiscrepancySeverity,
    DiscrepancyType,
)

# ============================================================
# Fixtures / helpers
# ============================================================


def _make_candidate(*, signature: str, docstring: str | None = None) -> RetrieveMatch:
    return RetrieveMatch(
        symbol_id=uuid4(),
        kind=CodeSymbolKind.FUNCTION,
        language=CodeLanguage.PYTHON,
        file_path="train.py",
        start_line=1,
        start_col=0,
        end_line=10,
        end_col=0,
        name="train",
        signature=signature,
        docstring=docstring,
        similarity=0.95,
    )


def _checker() -> HyperparameterChecker:
    return HyperparameterChecker()


# ============================================================
# extract
# ============================================================


def test_extract_empty_text_returns_empty() -> None:
    assert _checker().extract("") == {}


def test_extract_learning_rate_aliases() -> None:
    """``lr``, ``η``, and ``learning_rate`` all canonicalise to learning_rate."""
    c = _checker()
    assert c.extract("lr=5e-5") == {"learning_rate": 5e-5}
    assert c.extract("learning_rate=5e-5") == {"learning_rate": 5e-5}
    assert c.extract("η=5e-5") == {"learning_rate": 5e-5}


def test_extract_batch_size_additive() -> None:
    assert _checker().extract("batch_size: 32") == {"batch_size": 32.0}


def test_extract_python_signature_form() -> None:
    """``lr: float = 5e-5`` is a real Python default-arg form."""
    c = _checker()
    out = c.extract("def train(lr: float = 5e-5, batch_size: int = 32): pass")
    assert out == {"learning_rate": 5e-5, "batch_size": 32.0}


def test_extract_multiple_occurrences_last_wins() -> None:
    c = _checker()
    out = c.extract("lr=1e-3\nlr=5e-5")
    assert out == {"learning_rate": 5e-5}


def test_extract_drops_unparseable_values() -> None:
    c = _checker()
    out = c.extract("lr=abc")
    assert out == {}


# ============================================================
# compare — both sides present, matching values
# ============================================================


def test_compare_no_discrepancy_when_values_match() -> None:
    c = _checker()
    paper_text = "We use learning_rate=5e-5 with batch_size=32."
    cand = _make_candidate(
        signature="def train(learning_rate=5e-5, batch_size=32): ..."
    )
    out = c.compare(
        claim_text="", paper_evidence_quote=paper_text, candidates=[cand],
    )
    assert out == []


# ============================================================
# compare — both sides present, mismatching values
# ============================================================


def test_compare_emits_high_for_lr_10x_mismatch() -> None:
    c = _checker()
    out = c.compare(
        claim_text="",
        paper_evidence_quote="learning_rate=5e-5",
        candidates=[_make_candidate(signature="def train(learning_rate=1e-3): ...")],
    )
    assert len(out) == 1
    assert out[0].type == DiscrepancyType.HYPERPARAMETER_MISMATCH
    assert out[0].severity == DiscrepancySeverity.HIGH


def test_compare_emits_high_for_batch_size_5pct_mismatch() -> None:
    c = _checker()
    # Paper says 32; code says 64 → 100% relative diff > 5% threshold.
    out = c.compare(
        claim_text="",
        paper_evidence_quote="batch_size=32",
        candidates=[_make_candidate(signature="def train(batch_size=64): ...")],
    )
    assert len(out) == 1
    assert out[0].type == DiscrepancyType.HYPERPARAMETER_MISMATCH
    assert out[0].severity == DiscrepancySeverity.HIGH


def test_compare_tolerates_batch_size_within_5pct() -> None:
    """batch_size 32 vs 33 → 3% diff, within 5% tolerance."""
    c = _checker()
    out = c.compare(
        claim_text="",
        paper_evidence_quote="batch_size=32",
        candidates=[_make_candidate(signature="def train(batch_size=33): ...")],
    )
    assert out == []


def test_compare_tolerates_learning_rate_within_10x() -> None:
    """lr 5e-5 vs 1e-4 → ratio 2×, within 10× tolerance."""
    c = _checker()
    out = c.compare(
        claim_text="",
        paper_evidence_quote="learning_rate=5e-5",
        candidates=[_make_candidate(signature="def train(learning_rate=1e-4): ...")],
    )
    assert out == []


def test_compare_does_not_emit_when_only_one_side_has_name() -> None:
    """Paper mentions lr; code does not → no discrepancy (LLM residual)."""
    c = _checker()
    out = c.compare(
        claim_text="",
        paper_evidence_quote="learning_rate=5e-5",
        candidates=[_make_candidate(signature="def train(batch_size=32): ...")],
    )
    assert out == []


def test_compare_does_not_emit_when_only_one_side_has_name_reverse() -> None:
    """Code mentions lr; paper does not → no discrepancy."""
    c = _checker()
    out = c.compare(
        claim_text="",
        paper_evidence_quote="batch_size=32",
        candidates=[_make_candidate(signature="def train(learning_rate=1e-3): ...")],
    )
    assert out == []


# ============================================================
# compare — detail formatting
# ============================================================


def test_compare_detail_names_both_values() -> None:
    c = _checker()
    out = c.compare(
        claim_text="",
        paper_evidence_quote="learning_rate=5e-5",
        candidates=[_make_candidate(signature="def train(learning_rate=1e-3): ...")],
    )
    # ``f"{value:g}"`` may render ``5e-5`` as ``5e-05`` (drops trailing
    # zero) and ``1e-3`` as ``0.001``; accept either form.
    detail = out[0].detail
    assert "learning_rate" in detail
    assert "5e-5" in detail or "5e-05" in detail or "0.00005" in detail
    assert "1e-3" in detail or "0.001" in detail


# ============================================================
# Empty inputs
# ============================================================


def test_compare_empty_paper_and_empty_candidates() -> None:
    c = _checker()
    out = c.compare(claim_text="", paper_evidence_quote="", candidates=[])
    assert out == []


# ============================================================
# Custom tolerance override (testing seam)
# ============================================================


def test_compare_respects_custom_additive_tolerance() -> None:
    """With additive_relative_tolerance=0.0, any diff fails."""
    c = HyperparameterChecker(additive_relative_tolerance=0.0)
    out = c.compare(
        claim_text="",
        paper_evidence_quote="batch_size=32",
        candidates=[_make_candidate(signature="def train(batch_size=33): ...")],
    )
    assert len(out) == 1
    assert out[0].severity == DiscrepancySeverity.HIGH