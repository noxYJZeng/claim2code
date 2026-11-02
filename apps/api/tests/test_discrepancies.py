"""Unit tests for :mod:`claim2code_api.verification.discrepancies`.

Coverage:

  - ``merge_discrepancies``: deterministic first, LLM-emitted appended.
  - ``merge_discrepancies``: ``None`` LLM list returns deterministic unchanged.
  - ``determine_status``:
      * structural guard — no code_evidence → UNVERIFIED.
      * HIGH-severity discrepancy → CONTRADICTED.
      * MEDIUM/LOW discrepancy + code evidence → PARTIALLY_VERIFIED.
      * no discrepancies + LLM VERIFIED → VERIFIED.
      * no discrepancies + no LLM → UNVERIFIED.
"""
from __future__ import annotations

import pytest

from claim2code_api.verification.discrepancies import (
    determine_status,
    merge_discrepancies,
)
from claim2code_api.verification.schemas import (
    Discrepancy,
    DiscrepancySeverity,
    DiscrepancyType,
    VerificationStatus,
)


def _disc(t: DiscrepancyType, s: DiscrepancySeverity, d: str = "x") -> Discrepancy:
    return Discrepancy(type=t, severity=s, detail=d)


# ============================================================
# merge_discrepancies
# ============================================================


def test_merge_empty_inputs_returns_empty() -> None:
    assert merge_discrepancies(deterministic=[], llm_emitted=None) == []
    assert merge_discrepancies(deterministic=[], llm_emitted=[]) == []


def test_merge_deterministic_only() -> None:
    a = _disc(DiscrepancyType.HYPERPARAMETER_MISMATCH, DiscrepancySeverity.HIGH)
    out = merge_discrepancies(deterministic=[a], llm_emitted=None)
    assert out == [a]


def test_merge_deterministic_then_llm() -> None:
    a = _disc(DiscrepancyType.HYPERPARAMETER_MISMATCH, DiscrepancySeverity.HIGH)
    b = _disc(DiscrepancyType.IMPLEMENTATION_MISSING, DiscrepancySeverity.MEDIUM)
    out = merge_discrepancies(deterministic=[a], llm_emitted=[b])
    assert out == [a, b]


# ============================================================
# determine_status — structural guard
# ============================================================


def test_determine_status_structural_guard_no_code_evidence() -> None:
    """No code_evidence → UNVERIFIED, regardless of discrepancies."""
    assert (
        determine_status(
            has_code_evidence=False,
            discrepancies=[
                _disc(DiscrepancyType.HYPERPARAMETER_MISMATCH, DiscrepancySeverity.HIGH)
            ],
            llm_status=VerificationStatus.VERIFIED,
        )
        == VerificationStatus.UNVERIFIED
    )


# ============================================================
# determine_status — CONTRADICTED
# ============================================================


def test_determine_status_high_severity_is_contradicted() -> None:
    assert (
        determine_status(
            has_code_evidence=True,
            discrepancies=[
                _disc(DiscrepancyType.OBJECTIVE_MISMATCH, DiscrepancySeverity.HIGH)
            ],
            llm_status=VerificationStatus.VERIFIED,
        )
        == VerificationStatus.CONTRADICTED
    )


def test_determine_status_llm_contradicted_wins_over_partial() -> None:
    """LLM says CONTRADICTED even with no HIGH deterministic → CONTRADICTED."""
    assert (
        determine_status(
            has_code_evidence=True,
            discrepancies=[
                _disc(DiscrepancyType.METRIC_MISMATCH, DiscrepancySeverity.LOW)
            ],
            llm_status=VerificationStatus.CONTRADICTED,
        )
        == VerificationStatus.CONTRADICTED
    )


@pytest.mark.parametrize("llm_status", [VerificationStatus.VERIFIED, VerificationStatus.CONTRADICTED])
def test_determine_status_setting_mismatches_are_partial(llm_status: VerificationStatus) -> None:
    """A differing default or initialisation is a detail, even when sent as HIGH."""
    assert (
        determine_status(
            has_code_evidence=True,
            discrepancies=[
                _disc(DiscrepancyType.HYPERPARAMETER_MISMATCH, DiscrepancySeverity.HIGH),
                _disc(DiscrepancyType.CONFIGURATION_MISMATCH, DiscrepancySeverity.HIGH),
            ],
            llm_status=llm_status,
        )
        == VerificationStatus.PARTIALLY_VERIFIED
    )


def test_determine_status_setting_mismatch_does_not_mask_a_central_one() -> None:
    assert (
        determine_status(
            has_code_evidence=True,
            discrepancies=[
                _disc(DiscrepancyType.HYPERPARAMETER_MISMATCH, DiscrepancySeverity.MEDIUM),
                _disc(DiscrepancyType.OBJECTIVE_MISMATCH, DiscrepancySeverity.HIGH),
            ],
            llm_status=VerificationStatus.PARTIALLY_VERIFIED,
        )
        == VerificationStatus.CONTRADICTED
    )


def test_determine_status_llm_contradicted_without_discrepancies_stays() -> None:
    assert (
        determine_status(
            has_code_evidence=True,
            discrepancies=[],
            llm_status=VerificationStatus.CONTRADICTED,
        )
        == VerificationStatus.CONTRADICTED
    )


# ============================================================
# determine_status — PARTIALLY_VERIFIED
# ============================================================


def test_determine_status_low_discrepancy_is_partial() -> None:
    assert (
        determine_status(
            has_code_evidence=True,
            discrepancies=[
                _disc(DiscrepancyType.METRIC_MISMATCH, DiscrepancySeverity.LOW)
            ],
            llm_status=VerificationStatus.VERIFIED,
        )
        == VerificationStatus.PARTIALLY_VERIFIED
    )


def test_determine_status_medium_discrepancy_is_partial() -> None:
    assert (
        determine_status(
            has_code_evidence=True,
            discrepancies=[
                _disc(DiscrepancyType.DATASET_MISMATCH, DiscrepancySeverity.MEDIUM)
            ],
            llm_status=VerificationStatus.VERIFIED,
        )
        == VerificationStatus.PARTIALLY_VERIFIED
    )


# ============================================================
# determine_status — VERIFIED
# ============================================================


def test_determine_status_no_discrepancy_and_verified_is_verified() -> None:
    assert (
        determine_status(
            has_code_evidence=True,
            discrepancies=[],
            llm_status=VerificationStatus.VERIFIED,
        )
        == VerificationStatus.VERIFIED
    )


def test_determine_status_partial_without_discrepancies_is_verified() -> None:
    """LLM says PARTIALLY_VERIFIED but no discrepancy backs it, code_evidence
    present → VERIFIED: PARTIALLY_VERIFIED needs >=1 LOW/MEDIUM discrepancy
    (phase-6-design.md §2.5)."""
    assert (
        determine_status(
            has_code_evidence=True,
            discrepancies=[],
            llm_status=VerificationStatus.PARTIALLY_VERIFIED,
        )
        == VerificationStatus.VERIFIED
    )


# ============================================================
# determine_status — UNVERIFIED fallbacks
# ============================================================


def test_determine_status_no_llm_and_no_discrepancies_is_unverified() -> None:
    """Rare path: LLM skipped (HIGH), code_evidence not populated,
    no discrepancies surfaced → default to UNVERIFIED."""
    assert (
        determine_status(
            has_code_evidence=True,
            discrepancies=[],
            llm_status=None,
        )
        == VerificationStatus.UNVERIFIED
    )


def test_determine_status_llm_says_unverified_is_unverified() -> None:
    assert (
        determine_status(
            has_code_evidence=True,
            discrepancies=[],
            llm_status=VerificationStatus.UNVERIFIED,
        )
        == VerificationStatus.UNVERIFIED
    )