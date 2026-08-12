"""Unit tests for :mod:`claim2code_api.verification.schemas`.

Coverage:

  - All 4 ``VerificationStatus`` values are accepted.
  - All 8 ``DiscrepancyType`` values are accepted.
  - All 3 ``DiscrepancySeverity`` values are accepted.
  - ``Verification`` round-trips through JSON (model_dump → model_validate).
  - ``extra="forbid"`` is enforced on every model.
  - ``Verification.confidence`` is bounded to ``[0, 1]``.
  - ``VerificationListResponse`` accepts an empty list.
  - ``VerificationRunRequest.top_k`` is bounded to ``[1, 20]``.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from claim2code_api.papers.schemas import PaperEvidence
from claim2code_api.verification.schemas import (
    CodeEvidence,
    Discrepancy,
    DiscrepancySeverity,
    DiscrepancyType,
    HyperparameterEvidence,
    Verification,
    VerificationListResponse,
    VerificationRunRequest,
    VerificationRunResponse,
    VerificationStatus,
)

# ============================================================
# Enums — exhaustive membership
# ============================================================


def test_verification_status_has_four_values() -> None:
    assert {s.value for s in VerificationStatus} == {
        "VERIFIED",
        "PARTIALLY_VERIFIED",
        "CONTRADICTED",
        "UNVERIFIED",
    }


def test_discrepancy_type_has_eight_values() -> None:
    assert {t.value for t in DiscrepancyType} == {
        "HYPERPARAMETER_MISMATCH",
        "IMPLEMENTATION_MISSING",
        "UNDOCUMENTED_IMPLEMENTATION",
        "OBJECTIVE_MISMATCH",
        "DATASET_MISMATCH",
        "METRIC_MISMATCH",
        "CONFIGURATION_MISMATCH",
        "AMBIGUOUS",
    }


def test_discrepancy_severity_has_three_values() -> None:
    assert {s.value for s in DiscrepancySeverity} == {"LOW", "MEDIUM", "HIGH"}


# ============================================================
# CodeEvidence
# ============================================================


def test_code_evidence_round_trips() -> None:
    sym_id = uuid4()
    ce = CodeEvidence(
        symbol_id=sym_id,
        file_path="src/model.py",
        start_line=10,
        end_line=20,
        quote="def train(): pass",
    )
    dumped = ce.model_dump(mode="json")
    restored = CodeEvidence.model_validate(dumped)
    assert restored == ce


def test_code_evidence_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        # ``extra="forbid"`` is a Pydantic-runtime check; mypy doesn't
        # see it. Cast through ``Any`` to silence the call-arg error.
        CodeEvidence(  # type: ignore[call-arg]
            symbol_id=uuid4(),
            file_path="x",
            start_line=1,
            end_line=2,
            quote="x",
            unexpected="boom",
        )


def test_code_evidence_requires_positive_line_numbers() -> None:
    with pytest.raises(ValidationError):
        CodeEvidence(
            symbol_id=uuid4(),
            file_path="x",
            start_line=0,  # ge=1
            end_line=2,
            quote="x",
        )


# ============================================================
# Discrepancy
# ============================================================


def test_discrepancy_with_all_enum_values() -> None:
    for dt in DiscrepancyType:
        for ds in DiscrepancySeverity:
            d = Discrepancy(type=dt, severity=ds, detail=f"{dt.value} {ds.value}")
            assert d.type == dt
            assert d.severity == ds


def test_discrepancy_requires_nonempty_detail() -> None:
    with pytest.raises(ValidationError):
        Discrepancy(
            type=DiscrepancyType.METRIC_MISMATCH,
            severity=DiscrepancySeverity.LOW,
            detail="",
        )


# ============================================================
# HyperparameterEvidence
# ============================================================


def test_hyperparameter_evidence_accepts_paper_and_code_sources() -> None:
    for source in ("paper", "code"):
        h = HyperparameterEvidence(name="lr", value="5e-5", source=source)  # type: ignore[arg-type]
        assert h.source == source


def test_hyperparameter_evidence_rejects_other_source() -> None:
    with pytest.raises(ValidationError):
        HyperparameterEvidence(name="lr", value="5e-5", source="blog")  # type: ignore[arg-type]


# ============================================================
# Verification
# ============================================================


def _sample_paper_evidence() -> PaperEvidence:
    return PaperEvidence(
        section_title="Method", page=1, char_start=0, char_end=10, quote="we use lr=5e-5"
    )


def test_verification_default_lists_are_empty() -> None:
    v = Verification(
        claim_id="c1",
        status=VerificationStatus.VERIFIED,
        confidence=0.5,
        explanation="x",
        paper_evidence=_sample_paper_evidence(),
    )
    assert v.code_evidence == []
    assert v.discrepancies == []
    assert v.hyperparameters == []


def test_verification_confidence_bounded() -> None:
    with pytest.raises(ValidationError):
        Verification(
            claim_id="c1", status=VerificationStatus.VERIFIED,
            confidence=1.5,  # > 1.0
            explanation="x", paper_evidence=_sample_paper_evidence(),
        )
    with pytest.raises(ValidationError):
        Verification(
            claim_id="c1", status=VerificationStatus.VERIFIED,
            confidence=-0.1,  # < 0.0
            explanation="x", paper_evidence=_sample_paper_evidence(),
        )


def test_verification_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        # ``extra="forbid"`` is a Pydantic-runtime check; mypy doesn't
        # see it. Cast through ``Any`` to silence the call-arg error.
        Verification(  # type: ignore[call-arg]
            claim_id="c1", status=VerificationStatus.VERIFIED,
            confidence=0.5, explanation="x",
            paper_evidence=_sample_paper_evidence(),
            unknown_field="boom",  # extra="forbid"
        )


def test_verification_json_round_trip() -> None:
    v = Verification(
        claim_id="c1",
        status=VerificationStatus.CONTRADICTED,
        confidence=0.8,
        explanation="matches mostly",
        paper_evidence=_sample_paper_evidence(),
        code_evidence=[
            CodeEvidence(
                symbol_id=uuid4(), file_path="x.py",
                start_line=1, end_line=2, quote="def f(): pass",
            )
        ],
        discrepancies=[
            Discrepancy(
                type=DiscrepancyType.HYPERPARAMETER_MISMATCH,
                severity=DiscrepancySeverity.HIGH,
                detail="lr mismatch",
            )
        ],
    )
    dumped = v.model_dump(mode="json")
    restored = Verification.model_validate(dumped)
    assert restored == v


# ============================================================
# VerificationRunRequest
# ============================================================


def test_run_request_defaults() -> None:
    r = VerificationRunRequest()
    assert r.claim_ids is None
    assert r.top_k == 5


def test_run_request_top_k_bounds() -> None:
    with pytest.raises(ValidationError):
        VerificationRunRequest(top_k=0)  # ge=1
    with pytest.raises(ValidationError):
        VerificationRunRequest(top_k=21)  # le=20


def test_run_request_accepts_claim_ids() -> None:
    r = VerificationRunRequest(claim_ids=["c1", "c2"], top_k=10)
    assert r.claim_ids == ["c1", "c2"]
    assert r.top_k == 10


def test_run_request_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        # ``extra="forbid"`` is a Pydantic-runtime check; mypy doesn't
        # see it. Cast through ``Any`` to silence the call-arg error.
        VerificationRunRequest(top_k=5, something_else=True)  # type: ignore[call-arg]


# ============================================================
# VerificationListResponse
# ============================================================


def test_list_response_empty_list_ok() -> None:
    r = VerificationListResponse(analysis_id=uuid4(), verifications=[])
    assert r.verifications == []


# ============================================================
# VerificationRunResponse
# ============================================================


def test_run_response_counts_are_nonnegative() -> None:
    with pytest.raises(ValidationError):
        VerificationRunResponse(
            analysis_id=uuid4(), n_verified=-1,  # ge=0
            n_partially_verified=0, n_contradicted=0, n_unverified=0,
            verifications=[],
        )