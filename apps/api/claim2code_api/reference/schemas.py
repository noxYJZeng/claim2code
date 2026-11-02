"""Pydantic models for reference checks (differential tests).

A reference check asks whether a repository function computes what
the paper describes, by running it next to a reference implementation
the model wrote from the paper passage alone:

  - :class:`ReferenceInputs` / :class:`ReferenceImplementation` are
    the two Structured Outputs the model returns. They are separate
    calls on purpose: the inputs are written with the function body
    in view (so they match its types and shapes); the reference is
    written WITHOUT it, so it cannot copy the code it is checking.
  - :class:`ReferenceCheck` is the persisted result, one per claim,
    stored in ``AnalysisState.reference_checks`` (JSONB).
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ReferenceCheckStatus(StrEnum):
    """Outcome of one reference check.

    ``MATCH``: every input that both sides ran gave the same output.
    ``MISMATCH``: at least one input gave different outputs.
    ``ERROR``: nothing could be compared (import failure, crash on
    every input, timeout).
    ``UNTESTABLE``: no callable function, or the claim is not an
    input-to-output computation; nothing was run.
    """

    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    ERROR = "ERROR"
    UNTESTABLE = "UNTESTABLE"


class CaseOutcome(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    TARGET_ERROR = "target_error"
    REFERENCE_ERROR = "reference_error"


# ============================================================
# Model outputs
# ============================================================


class ReferenceInputs(BaseModel):
    """Inputs for the function under test (written with its body in view)."""

    model_config = ConfigDict(extra="forbid")

    testable: bool
    reason: str = Field(
        description="Why the function can or cannot be checked by calling it on inputs.",
    )
    inputs_code: str = Field(
        description=(
            "Python source defining make_inputs() -> list of "
            '{"args": [...], "kwargs": {...}}. Empty when not testable.'
        ),
    )
    instance_code: str = Field(
        description=(
            "For an instance method: Python source defining make_instance() that "
            "returns a small, configured object to call the method on. Empty otherwise."
        ),
    )
    input_notes: str = Field(
        description="Types, shapes and meaning of each argument, for the reference author.",
    )


class ReferenceImplementation(BaseModel):
    """Reference written from the paper passage and the signature only."""

    model_config = ConfigDict(extra="forbid")

    reference_code: str = Field(
        description="Python source defining reference(...) with the same parameters.",
    )
    rtol: float = Field(description="Relative tolerance for comparing outputs, e.g. 1e-5.")
    atol: float = Field(description="Absolute tolerance for comparing outputs, e.g. 1e-8.")
    assumptions: list[str] = Field(
        description="Choices the paper leaves open that the reference had to make.",
    )


# ============================================================
# Persisted result
# ============================================================


class ReferenceTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol_id: UUID
    file_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    qualname: str
    signature: str


class ReferenceCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    inputs: str
    outcome: CaseOutcome
    actual: str | None = None
    expected: str | None = None
    detail: str = ""


class ReferenceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    status: ReferenceCheckStatus
    explanation: str
    target: ReferenceTarget | None = None
    reference_code: str = ""
    inputs_code: str = ""
    instance_code: str = ""
    assumptions: list[str] = Field(default_factory=list)
    rtol: float | None = None
    atol: float | None = None
    cases: list[ReferenceCase] = Field(default_factory=list)
    isolation: list[str] = Field(default_factory=list)
    generated_at: datetime


# ============================================================
# API shapes
# ============================================================


class ReferenceCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    symbol_id: UUID | None = None


class ReferenceCheckListResponse(BaseModel):
    analysis_id: UUID
    enabled: bool
    checks: list[ReferenceCheck]


__all__ = [
    "CaseOutcome",
    "ReferenceCase",
    "ReferenceCheck",
    "ReferenceCheckListResponse",
    "ReferenceCheckRequest",
    "ReferenceCheckStatus",
    "ReferenceImplementation",
    "ReferenceInputs",
    "ReferenceTarget",
]
