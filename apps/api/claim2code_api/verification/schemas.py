"""Pydantic models for the verification layer.

Three layers of schema, mirroring ``papers/schemas.py``:

  - :class:`VerificationStatus` / :class:`DiscrepancyType` /
    :class:`DiscrepancySeverity` are the StrEnums from
    ``docs/specs/spec-summary.md`` §5 (4 statuses, 8 types,
    3 severities).

  - :class:`Verification` / :class:`Discrepancy` /
    :class:`CodeEvidence` / :class:`HyperparameterEvidence` are
    the **core verification representation**. The
    :class:`Verification` schema is what the LLM's Structured
    Outputs ``response_format`` enforces; the deterministic
    checkers build the same shape.

  - :class:`VerificationRunRequest` /
    :class:`VerificationRunResponse` /
    :class:`VerificationListResponse` are the **API response
    shapes** consumed only by the FastAPI router.

Why re-export ``PaperEvidence`` from the paper layer rather than
redefining it: the verifier must cite the exact section / page
/ char range that grounds the claim, and that's a property of
the paper — not a verification-layer concept. Importing keeps
a single source of truth.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from claim2code_api.papers.schemas import PaperEvidence

# ============================================================
# 1. Enums (spec §5)
# ============================================================


class VerificationStatus(StrEnum):
    """The 4-status verdict taxonomy from ``spec-summary.md`` §5.

    The status determines how the structural acceptance rule
    (``paper_evidence AND code_evidence OR UNVERIFIED``) maps
    onto a response — see ``VerificationRunResponse`` and the
    :func:`compose_verification` structural guard.
    """

    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"


class DiscrepancyType(StrEnum):
    """The 8-type discrepancy taxonomy from ``spec-summary.md`` §5."""

    HYPERPARAMETER_MISMATCH = "HYPERPARAMETER_MISMATCH"
    IMPLEMENTATION_MISSING = "IMPLEMENTATION_MISSING"
    UNDOCUMENTED_IMPLEMENTATION = "UNDOCUMENTED_IMPLEMENTATION"
    OBJECTIVE_MISMATCH = "OBJECTIVE_MISMATCH"
    DATASET_MISMATCH = "DATASET_MISMATCH"
    METRIC_MISMATCH = "METRIC_MISMATCH"
    CONFIGURATION_MISMATCH = "CONFIGURATION_MISMATCH"
    AMBIGUOUS = "AMBIGUOUS"


class DiscrepancySeverity(StrEnum):
    """The 3-severity taxonomy from ``spec-summary.md`` §5."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ============================================================
# 2. Core verification representation
# ============================================================


class CodeEvidence(BaseModel):
    """A specific code citation from one of the top-K candidates.

    The verifier's "no model knowledge" rule (see
    ``docs/specs/spec-summary.md:148-150``) means each
    ``CodeEvidence`` entry must:

      - reference a ``symbol_id`` that appears in the candidate
        list provided to the LLM;
      - carry a ``quote`` that is a verbatim substring of that
        candidate's ``signature`` or ``docstring``;
      - cite a ``file_path`` / line range that matches the
        candidate.

    Empty ``code_evidence`` is permitted at the schema level;
    the structural guard in :func:`compose_verification` coerces
    such verifications to ``UNVERIFIED``.
    """

    model_config = ConfigDict(extra="forbid")

    symbol_id: UUID
    file_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    quote: str = Field(min_length=1)


class Discrepancy(BaseModel):
    """One detected discrepancy between paper claim and code.

    ``detail`` is a human-readable explanation suitable for the
    frontend (e.g. ``"paper says lr=5e-5, code defaults to
    lr=1e-3"``). Discrepancies are emitted by both the
    deterministic checkers (with a fixed ``type`` /
    ``severity`` pair) and the LLM (with the LLM's own choice).
    """

    model_config = ConfigDict(extra="forbid")

    type: DiscrepancyType
    severity: DiscrepancySeverity
    detail: str = Field(min_length=1)


class HyperparameterEvidence(BaseModel):
    """Structured hyperparameter value extracted by the verifier LLM.

    The deterministic ``HyperparameterChecker`` cross-validates
    its regex-extracted numbers against this field. If the two
    disagree (e.g. paper says ``lr=5e-5`` but the LLM extracted
    ``lr=1e-3``), an ``AMBIGUOUS`` discrepancy is emitted with
    severity ``LOW``.

    ``value`` is stored as the original string from the LLM
    (e.g. ``"5e-5"``) rather than parsed into a ``float`` so we
    don't lose precision or format on round-trips through JSONB.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source: Literal["paper", "code"]


class Verification(BaseModel):
    """One claim's verification result.

    The structural acceptance rule from
    ``docs/architecture/phase-roadmap.md:128`` is enforced at the
    :func:`compose_verification` layer (auto-coerce to
    ``UNVERIFIED`` when ``code_evidence`` is empty), not by this
    schema — so the schema permits any combination of fields
    that the LLM or checkers emit, and the structural check
    happens at composition time.

    Persisted as JSONB in ``AnalysisState.verifications`` (the
    column was pre-staged in Phase 1's ``0001_init`` migration).
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    paper_evidence: PaperEvidence
    code_evidence: list[CodeEvidence] = Field(default_factory=list)
    discrepancies: list[Discrepancy] = Field(default_factory=list)
    hyperparameters: list[HyperparameterEvidence] = Field(default_factory=list)


# ============================================================
# 3. API request / response shapes
# ============================================================


class VerificationRunRequest(BaseModel):
    """Optional request body for ``POST /analyses/{id}/verifications``.

    Empty body / no body → verify all extracted claims with
    ``top_k=8``: paper prose ranks the implementing symbol 6th-8th
    often enough (e.g. a sampler named ``generalized_steps`` for a
    claim about σ_t = 0) that 5 candidates miss it.
    """

    model_config = ConfigDict(extra="forbid")

    claim_ids: list[str] | None = None
    top_k: int = Field(default=8, ge=1, le=20)


class VerificationRunResponse(BaseModel):
    """Response body for ``POST /analyses/{id}/verifications``.

    The four ``n_*`` counters surface the per-status totals so
    the frontend can render a summary chart without re-iterating
    the verifications list. ``verifications`` is the full list,
    sorted by ``claim_id`` for determinism.
    """

    analysis_id: UUID
    n_verified: int = Field(ge=0)
    n_partially_verified: int = Field(ge=0)
    n_contradicted: int = Field(ge=0)
    n_unverified: int = Field(ge=0)
    verifications: list[Verification]


class VerificationListResponse(BaseModel):
    """Response body for ``GET /analyses/{id}/verifications``.

    The one named in the roadmap (``phase-roadmap.md:126``).
    Mirrors ``GET /analyses/{id}/claims`` (papers/router.py:150-178):
    read ``AnalysisState.verifications`` JSONB and return verbatim.
    """

    analysis_id: UUID
    verifications: list[Verification]