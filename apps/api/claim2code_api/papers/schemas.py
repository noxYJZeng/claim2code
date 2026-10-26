"""Pydantic models for the paper layer.

Three layers of schema:

  - :class:`Paper` / :class:`PaperSection` / :class:`Equation` / :class:`Table`
    are the **parsed-paper** representation produced by PyMuPDF. They
    flow into the LLM prompt and are persisted into the ``papers``
    table's surrounding state in later phases.

  - :class:`Claim` / :class:`PaperEvidence` / :class:`ClaimType` /
    :class:`ClaimList` are the **extracted-claims** representation.
    :class:`ClaimList` is the root model whose ``model_json_schema()``
    is sent verbatim to OpenAI as a Structured-Outputs ``response_format``.

  - :class:`PaperUploadResponse` / :class:`ClaimsResponse` are the
    **API response** shapes that the FastAPI router returns. They are
    the only models the HTTP layer ever imports.

The separation is deliberate: a future Phase 6 verdict field can be
added to :class:`Claim` without touching the parsed-paper or
API-response models.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ============================================================
# 1. Claim + evidence (LLM output)
# ============================================================


class ClaimType(StrEnum):
    """The taxonomy defined in ``docs/specs/spec-summary.md`` §5.

    Inheriting from :class:`enum.StrEnum` (Python 3.11+) means JSON
    serialisation emits the *string* value, not an int ordinal, and
    equality with the string form (``ClaimType.METHOD == "METHOD"``)
    is preserved.
    """

    METHOD = "METHOD"
    IMPLEMENTATION = "IMPLEMENTATION"
    PERFORMANCE = "PERFORMANCE"
    ABLATION = "ABLATION"
    HYPERPARAMETER = "HYPERPARAMETER"
    DATASET = "DATASET"
    COMPLEXITY = "COMPLEXITY"
    TRAINING = "TRAINING"
    EVALUATION = "EVALUATION"


class PaperEvidence(BaseModel):
    """A verbatim snippet of the paper that grounds a claim.

    The offsets are into the *section text* (not the full document)
    so the frontend can render a citation without re-tokenising the
    PDF. ``quote`` is the literal substring — the frontend can verify
    it by slicing the section text.
    """

    model_config = ConfigDict(extra="forbid")

    section_title: str
    page: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    quote: str = Field(min_length=1)


class Claim(BaseModel):
    """One structured technical claim extracted from the paper.

    ``claim_id`` is assigned after extraction (``C1``..``Cn`` in
    paper order) by :meth:`LLMClient.extract_claims`; whatever id
    the model emitted is discarded.
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    claim_type: ClaimType
    confidence: float = Field(ge=0.0, le=1.0)
    paper_evidence: PaperEvidence


class ClaimList(BaseModel):
    """Root schema sent to OpenAI as ``response_format.json_schema``.

    OpenAI's Structured Outputs enforce ``min_items`` / ``max_items``
    constraints, so the LLM is rejected by the API itself if it
    tries to return 0 or 16+ claims. The constraints here are also
    applied at validation time (defence-in-depth — the API enforces
    the same range).
    """

    model_config = ConfigDict(extra="forbid")

    claims: list[Claim] = Field(min_length=1, max_length=20)


# ============================================================
# 2. Parsed paper (PyMuPDF output)
# ============================================================


class PaperSection(BaseModel):
    """A contiguous block of text under one heading.

    ``title`` is empty for the "preamble" before the first heading
    (Abstract + Introduction often merge into one section in this
    representation).
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    page: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str


class Equation(BaseModel):
    """An equation or display-math block (parsed but not extracted
    into claims in Phase 2)."""

    model_config = ConfigDict(extra="forbid")

    text: str
    page: int = Field(ge=1)


class Table(BaseModel):
    """A table (parsed but not extracted into claims in Phase 2)."""

    model_config = ConfigDict(extra="forbid")

    caption: str
    page: int = Field(ge=1)
    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)


class Paper(BaseModel):
    """The fully-parsed representation of one PDF.

    Returned by :func:`claim2code_api.papers.parser.parse_pdf` and
    passed to the LLM claim extractor. Also persisted into the
    ``papers`` table in later phases.
    """

    model_config = ConfigDict(extra="forbid")

    filename: str
    sha256: str = Field(min_length=64, max_length=64)
    n_pages: int = Field(ge=1)
    n_chars: int = Field(ge=0)
    sections: list[PaperSection]
    equations: list[Equation] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)


# ============================================================
# 3. API response shapes
# ============================================================


class PaperUploadResponse(BaseModel):
    """Response body for ``POST /papers``."""

    analysis_id: UUID
    paper_id: UUID
    parse_status: str
    n_claims: int


class PaperMetadataResponse(BaseModel):
    """Response body for ``GET /papers/{paper_id}``."""

    paper_id: UUID
    analysis_id: UUID
    filename: str
    sha256: str
    size_bytes: int
    n_pages: int | None
    n_chars: int | None
    parse_status: str
    uploaded_at: datetime
    parsed_at: datetime | None
    error: str | None


class ClaimsResponse(BaseModel):
    """Response body for ``GET /analyses/{analysis_id}/claims``."""

    analysis_id: UUID
    paper_status: str
    claims: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    """Standard error body returned by the paper router."""

    error: str
    detail: str | None = None
