"""Pydantic models for the related-work (literature) layer.

Four layers of schema, mirroring the structure of
``verification/schemas.py``:

  - :class:`ScholarSource` (StrEnum) — which scholarly API
    served the paper.
  - :class:`ScholarPaper` — the canonical paper shape returned by
    either :class:`SemanticScholarClient` or :class:`OpenAlexClient`
    (after the latter's inverted-index abstract reconstruction).
    Mirrors the S2 paper-search response's `data[]` items + TLDR
    + openAccessPdf fields.
  - :class:`Comparison` — the 6-field structured extract the
    LLM returns from one ``extract_comparison`` call. The
    "no model knowledge" rule means every field must cite the
    verbatim abstract or TLDR provided in the inputs.
  - :class:`RankedPaper` — :class:`ScholarPaper` + ``score`` +
    ``score_breakdown`` + ``comparison`` (allowed to be ``None``).
  - :class:`RelatedWorkEntry` — per-claim wrapper (claim_id +
    papers + generated_at + source).
  - :class:`RelatedWorkRunRequest` / ``RunResponse`` /
    ``ListResponse`` — API request / response shapes consumed only
    by the FastAPI router.

The LLM's Structured Outputs ``response_format`` enforces the
:class:`Comparison` schema exactly (``strict: True``).
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ============================================================
# 1. Enums
# ============================================================


class ScholarSource(StrEnum):
    """Which scholarly API served the related papers.

    Stored on :class:`RelatedWorkEntry` so Phase 10's eval can
    measure S2-vs-OpenAlex quality + coverage. Not a literal
    here so the same field can grow to ``crossref`` / ``arxiv``
    in Phase 7.5 without breaking the schema.
    """

    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENALEX = "openalex"


# ============================================================
# 2. Core paper representation
# ============================================================


class ScholarPaper(BaseModel):
    """Canonical paper shape returned by either scholar client.

    Fields are populated from Semantic Scholar's
    ``/graph/v1/paper/search`` response (and translated from
    OpenAlex's works response + inverted-index abstract
    reconstruction by :class:`OpenAlexClient`).

    The LLM comparison extractor consumes
    ``title`` + ``abstract`` + optional ``tldr`` + ``year`` +
    ``authors`` + ``venue``. The other fields are persisted for
    the UI (link to PDF, citation count badge, external IDs).
    """

    model_config = ConfigDict(extra="forbid")

    paper_id: str = Field(min_length=1, description="S2 paperId UUID or OpenAlex W-ID")
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int = Field(ge=1900, le=2100)
    venue: str | None = None
    citation_count: int = Field(ge=0)
    tldr: str | None = None
    abstract: str = Field(min_length=1)
    pdf_url: str | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)


# ============================================================
# 3. LLM comparison extract (6 fields)
# ============================================================


class Comparison(BaseModel):
    """Structured 6-field comparison extracted by the LLM.

    Mirrors the 6 fields from the roadmap's acceptance criterion
    (``phase-roadmap.md:139-141``): Problem / Method / Dataset /
    Metric / Difference / Agreement.

    The "no model knowledge" rule from ``spec-summary.md:148-150``
    is operationalised in ``_COMPARE_SYSTEM_PROMPT`` (see
    ``llm/client.py``): every field must cite the verbatim
    abstract or TLDR provided in the inputs. ``dataset`` and
    ``metric`` are allowed to be ``None`` because not every paper
    evaluates on a standard dataset or reports a single metric.
    """

    model_config = ConfigDict(extra="forbid")

    problem: str = Field(min_length=1)
    method: str = Field(min_length=1)
    dataset: str | None = None
    metric: str | None = None
    difference: str = Field(min_length=1)
    agreement: str = Field(min_length=1)


# ============================================================
# 4. Ranked + entry representation
# ============================================================


class RankedPaper(BaseModel):
    """One paper after ranking + LLM comparison.

    Composed by the workflow from a :class:`ScholarPaper` +
    the ranker's score / breakdown + the LLM's
    :class:`Comparison` (or ``None`` if the LLM call failed —
    the structural guard).

    The acceptance criterion is testable on this model:
    ``title/year/abstract`` non-empty AND ``score ∈ [0, 1]`` AND
    ``comparison IS None OR all 6 fields populated``.
    """

    model_config = ConfigDict(extra="forbid")

    paper: ScholarPaper
    score: float = Field(ge=0.0, le=1.0)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    comparison: Comparison | None = None


class RelatedWorkEntry(BaseModel):
    """One claim's related-work result.

    Persisted as JSONB in ``AnalysisState.related_work``. The
    ``source`` field carries which scholarly API served the
    papers (S2 or OpenAlex — both are recorded so Phase 10 eval
    can measure S2-vs-OpenAlex coverage).
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    papers: list[RankedPaper]
    generated_at: datetime
    source: ScholarSource


# ============================================================
# 5. API request / response shapes
# ============================================================


class RelatedWorkRunRequest(BaseModel):
    """Request body for ``POST /analyses/{id}/related-work``.

    Empty body / no body is rejected by FastAPI's default
    ("body required") — unlike Phase 6's verification where
    empty body meant "verify all claims". Phase 7 takes a
    single ``claim_id`` because related-work is a per-claim
    query (the LLM's input is the claim text + paper evidence).
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    force_refresh: bool = False


class RelatedWorkRunResponse(BaseModel):
    """Response body for ``POST /analyses/{id}/related-work``.

    Carries ``n_papers`` and ``n_with_comparison`` so the UI can
    render summary badges ("5 papers, 4 with comparison") without
    re-iterating the papers list. ``errors`` surfaces per-paper
    LLM errors from the workflow's per-paper isolation pattern.
    """

    model_config = ConfigDict(extra="forbid")

    analysis_id: UUID
    entry: RelatedWorkEntry
    n_papers: int = Field(ge=0)
    n_with_comparison: int = Field(ge=0)
    errors: list[str] = Field(default_factory=list)


class RelatedWorkListResponse(BaseModel):
    """Response body for ``GET /analyses/{id}/related-work``.

    Returns the persisted JSONB verbatim. The GET endpoint
    re-validates each entry through
    :meth:`RelatedWorkEntry.model_validate` and skips malformed
    entries with a log warning (mirrors Phase 6's GET behavior).

    The optional ``claim_id`` query parameter filters to one
    entry; omit it to return all entries.
    """

    model_config = ConfigDict(extra="forbid")

    analysis_id: UUID
    entries: list[RelatedWorkEntry]


# Re-export
__all__ = [
    "ScholarSource",
    "ScholarPaper",
    "Comparison",
    "RankedPaper",
    "RelatedWorkEntry",
    "RelatedWorkRunRequest",
    "RelatedWorkRunResponse",
    "RelatedWorkListResponse",
]