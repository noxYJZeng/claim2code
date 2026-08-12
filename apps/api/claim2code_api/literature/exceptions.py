"""Custom exceptions for the related-work (literature) layer.

Convention (mirrors ``verification/exceptions.py``): a base
``LiteratureError`` plus specific subclasses that the router
maps to HTTP status codes. Keeping these separate from the
Pydantic schemas keeps the schemas module pure data (no
import side-effects from exception classes used in only a
handful of call sites).
"""
from __future__ import annotations


class LiteratureError(Exception):
    """Base for every literature-layer error that maps to an HTTP response."""


class LiteratureSearchError(LiteratureError):
    """Raised when both Semantic Scholar AND OpenAlex fail to return papers.

    Both clients are tried in sequence (S2 first, OpenAlex as
    fallback). If the second also fails, the workflow bubbles
    this error so the router can map it to HTTP 500 with
    ``{"error": "literature_search_failed", ...}``. The error
    string carries the underlying status codes for observability.
    """


class LiteratureConfigError(LiteratureError):
    """Raised when the scholar client is misconfigured.

    Examples: invalid API key format, missing required query
    parameter, malformed ScholarPaper after parsing. Maps to
    HTTP 500 with ``{"error": "literature_config_invalid"}``.
    Configuration errors are developer errors, not runtime
    errors; the response should never reach a healthy production
    deployment.
    """


class LiteratureNotFoundError(LiteratureError):
    """Raised when ``claim_id`` references an unknown claim_id.

    The workflow validates ``claim_id`` against
    ``AnalysisState.claims`` before persisting. Maps to HTTP 422
    with ``{"error": "claim_id_not_found", ...}``.
    """


class LiteratureComparisonError(LiteratureError):
    """Raised when the LLM comparison call fails for a single paper.

    Per-paper isolation (mirrors Phase 6's per-claim isolation):
    one paper's LLM failure does NOT abort the rest of the run.
    The workflow catches this exception, records the error on
    ``AnalysisState.errors``, sets that paper's ``comparison`` to
    ``None`` (the structural guard), and continues.

    Maps to HTTP 500 only if raised at the top level (which the
    workflow deliberately prevents). The router never sees this
    exception directly — the workflow handles it.
    """