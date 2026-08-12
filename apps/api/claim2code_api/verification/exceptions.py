"""Custom exceptions for the verification layer.

Convention (mirrors ``papers/exceptions.py`` and
``repos/exceptions.py``): a base ``VerificationError`` plus
specific subclasses that the router maps to HTTP status codes.
Keeping these separate from the Pydantic schemas keeps the
schemas module pure data (no import side-effects from exception
classes used in only a handful of call sites).
"""
from __future__ import annotations


class VerificationError(Exception):
    """Base for every verification-layer error that maps to an HTTP response."""


class VerificationLLMError(VerificationError):
    """Raised when the LLM call fails or returns an unusable response.

    Mirrors ``papers.exceptions.ClaimExtractionError`` — but the
    two are deliberately separate so the router can map verifier
    failures to a distinct ``error_code`` in the JSON body.

    Maps to HTTP 500 with ``{"error": "verification_llm_failed"}``.
    """


class VerificationConfigError(VerificationError):
    """Raised when the alias JSON files are missing or malformed.

    Maps to HTTP 500 with ``{"error": "verification_config_invalid"}``.
    Configuration errors are developer errors, not runtime errors;
    the response should never reach a healthy production deployment.
    """


class VerificationNotFoundError(VerificationError):
    """Raised when ``claim_ids`` references an unknown claim_id.

    Maps to HTTP 422 with ``{"error": "claim_id_not_found", ...}``.
    """