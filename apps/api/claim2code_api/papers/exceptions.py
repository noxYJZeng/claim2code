"""Custom exceptions for the paper layer.

Kept separate from :mod:`schemas` so the schema module is pure data
(Pydantic models) and doesn't carry import side-effects from
exception classes used in only a few call sites.
"""

from __future__ import annotations


class PaperError(Exception):
    """Base for every paper-layer error that maps to an HTTP response."""


class PdfNoTextLayerError(PaperError):
    """Raised when the uploaded PDF has no extractable text.

    Maps to HTTP 422 with ``{"error": "pdf_has_no_text_layer"}``.
    Scanned / image-only papers need OCR before they can be processed.
    """


class PdfTooLargeError(PaperError):
    """Raised when the uploaded PDF exceeds ``Settings.max_pdf_size_mb``.

    Maps to HTTP 413 with ``{"error": "pdf_too_large"}``.
    """


class PdfInvalidError(PaperError):
    """Raised when the upload is not a valid PDF (bad magic bytes, etc.).

    Maps to HTTP 422 with ``{"error": "pdf_invalid"}``.
    """


class ClaimExtractionError(PaperError):
    """Raised when the LLM call fails or returns an unusable response.

    Maps to HTTP 500 with a JSON body pointing the caller at
    ``GET /analyses/{id}`` for the detailed error trace.
    """
