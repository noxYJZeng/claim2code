"""Typed exceptions for the indexing layer.

Mirrors the convention established by :mod:`claim2code_api.papers.exceptions`
and :mod:`claim2code_api.repos.exceptions`: a base class plus
specific subclasses so the FastAPI router can map each to an
HTTP status without leaking SDK details.

  - :class:`EmbeddingError` → 500 (the embed call failed —
    network, auth, schema, or batch-level error).
  - :class:`EmbeddingDimMismatchError` → 500 (the model
    returned a vector of unexpected dimensionality — usually
    a settings typo).
"""

from __future__ import annotations


class IndexingError(Exception):
    """Base class for all indexing-layer errors."""


class EmbeddingError(IndexingError):
    """A provider call failed: connection, auth, timeout, schema.

    Maps to HTTP 500 in the router.
    """


class EmbeddingDimMismatchError(IndexingError):
    """A provider returned a vector whose length differs from
    ``Settings.openai_embed_dim``.

    Typically caused by switching ``openai_model_embed`` between
    providers without restarting the process, or a misconfigured
    ``openai_embed_dim`` in .env.
    """

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"embedding dimension mismatch: expected {expected}, got {actual}")
        self.expected = expected
        self.actual = actual
