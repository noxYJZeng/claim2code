"""Typed exceptions for the Repository Intelligence layer.

Each subclass maps to a specific HTTP status code in the FastAPI
router; the ``router`` re-raises them through :class:`fastapi.HTTPException`
so the client sees a meaningful status without leaking internal details.

The split mirrors :mod:`claim2code_api.papers.exceptions`:

  - :class:`RepoInvalidUrlError` → 422 (the URL is well-formed but
    does not match a GitHub ``<owner>/<repo>.git`` shape, or it points
    at a host we don't support in v1).
  - :class:`RepoCloneError` → 500 (network, auth, or git-side failure
    during ``git clone``).
  - :class:`RepoTooLargeError` → 413 (HEAD pre-flight Content-Length
    exceeds ``Settings.max_repo_mb_megs``).
  - :class:`ParseError` → 500 (a tree-sitter or ``ast`` failure that
    is not just "language unsupported").
  - :class:`UnsupportedLanguageError` → 422 (the file extension is
    not in the v1 supported-languages set).
"""

from __future__ import annotations


class RepoError(Exception):
    """Base class for all repo-layer errors."""


class RepoInvalidUrlError(RepoError):
    """The URL is malformed or does not match a supported host."""


class RepoCloneError(RepoError):
    """``git clone`` returned non-zero or its output was unusable."""


class RepoTooLargeError(RepoError):
    """The repo's reported Content-Length exceeds the configured cap."""


class ParseError(RepoError):
    """The tree-sitter or ``ast`` parser raised an unexpected error."""


class UnsupportedLanguageError(RepoError):
    """The file's extension is not in the supported-languages set."""
