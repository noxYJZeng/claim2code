"""Errors raised by the reference-check layer."""
from __future__ import annotations


class ReferenceCheckLLMError(RuntimeError):
    """The model call failed or returned output that does not validate."""


class ReferenceCheckNotFoundError(LookupError):
    """The claim or symbol named in the request does not exist."""


__all__ = ["ReferenceCheckLLMError", "ReferenceCheckNotFoundError"]
