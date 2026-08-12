"""Responsible-AI safety gate for LLM calls.

This package provides the Phase 8 safety primitives:

* :class:`SafetyGate` (in :mod:`.gate`) — the wrapper around every
  ``chat.completions.create`` call. Redacts secrets + detects prompt
  injections; never raises on detection (audit log only).
* :class:`SecretRedactor` (in :mod:`.redactor`) — replaces matched
  credentials with ``[REDACTED:<label>]``.
* :class:`PromptInjectionDetector` (in :mod:`.detector`) — returns
  hits but does not mutate text (heuristic patterns).
* :class:`SafetyViolation` (in :mod:`.exceptions`) — dataclass
  surfaced via the audit log.

See ``docs/architecture/phase-8-design.md`` for the design rationale.
"""
from __future__ import annotations

from claim2code_api.safety.exceptions import (
    GateResult,
    InjectionHit,
    SafetyError,
    SafetyViolation,
    SecretHit,
)
from claim2code_api.safety.gate import SafetyGate

__all__ = [
    "GateResult",
    "InjectionHit",
    "PromptInjectionDetector",
    "SafetyError",
    "SafetyGate",
    "SafetyViolation",
    "SecretHit",
    "SecretRedactor",
]

# Re-exported lazily via __getattr__ to avoid circular imports.
_LAZY = {"SecretRedactor", "PromptInjectionDetector"}


def __getattr__(name: str) -> object:
    if name == "SecretRedactor":
        from claim2code_api.safety.redactor import SecretRedactor
        return SecretRedactor
    if name == "PromptInjectionDetector":
        from claim2code_api.safety.detector import PromptInjectionDetector
        return PromptInjectionDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")