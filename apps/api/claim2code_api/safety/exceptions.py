"""Safety primitives — exceptions + dataclasses.

The safety gate (``SafetyGate``) wraps every LLM call to redact secrets
and detect prompt-injection attempts. Detection is non-fatal: violations
are logged on ``AnalysisState.errors`` and the call proceeds with the
redacted payload. This module defines:

* ``SafetyError`` — base class for any safety subsystem failure
  (e.g. compilation error in the pattern set). Not raised during normal
  operation.
* ``SafetyViolation`` — dataclass capturing a single hit (either a
  redacted secret or a detected injection attempt) plus a redacted
  preview for the audit log.
* ``SecretHit`` / ``InjectionHit`` — dataclasses distinguishing the two
  kinds of detection (redactor mutates; detector does not).

The gate runs at the SDK seam (``llm/client.py``), not at the HTTP
boundary. See ``docs/architecture/phase-8-design.md`` for the full design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


class SafetyError(Exception):
    """Base for safety subsystem failures (e.g. pattern compile error)."""


# ============================================================
# Hit dataclasses
# ============================================================


@dataclass(frozen=True)
class SecretHit:
    """A single redacted credential span.

    ``label`` is the human-readable pattern label (``openai_key``,
    ``aws_access_key``, etc.). ``preview`` is a length-capped snippet of
    the matched span — useful for the audit log but never persisted to
    ``state.errors`` (which keeps only metadata).
    """

    label: str
    preview: str
    start: int
    end: int


@dataclass(frozen=True)
class InjectionHit:
    """A single prompt-injection detection.

    Injections are detected but not mutated (heuristic patterns — false
    positives would lose legitimate paper text like "ignore the previous
    state-of-the-art baseline").
    """

    label: str
    preview: str
    start: int
    end: int


# ============================================================
# Violation (audit-log shape)
# ============================================================

ViolationKind = Literal["secret", "injection"]


@dataclass(frozen=True)
class SafetyViolation:
    """A single safety violation, surfaced for the audit log.

    ``kind`` distinguishes redactions (``"secret"``) from detections
    (``"injection"``). ``pattern_label`` is the per-pattern name
    (``openai_key``, ``ignore_previous``, etc.). ``span_count`` is the
    number of matched spans (one redactor pass may match multiple
    occurrences of the same pattern).
    """

    kind: ViolationKind
    pattern_label: str
    span_count: int
    redacted_preview: str = ""


@dataclass(frozen=True)
class GateResult:
    """Result of ``SafetyGate.wrap_messages(...)``.

    ``messages`` is the redacted message list (same shape as input;
    mutated copies of untrusted strings). ``violations`` is the audit
    log; empty when the input was clean.
    """

    messages: list[dict[str, object]] = field(default_factory=list)
    violations: list[SafetyViolation] = field(default_factory=list)