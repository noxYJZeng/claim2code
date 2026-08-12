"""PromptInjectionDetector — heuristic regex detection.

Five heuristic patterns (all case-insensitive, all anchored where
useful):

* ``ignore_previous`` — "ignore (all |the |any )?previous
  (instructions|prompts|rules)"
* ``reveal_system`` — "(reveal|show|print|output) (your |the
  )?(system prompt|hidden prompt|instructions)"
* ``role_injection`` — line-start role markers ("system:" or
  "assistant:") embedded inside what should be user content
* ``disregard_prior`` — "disregard (all |the |any )(prior|previous|above)"
* ``you_are_now`` — "you (are|will be) now (a |an )?..."

The detector is **read-only**: it returns hits but does not mutate the
text. False positives would otherwise corrupt legitimate paper text
(e.g. "We ignore the previous state-of-the-art baseline"). The
contract is detect-then-log; the caller decides what to do with the hit
list.

NOT detected (out of scope for Phase 8; different threat model or
explicit non-goal):

* Token-level Unicode evasion (zero-width chars, homoglyphs) — defer
  to a future phase if real attacks surface.
* Semantic-class injection ("forget everything you know") — would need
  an LLM call, defeating the purpose of a cheap regex gate.
"""
from __future__ import annotations

import re

from claim2code_api.safety.exceptions import InjectionHit

_PREVIEW_CAP = 64

# Compiled patterns (all case-insensitive). Order is immaterial — the
# detector does not mutate, just collects.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_previous",
        re.compile(
            r"(?i)ignore (?:all |the |any )?previous "
            r"(?:instructions|prompts|rules)",
        ),
    ),
    (
        "reveal_system",
        re.compile(
            r"(?i)(?:reveal|show|print|output) (?:your |the )?"
            r"(?:system prompt|hidden prompt|instructions)",
        ),
    ),
    (
        "role_injection",
        re.compile(r"(?im)^[ \t]*(?:system|assistant)[ \t]*:[ \t]*\S"),
    ),
    (
        "disregard_prior",
        re.compile(
            r"(?i)disregard (?:all |the |any )(?:prior|previous|above)",
        ),
    ),
    (
        "you_are_now",
        re.compile(r"(?i)you (?:are|will be) now (?:a |an )?[A-Za-z]+"),
    ),
)


def _cap(s: str, *, n: int = _PREVIEW_CAP) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "..."


class PromptInjectionDetector:
    """Detect prompt-injection attempts; do not mutate text."""

    def detect(self, text: str) -> list[InjectionHit]:
        """Return all matches across the pattern set.

        Multiple matches of the same pattern produce multiple
        :class:`InjectionHit` entries (the gate aggregates them into a
        single :class:`SafetyViolation` with ``span_count``).
        """
        hits: list[InjectionHit] = []
        for label, pattern in _PATTERNS:
            for m in pattern.finditer(text):
                hits.append(
                    InjectionHit(
                        label=label,
                        preview=_cap(m.group(0)),
                        start=m.start(),
                        end=m.end(),
                    ),
                )
        return hits