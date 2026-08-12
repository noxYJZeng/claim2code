"""SecretRedactor — regex-based replacement of credential spans.

Eight credential patterns (all compiled at module import time):

* ``openai_key`` — ``sk-...`` and ``sk-proj-...``
* ``anthropic_key`` — ``sk-ant-...``
* ``aws_access_key`` — ``AKIA...``
* ``gcp_api_key`` — ``AIza...``
* ``github_pat`` — ``gh[pousr]_...``
* ``pem_private_key`` — ``-----BEGIN ... PRIVATE KEY-----`` (with
  ``-----END ...`` line pair)
* ``bearer_token`` — ``Authorization: Bearer ...``
* ``env_assignment`` — known environment-variable assignments in
  ``.env``-style lines (``OPENAI_API_KEY=...``, etc.)

Each pattern is compiled with ``re.IGNORECASE`` to catch
case-tampered variants. The redactor mutates the input by replacing
matched spans with ``[REDACTED:<label>]``.

NOT matched (explicit non-goals of Phase 8, deferred to a future phase
when enterprise users request them):

* emails, phone numbers, SSNs, IPv4, URLs — papers legitimately cite
  these; false positives would corrupt the structured extraction.

The redactor never raises on malformed input — it always returns a
string (possibly equal to the input) and a list of hits (possibly empty).
"""
from __future__ import annotations

import re

from claim2code_api.safety.exceptions import SecretHit

# Preview length cap: how much of the matched span to include in the
# audit log. 32 chars is enough for debugging ("starts with sk-proj-...")
# without leaking the secret itself if the audit log ever leaks.
_PREVIEW_CAP = 32

# Order matters: PEM must run before env_assignment (because an env
# assignment like ``-----BEGIN ... PRIVATE KEY-----=foo`` would
# otherwise match the .env regex first). All patterns are case-insensitive.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9]{20,}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9-]{20,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("gcp_api_key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("github_pat", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    (
        "bearer_token",
        re.compile(r"(?im)Authorization:\s*Bearer\s+[A-Za-z0-9-_.]{20,}"),
    ),
    (
        "env_assignment",
        re.compile(
            r"(?im)^[ \t]*(?:OPENAI_API_KEY|ANTHROPIC_API_KEY|AWS_ACCESS_KEY_ID"
            r"|AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN|DATABASE_URL"
            r"|POSTGRES_PASSWORD)[ \t]*=[ \t]*\S+",
        ),
    ),
)

# PEM private keys: BEGIN/END pair (multi-line). Handled separately
# because re.finditer cannot easily match across lines with a single
# pattern while preserving exact spans.
_PEM_BEGIN = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_PEM_END = re.compile(r"-----END [A-Z ]*PRIVATE KEY-----")


def _cap(s: str, *, n: int = _PREVIEW_CAP) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "..."


def _match(label: str, pattern: re.Pattern[str], text: str) -> tuple[str, list[SecretHit]]:
    """Apply ``pattern`` to ``text``; return (new_text, hits)."""
    hits: list[SecretHit] = []
    parts: list[str] = []
    cursor = 0
    for m in pattern.finditer(text):
        parts.append(text[cursor:m.start()])
        parts.append(f"[REDACTED:{label}]")
        hits.append(
            SecretHit(
                label=label,
                preview=_cap(m.group(0)),
                start=m.start(),
                end=m.end(),
            ),
        )
        cursor = m.end()
    parts.append(text[cursor:])
    return "".join(parts), hits


def _redact_pem(text: str) -> tuple[str, list[SecretHit]]:
    """Redact PEM private-key blocks (BEGIN...END)."""
    hits: list[SecretHit] = []
    parts: list[str] = []
    cursor = 0
    while True:
        begin_match = _PEM_BEGIN.search(text, cursor)
        if begin_match is None:
            parts.append(text[cursor:])
            break
        end_match = _PEM_END.search(text, begin_match.end())
        if end_match is None:
            # Unterminated BEGIN: redact up to end of text and stop.
            parts.append(text[cursor:begin_match.start()])
            parts.append("[REDACTED:pem_private_key]")
            hits.append(
                SecretHit(
                    label="pem_private_key",
                    preview=_cap(text[begin_match.start() : begin_match.end()]),
                    start=begin_match.start(),
                    end=len(text),
                ),
            )
            cursor = len(text)
            break
        parts.append(text[cursor:begin_match.start()])
        parts.append("[REDACTED:pem_private_key]")
        hits.append(
            SecretHit(
                label="pem_private_key",
                preview=_cap(text[begin_match.start() : end_match.end()]),
                start=begin_match.start(),
                end=end_match.end(),
            ),
        )
        cursor = end_match.end()
    return "".join(parts), hits


class SecretRedactor:
    """Replace credential spans with ``[REDACTED:<label>]``.

    The redactor is stateless — instantiation is cheap; call
    :meth:`redact` repeatedly with arbitrary input. Pattern compilation
    happens at module import; the class only owns the dispatch logic.
    """

    def redact(self, text: str) -> tuple[str, list[SecretHit]]:
        """Return ``(new_text, hits)`` after applying all patterns.

        ``new_text`` is the input with all matched spans replaced.
        ``hits`` is the per-pattern list of :class:`SecretHit` (one
        entry per match, with label + preview + offset).
        """
        hits: list[SecretHit] = []
        # PEM spans multiple lines; handle first so subsequent
        # patterns don't accidentally match BEGIN/END lines.
        text, pem_hits = _redact_pem(text)
        hits.extend(pem_hits)
        for label, pattern in _PATTERNS:
            new_text, new_hits = _match(label, pattern, text)
            if new_hits:
                text = new_text
                hits.extend(new_hits)
        return text, hits