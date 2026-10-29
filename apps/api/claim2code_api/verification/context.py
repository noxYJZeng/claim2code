"""Bounded source windows with real identities and line locations.

Long classes often hide the relevant implementation below the first prompt
page. Keep a readable prefix and add claim-focused windows, without inventing
source text or allowing the citation validator to inspect hidden source.
"""
from __future__ import annotations

import re

from claim2code_api.repos.schemas import RetrieveMatch, SourceSpan

SOURCE_CHAR_CAP = 3000
CONTEXT_CHAR_CAP = 5000
# Request-wide limits also apply to the expanded evidence-retry shortlist.
# Character budgets are intentionally conservative for a local 32k model;
# the model's exact tokenizer and output allowance remain provider-specific.
TOTAL_SOURCE_CHAR_CAP = 32000
TOTAL_METADATA_CHAR_CAP = 8000
MAX_CANDIDATES = 16
MAX_VERIFY_PAYLOAD_CHAR_CAP = 52000
_TOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+")
_STOP = frozenset({
    "the", "and", "for", "from", "with", "that", "this", "are", "uses", "use",
    "into", "each", "its", "can", "all", "both", "during", "which", "than",
    "code", "implementation", "paper", "claim", "model", "models",
})


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 3} - _STOP


def _slice(span: SourceSpan, first: int, last: int) -> SourceSpan:
    lines = span.source.splitlines()
    return span.model_copy(update={
        "start_line": span.start_line + first,
        "end_line": span.start_line + last - 1,
        "source": "\n".join(lines[first:last]),
    })


def _prefix(span: SourceSpan, cap: int) -> SourceSpan | None:
    lines = span.source.splitlines()
    used = 0
    count = 0
    for line in lines:
        cost = len(line) + (1 if count else 0)
        if used + cost > cap:
            break
        used += cost
        count += 1
    return _slice(span, 0, count) if count else None


def _windows(
    span: SourceSpan, terms: set[str], char_cap: int = SOURCE_CHAR_CAP,
) -> list[tuple[int, SourceSpan]]:
    """Candidate windows around useful lines, strongest matches first."""
    lines = span.source.splitlines()
    ranked = sorted(
        ((len(terms & _tokens(line)), i) for i, line in enumerate(lines)),
        key=lambda row: (-row[0], row[1]),
    )
    out: list[tuple[int, SourceSpan]] = []
    covered: set[int] = set()
    for score, i in ranked:
        if not score or i in covered:
            continue
        first, last = max(0, i - 4), min(len(lines), i + 8)
        window = _slice(span, first, last)
        # A very long generated line is not useful prompt context.
        if len(window.source) > char_cap:
            window = _slice(span, i, i + 1)
        if len(window.source) > char_cap:
            continue
        out.append((score, window))
        covered.update(range(first, last))
        if len(out) == 4:
            break
    return out


def _bounded_metadata(candidate: RetrieveMatch, cap: int) -> RetrieveMatch | None:
    # File paths and names retain their identities. An exceptionally long
    # identity is excluded rather than displayed under an invented path.
    remaining = cap - len(candidate.file_path) - len(candidate.name) - 256
    if remaining < 0:
        return None
    signature = candidate.signature[:min(512, remaining)]
    remaining -= len(signature)
    docstring = candidate.docstring[:min(1000, remaining)] if candidate.docstring else None
    return candidate.model_copy(update={"signature": signature, "docstring": docstring})


def prepare_candidates(
    candidates: list[RetrieveMatch], claim: str,
) -> list[RetrieveMatch]:
    """Return exactly the source blocks that can be shown within the budget.

    The workflow passes these same objects to the LLM and citation validator.
    Every retained block is a contiguous substring with its original line
    offset. Windows from a candidate body become located context when they
    lie beyond its prefix. Existing setup/callee/registry spans share the
    context budget; unlocated legacy context is never promoted to evidence.

    Source and metadata budgets apply to the entire request, with a fair
    source share for each candidate. Half a share is reserved for focused
    context, so a long prefix cannot crowd out a useful tail window. The
    existing PARTIAL status-composition contract is unaffected by budgeting.
    """
    terms = _tokens(claim)
    out: list[RetrieveMatch] = []
    selected = candidates[:MAX_CANDIDATES]
    if not selected:
        return []
    source_share = TOTAL_SOURCE_CHAR_CAP // len(selected)
    metadata_share = TOTAL_METADATA_CHAR_CAP // len(selected)
    prefix_cap = min(SOURCE_CHAR_CAP, source_share // 2)
    for original_candidate in selected:
        candidate = _bounded_metadata(original_candidate, metadata_share)
        if candidate is None:
            continue
        original = SourceSpan(
            symbol_id=candidate.symbol_id, file_path=candidate.file_path,
            start_line=candidate.start_line, end_line=candidate.end_line,
            name=candidate.name, source=candidate.source or "",
        )
        prefix = _prefix(original, prefix_cap) if original.source else None
        context_cap = min(CONTEXT_CHAR_CAP, source_share - len(prefix.source if prefix else ""))
        options: list[tuple[int, SourceSpan]] = []
        for span in candidate.context_spans:
            # A context header consumes part of its share. When a full
            # helper no longer fits, retain focused, contiguous windows.
            span_cap = min(SOURCE_CHAR_CAP, max(0, context_cap - len(span.file_path) - len(span.name) - 160))
            if len(span.source) <= span_cap:
                options.append((len(terms & _tokens(span.source)), span))
            else:
                options.extend(_windows(span, terms, span_cap))
                first = _prefix(span, span_cap)
                if first is not None:
                    options.append((len(terms & _tokens(first.source)), first))
        if original.source and (prefix is None or prefix.end_line < original.end_line):
            window_cap = max(0, context_cap - len(original.file_path) - len(original.name) - 160)
            options.extend(_windows(original, terms, min(SOURCE_CHAR_CAP, window_cap)))
        options.sort(key=lambda row: (-row[0], row[1].file_path, row[1].start_line))
        retained: list[SourceSpan] = []
        used = 0
        for _, span in options:
            already = [*retained, *([prefix] if prefix else [])]
            if any(
                prior.file_path == span.file_path
                and prior.start_line <= span.start_line <= span.end_line <= prior.end_line
                for prior in already
            ):
                continue
            # Include source-location headers in the prompt budget.
            cost = len(span.source) + len(span.file_path) + len(span.name) + 160
            if used + cost > context_cap:
                continue
            retained.append(span)
            used += cost
        out.append(candidate.model_copy(update={
            "source": prefix.source if prefix else None,
            "end_line": prefix.end_line if prefix else candidate.start_line,
            "context_spans": retained,
            "class_context": None,
            "callee_context": None,
        }))
    return out
