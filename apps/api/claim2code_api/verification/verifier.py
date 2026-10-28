"""Pure-function composition of a ``Verification``.

This module separates *pure* verification-shape assembly from
*I/O-bound* orchestration. ``compose_verification`` is a
function of its inputs — no DB, no LLM, no network — so it is
straightforward to unit-test against the acceptance criterion.

``extract_verdict`` is the thin wrapper that pulls together
the deterministic checkers' output with the optional LLM
verdict and calls ``compose_verification``. The workflow
(``verification/workflow.py``) is responsible for retrieving
candidates via Phase 5's ``hybrid_ranker`` and feeding them
in.
"""
from __future__ import annotations

from claim2code_api.papers.schemas import Claim
from claim2code_api.repos.schemas import RetrieveMatch, SourceSpan
from claim2code_api.verification.discrepancies import (
    determine_status,
    merge_discrepancies,
)
from claim2code_api.verification.hyperparameter_checker import (
    HyperparameterChecker,
)
from claim2code_api.verification.normalizers import (
    DatasetNormalizer,
    MetricNameComparator,
    ModelNameNormalizer,
)
from claim2code_api.verification.schemas import (
    CodeEvidence,
    Discrepancy,
    Verification,
)

# Module-level singletons — these normalizers load JSON config at
# import time, so reusing them across calls is cheaper than
# rebuilding the variant->canonical map on every verification.
_HYPERPARAMETER_CHECKER = HyperparameterChecker()
_MODEL_NORMALIZER = ModelNameNormalizer()
_DATASET_NORMALIZER = DatasetNormalizer()
_METRIC_COMPARATOR = MetricNameComparator()


def extract_verdict(
    *,
    claim: Claim,
    candidates: list[RetrieveMatch],
    llm_verdict: Verification | None,
) -> Verification:
    """Build the final ``Verification`` from inputs + LLM verdict.

    Runs the four deterministic checkers, merges with any LLM-
    emitted discrepancies, then defers to
    :func:`compose_verification` for the final shape.

    The LLM may be ``None`` only when the workflow decided to
    skip it (e.g. a HIGH-severity deterministic discrepancy
    already produced a CONTRADICTED verdict). In that case the
    final status is determined from the discrepancies alone.
    """
    deterministic_discrepancies = run_deterministic_checkers(
        claim=claim, candidates=candidates,
    )
    return compose_verification(
        claim=claim,
        candidates=candidates,
        deterministic_discrepancies=deterministic_discrepancies,
        llm_verdict=llm_verdict,
    )


def _validated_code_evidence(
    evidence: list[CodeEvidence], candidates: list[RetrieveMatch],
) -> list[CodeEvidence]:
    """Keep citations whose quotes occur in source the verifier was shown.

    The located sources are each candidate's body and the setup/callee
    methods attached to it as context (``context_spans``), all read
    from the clone with their real file and line range. A citation is
    kept at its stated lines when the quote is there. Otherwise the
    quote is looked up across all located sources: a model often copies
    the right code under the wrong candidate's ``symbol_id`` or a few
    lines off. If it occurs, the citation is moved to the occurrence
    nearest its stated lines (same file first); a tie is ambiguous and
    dropped. A quote that occurs nowhere is dropped, so invented code
    never survives. Free-form text without a location (signature,
    docstring) cannot establish one, and missing source fails closed.
    """
    spans = _located_spans(candidates)
    valid: list[CodeEvidence] = []
    seen = set()
    for citation in evidence:
        quote = _squash(citation.quote)
        if not quote:
            continue
        located = _locate(citation, quote, spans)
        if located is None:
            continue
        span, start, end = located
        identity = (span.file_path, start, end, quote)
        if identity in seen:
            continue
        seen.add(identity)
        valid.append(citation.model_copy(update={
            "symbol_id": span.symbol_id, "file_path": span.file_path,
            "start_line": start, "end_line": end,
        }))
    return valid


def _located_spans(candidates: list[RetrieveMatch]) -> list[SourceSpan]:
    spans: list[SourceSpan] = []
    for c in candidates:
        if c.source:
            spans.append(SourceSpan(
                symbol_id=c.symbol_id, file_path=c.file_path,
                start_line=c.start_line, end_line=c.end_line, name=c.name, source=c.source,
            ))
        spans.extend(c.context_spans)
    return spans


def _locate(
    citation: CodeEvidence, quote: str, spans: list[SourceSpan],
) -> tuple[SourceSpan, int, int] | None:
    """Where ``quote`` sits: the stated lines if it is there, else the nearest occurrence."""
    # A model sometimes appends the line range to the path ("m.py:10-12").
    cited_file = citation.file_path.split(":", 1)[0]
    for span in spans:
        if span.file_path != cited_file or not (
            span.start_line <= citation.start_line <= citation.end_line <= span.end_line
        ):
            continue
        lines = span.source.splitlines()
        first = citation.start_line - span.start_line
        last = citation.end_line - span.start_line + 1
        if 0 <= first < last <= len(lines) and quote in _squash("\n".join(lines[first:last])):
            return span, citation.start_line, citation.end_line
    found: dict[tuple[str, int, int], SourceSpan] = {}
    for span in spans:
        for start, end in _occurrences(quote, span):
            found.setdefault((span.file_path, start, end), span)
    if not found:
        return None
    same_file = [k for k in found if k[0] == cited_file]
    keys = same_file or list(found)
    distance = {k: abs(k[1] - citation.start_line) for k in keys}
    best = min(distance.values())
    nearest = [k for k in keys if distance[k] == best]
    if len(nearest) != 1:
        return None
    _, start, end = nearest[0]
    return found[nearest[0]], start, end


def _occurrences(quote: str, span: SourceSpan) -> list[tuple[int, int]]:
    """Smallest line ranges of ``span`` whose text contains ``quote``."""
    lines = span.source.splitlines()
    out: list[tuple[int, int]] = []
    for i in range(len(lines)):
        if quote not in _squash("\n".join(lines[i:])):
            break  # no later start can match either
        for j in range(i + 1, len(lines) + 1):
            if quote in _squash("\n".join(lines[i:j])):
                # Minimal: dropping the first line loses the match.
                if not (j - i > 1 and quote in _squash("\n".join(lines[i + 1:j]))):
                    out.append((span.start_line + i, span.start_line + j - 1))
                break
    return out


def compose_verification(
    *,
    claim: Claim,
    candidates: list[RetrieveMatch],
    deterministic_discrepancies: list[Discrepancy],
    llm_verdict: Verification | None,
) -> Verification:
    """Assemble the final ``Verification``.

    Inputs:
      - ``claim``: the Phase 2 claim being verified.
      - ``candidates``: Phase 5 retrieval results (may be empty).
      - ``deterministic_discrepancies``: output of the four
        checkers (Hyperparameter + 3 normalizers).
      - ``llm_verdict``: the LLM's structured-output
        ``Verification``, or ``None`` if the workflow skipped
        the LLM call.

    The status decision (the 4-status rule) is delegated to
    :func:`determine_status` from :mod:`discrepancies`.

    The LLM's ``code_evidence``, ``hyperparameters``, and any
    additional ``discrepancies`` are merged in. The LLM's
    ``explanation`` is preserved when present; otherwise we
    synthesise one from the discrepancies ("Claim X; Y
    discrepancies found; no LLM verdict").
    """
    submitted_evidence = llm_verdict.code_evidence if llm_verdict else []
    code_evidence = _validated_code_evidence(submitted_evidence, candidates)
    llm_emitted = llm_verdict.discrepancies if llm_verdict else []
    if not code_evidence:
        # Without code evidence the verdict is UNVERIFIED by the
        # structural guard; LLM "discrepancies" there are just "no
        # code found" restated (often as HIGH), which reads as a real
        # mismatch in the report. Keep only the deterministic ones.
        llm_emitted = []
    llm_emitted = [d for d in llm_emitted if _quote_in_candidates(d.code_quote, candidates)]
    all_discrepancies = merge_discrepancies(
        deterministic=deterministic_discrepancies,
        llm_emitted=llm_emitted,
    )
    hyperparameters = llm_verdict.hyperparameters if llm_verdict else []
    explanation = (
        llm_verdict.explanation
        if llm_verdict
        else _synthesise_explanation(
            deterministic_discrepancies, candidates,
        )
    )
    confidence = llm_verdict.confidence if llm_verdict else 0.0
    if submitted_evidence and not code_evidence:
        explanation = "No source-validated code citation was retained; evidence is insufficient."
        confidence = 0.0

    status = determine_status(
        has_code_evidence=bool(code_evidence),
        discrepancies=all_discrepancies,
        llm_status=llm_verdict.status if llm_verdict else None,
    )

    return Verification(
        claim_id=claim.claim_id,
        status=status,
        confidence=confidence,
        explanation=explanation,
        paper_evidence=claim.paper_evidence,
        code_evidence=code_evidence,
        discrepancies=all_discrepancies,
        hyperparameters=hyperparameters,
    )


# ------------------------------------------------------------
# Internals
# ------------------------------------------------------------


def run_deterministic_checkers(
    *,
    claim: Claim,
    candidates: list[RetrieveMatch],
) -> list[Discrepancy]:
    """Run the four deterministic checkers and concatenate their output.

    Public — the workflow (``verification/workflow.py``) calls
    this directly so it can decide whether to skip the LLM
    call based on whether any HIGH-severity discrepancy was
    produced.
    """
    hp = _HYPERPARAMETER_CHECKER.compare(
        claim_text=claim.text,
        paper_evidence_quote=claim.paper_evidence.quote,
        candidates=candidates,
    )
    model = _MODEL_NORMALIZER.compare(
        claim_text=claim.text, candidates=candidates,
    )
    dataset = _DATASET_NORMALIZER.compare(
        claim_text=claim.text, candidates=candidates,
    )
    metric = _METRIC_COMPARATOR.compare(
        claim_text=claim.text, candidates=candidates,
    )
    return [*hp, *model, *dataset, *metric]


def _quote_in_candidates(quote: str | None, candidates: list[RetrieveMatch]) -> bool:
    """True when ``quote`` occurs (whitespace-insensitively) in a candidate.

    An LLM discrepancy must point at code that conflicts with the
    claim; one without such a quote only says the code is silent,
    which rule 7 of the verify prompt says is not a discrepancy.
    """
    needle = _squash(quote or "")
    if not needle:
        return False
    return any(needle in _squash(span.source) for span in _located_spans(candidates))


def _squash(text: str) -> str:
    return " ".join(text.split())


def _synthesise_explanation(
    deterministic_discrepancies: list[Discrepancy],
    candidates: list[RetrieveMatch],
) -> str:
    """Build a one-paragraph explanation when the LLM was skipped."""
    parts: list[str] = []
    if not candidates:
        parts.append("No code candidates retrieved.")
    if deterministic_discrepancies:
        n = len(deterministic_discrepancies)
        parts.append(
            f"{n} discrepancy{'s' if n != 1 else ''} "
            f"detected deterministically."
        )
    if not parts:
        return "No LLM verdict and no discrepancies; defaulting."
    return " ".join(parts)
