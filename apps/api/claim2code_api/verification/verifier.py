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
from claim2code_api.repos.schemas import RetrieveMatch
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


def _with_candidate_paths(
    evidence: list[CodeEvidence], candidates: list[RetrieveMatch],
) -> list[CodeEvidence]:
    """Take each citation's file path from the candidate it cites.

    The LLM copies the path by hand and small models garble it (e.g.
    ``"utils.py:128-168"``); the cited symbol's own path is the truth.
    """
    paths = {c.symbol_id: c.file_path for c in candidates}
    return [
        e.model_copy(update={"file_path": paths[e.symbol_id]})
        if e.symbol_id in paths and paths[e.symbol_id] != e.file_path else e
        for e in evidence
    ]


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
    llm_emitted = llm_verdict.discrepancies if llm_verdict else []
    if llm_verdict is not None and not llm_verdict.code_evidence:
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
    code_evidence = _with_candidate_paths(
        llm_verdict.code_evidence if llm_verdict else [], candidates,
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
    for c in candidates:
        fields = (c.signature, c.docstring, c.source, c.class_context, c.callee_context)
        if any(needle in _squash(f) for f in fields if f):
            return True
    return False


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