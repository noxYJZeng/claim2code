"""Discrepancy classification — apply the 4-status rule.

The deterministic checkers (``HyperparameterChecker``,
``ModelNameNormalizer``, ``DatasetNormalizer``,
``MetricNameComparator``) each emit their own :class:`Discrepancy`
objects with appropriate ``type`` and ``severity``. The
:class:`LLMClient.verdict_for_claim` call (commit 5) emits its
own via the LLM's structured output. This module's job is the
final status decision that combines all of them.

The 4-status rule (per ``docs/architecture/phase-6-design.md`` §2.5):

  | Status             | code_evidence (>=1) | Discrepancies            |
  |--------------------|---------------------|--------------------------|
  | VERIFIED           | yes                 | none                     |
  | PARTIALLY_VERIFIED | yes                 | >=1 LOW or MEDIUM        |
  | CONTRADICTED       | yes                 | >=1 HIGH OR LLM=CONTRADICTED |
  | UNVERIFIED         | NO (structural)     | n/a                      |

The structural guard (``no code_evidence -> UNVERIFIED``)
trumps everything else — even if the LLM says VERIFIED, an
empty ``code_evidence`` list coerces to UNVERIFIED. This makes
the acceptance criterion (``paper_evidence AND code_evidence
OR UNVERIFIED``) testable as a single assertion on the
returned :class:`Verification`.

Discrepancies from deterministic checkers AND the LLM are
merged (no deduplication needed today — each checker emits
distinct types; future work may want fuzzy-dedup).
"""
from __future__ import annotations

from claim2code_api.verification.schemas import (
    Discrepancy,
    DiscrepancySeverity,
    VerificationStatus,
)


def merge_discrepancies(
    *,
    deterministic: list[Discrepancy],
    llm_emitted: list[Discrepancy] | None,
) -> list[Discrepancy]:
    """Combine deterministic + LLM-emitted discrepancies.

    Order: deterministic first (so the deterministic HIGH-severity
    flags appear at the top of any UI list), then LLM-emitted.

    No dedup today — each deterministic checker emits a
    distinct ``DiscrepancyType``, and the LLM is told in the
    system prompt not to re-emit types the deterministic
    checkers already produced. A future phase may want
    fuzzy-dedup on ``detail`` field.
    """
    if not llm_emitted:
        return list(deterministic)
    return [*deterministic, *llm_emitted]


def determine_status(
    *,
    has_code_evidence: bool,
    discrepancies: list[Discrepancy],
    llm_status: VerificationStatus | None,
) -> VerificationStatus:
    """Apply the 4-status rule.

    Returns the final :class:`VerificationStatus` for one
    verification. The structural guard
    (``has_code_evidence == False -> UNVERIFIED``) is checked
    first and overrides any other signal — this is how the
    acceptance criterion
    (``paper_evidence AND code_evidence OR UNVERIFIED``) is
    enforced.
    """
    # Structural guard — every verification must have paper
    # evidence AND code evidence, or be UNVERIFIED.
    if not has_code_evidence:
        return VerificationStatus.UNVERIFIED

    has_high = any(
        d.severity == DiscrepancySeverity.HIGH for d in discrepancies
    )
    llm_contradicted = llm_status == VerificationStatus.CONTRADICTED

    if has_high or llm_contradicted:
        return VerificationStatus.CONTRADICTED

    # Any LOW or MEDIUM (or AMBIGUOUS) discrepancy demotes VERIFIED.
    if discrepancies:
        return VerificationStatus.PARTIALLY_VERIFIED

    # No discrepancies and we have code evidence. PARTIALLY_VERIFIED
    # needs >=1 LOW/MEDIUM discrepancy (phase-6-design.md §2.5), so an
    # LLM "partial" with nothing left to show for it — the code is
    # merely silent on part of the claim, or its discrepancies were
    # dropped as unquoted — is VERIFIED.
    if llm_status in (
        VerificationStatus.VERIFIED,
        VerificationStatus.PARTIALLY_VERIFIED,
    ):
        return VerificationStatus.VERIFIED

    # Default: LLM said UNVERIFIED, or no LLM and no discrepancies.
    return VerificationStatus.UNVERIFIED