"""Verification layer — claim-to-code verdicts.

Given one ``Claim`` (Phase 2) and its top-K ``RetrieveMatch``
candidates (Phase 5), the verifier produces one
``Verification`` per claim: a 4-status verdict, paper evidence,
code evidence, and any discrepancies. The pipeline is:

  1. **Retrieve candidates** via Phase 5's ``hybrid_ranker``.
  2. **Deterministic checks** — ``HyperparameterChecker`` plus
     three name normalizers (``ModelNameNormalizer``,
     ``DatasetNormalizer``, ``MetricNameComparator``) emit
     discrepancies with no LLM involvement.
  3. **LLM verdict** — only when the deterministic checkers did
     not already produce a HIGH-severity discrepancy. The LLM
     reuses the same ``LLMClient`` as Phase 2's claim extractor
     (``extract_claims``); the new ``verdict_for_claim`` method
     adds the no-model-knowledge system prompt and Structured
     Outputs schema.
  4. **Compose** — ``compose_verification`` is a pure function;
    the structural guard coerces to ``UNVERIFIED`` when no
    ``code_evidence`` was cited.

Acceptance criterion (``docs/architecture/phase-roadmap.md:128``):
every verification has paper evidence AND code evidence, or
``UNVERIFIED``.

Persistence: verifications are written to ``AnalysisState.verifications``
(JSONB) — the column was pre-staged in Phase 1's migration
``0001_init.py`` specifically so Phase 6 could avoid a DDL change.
"""
from __future__ import annotations

from claim2code_api.verification.schemas import (
    CodeEvidence,
    Discrepancy,
    DiscrepancySeverity,
    DiscrepancyType,
    HyperparameterEvidence,
    Verification,
    VerificationListResponse,
    VerificationRunRequest,
    VerificationRunResponse,
    VerificationStatus,
)

__all__ = [
    "CodeEvidence",
    "Discrepancy",
    "DiscrepancySeverity",
    "DiscrepancyType",
    "HyperparameterEvidence",
    "Verification",
    "VerificationListResponse",
    "VerificationRunRequest",
    "VerificationRunResponse",
    "VerificationStatus",
]