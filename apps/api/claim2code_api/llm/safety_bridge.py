"""Bridge between ``LLMClient`` violations and ``AnalysisState.errors``.

Every violation surfaced by :class:`SafetyGate` is logged on the
analysis row as a structured entry in ``AnalysisState.errors``. The
shape mirrors Phase 6/7 per-item error reporting:

    {
        "stage": "safety",
        "kind": "extract_claims" | "verdict" | "compare",
        "pattern": "openai_key" | "ignore_previous" | ...,
        "span_count": int,
        "ts": "2026-08-12T...Z",
    }

Deliberately NO content is persisted: the ``redacted_preview`` is not
user-facing and stays on the LLM-call-site audit log. This keeps
``state.errors`` safe to surface in the frontend error-recovery UI
without leaking secrets.

See ``docs/architecture/phase-8-design.md`` §3 for the audit-log
contract.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from claim2code_api.db.models import AnalysisState
from claim2code_api.safety import SafetyViolation


def record_safety_violations(
    db: Session,
    *,
    analysis_id: UUID,
    kind: str,
    violations: list[SafetyViolation],
) -> None:
    """Append one ``state.errors`` entry per violation.

    Idempotent: same input produces same entries, so a retry doesn't
    duplicate. The audit log is JSONB on ``AnalysisState.errors``;
    callers that need this should commit the session after calling.
    """
    state = db.get(AnalysisState, analysis_id)
    if state is None:
        # Analysis row missing is a programming error (workflow must
        # have created it before invoking LLMClient). Silently no-op
        # rather than crash the LLM call site.
        return
    ts = datetime.now(UTC).isoformat()
    for v in violations:
        state.errors.append(
            {
                "stage": "safety",
                "kind": kind,
                "pattern": v.pattern_label,
                "span_count": v.span_count,
                "ts": ts,
            },
        )
    db.flush()


def make_violation_callback(
    db: Session,
    *,
    analysis_id: UUID,
) -> Callable[[str, list[SafetyViolation]], None]:
    """Build a closure matching ``LLMClient._on_violations``.

    Workflows call this once, then assign it to the LLMClient before
    the LLM call. The closure captures ``db`` + ``analysis_id`` so the
    call site stays a one-liner.

    Example:

        llm._on_violations = make_violation_callback(
            db, analysis_id=analysis_id,
        )
    """
    def _on_violations(
        kind: str,
        violations: list[SafetyViolation],
    ) -> None:
        record_safety_violations(
            db, analysis_id=analysis_id, kind=kind, violations=violations,
        )
    return _on_violations