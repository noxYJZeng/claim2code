"""Reference-check workflow: pick a function, write inputs and a reference, run both.

For one claim:

  1. **Pick the function under test.** The symbol the request names,
     else the claim's cited code (from its verification), else the
     top retrieval hits. The first Python function, or method of a
     top-level class, wins. For an instance method the inputs author
     also writes ``make_instance()`` from the class's ``__init__``.
  2. **Inputs** — model call with the function body in view.
  3. **Reference** — model call with the paper passage and the
     signature, never the body.
  4. **Run** both in the sandbox and compare outputs per input.
  5. **Persist** to ``AnalysisState.reference_checks`` (one entry per
     claim, replaced on re-run).

A model failure is a result (``ERROR`` with the message), not an
exception, so the UI can show it next to the claim like any other.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import UUID

from claim2code_api.app.config import get_settings
from claim2code_api.db.models import AnalysisState, CodeSymbol, Repository
from claim2code_api.llm.client import get_llm_client
from claim2code_api.observability import get_tracer
from claim2code_api.papers.schemas import Claim
from claim2code_api.reference import prompts
from claim2code_api.reference.exceptions import (
    ReferenceCheckLLMError,
    ReferenceCheckNotFoundError,
)
from claim2code_api.reference.sandbox import SandboxRun, run_harness
from claim2code_api.reference.schemas import (
    CaseOutcome,
    ReferenceCase,
    ReferenceCheck,
    ReferenceCheckStatus,
    ReferenceTarget,
)
from claim2code_api.repos.schemas import CodeSymbolKind, RetrieveMatch

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_CALLABLE_KINDS = (CodeSymbolKind.FUNCTION.value, CodeSymbolKind.METHOD.value)
_RETRIEVAL_K = 8


def run_reference_check(
    *, db: Session, analysis_id: UUID, claim_id: str, symbol_id: UUID | None = None,
) -> ReferenceCheck:
    """Run one reference check and persist it.

    :raises ReferenceCheckNotFoundError: unknown analysis or claim.
    """
    state = (
        db.query(AnalysisState)
        .filter(AnalysisState.analysis_id == analysis_id)
        .one_or_none()
    )
    if state is None:
        raise ReferenceCheckNotFoundError(f"analysis {analysis_id} not found")
    raw_claim = next((c for c in state.claims or [] if c.get("claim_id") == claim_id), None)
    if raw_claim is None:
        raise ReferenceCheckNotFoundError(f"claim_id {claim_id!r} not found in analysis {analysis_id}")
    claim = Claim.model_validate(raw_claim)

    with get_tracer(__name__).start_as_current_span(
        "reference_check.run", attributes={"claim.id": claim_id},
    ):
        check = _check(db, state, claim, symbol_id)

    entry = json.loads(check.model_dump_json())
    state.reference_checks = [
        e for e in state.reference_checks or [] if e.get("claim_id") != claim_id
    ] + [entry]
    db.flush()
    logger.info(
        "reference_check analysis=%s claim=%s status=%s cases=%d",
        analysis_id, claim_id, check.status.value, len(check.cases),
    )
    return check


def _check(db: Session, state: AnalysisState, claim: Claim, symbol_id: UUID | None) -> ReferenceCheck:
    def result(status: ReferenceCheckStatus, explanation: str, **fields: Any) -> ReferenceCheck:
        return ReferenceCheck(
            claim_id=claim.claim_id, status=status, explanation=explanation,
            generated_at=datetime.now(UTC), **fields,
        )

    repo = (
        db.query(Repository)
        .filter(Repository.analysis_id == state.analysis_id)
        .one_or_none()
    )
    if repo is None or not repo.clone_path:
        return result(ReferenceCheckStatus.UNTESTABLE, "No repository is bound to this analysis.")

    picked = _pick_target(db, state, claim, repo.repository_id, symbol_id)
    if picked is None:
        return result(
            ReferenceCheckStatus.UNTESTABLE,
            "None of the code found for this claim is a Python function or method of a "
            "top-level class, so there is nothing to call.",
        )
    target, qualname = picked
    target_info = ReferenceTarget(
        symbol_id=target.symbol_id, file_path=target.file_path,
        start_line=target.start_line, end_line=target.end_line,
        qualname=qualname, signature=target.signature,
    )

    client = get_llm_client()
    try:
        inputs = client.reference_inputs(
            system=prompts.INPUTS_SYSTEM_PROMPT,
            payload=prompts.build_inputs_payload(claim=claim, target=target, qualname=qualname),
        )
        if not inputs.testable or not inputs.inputs_code.strip():
            return result(ReferenceCheckStatus.UNTESTABLE, inputs.reason, target=target_info)
        impl = client.reference_implementation(
            system=prompts.REFERENCE_SYSTEM_PROMPT,
            payload=prompts.build_reference_payload(
                claim=claim, target=target, qualname=qualname,
                inputs_code=inputs.inputs_code, input_notes=inputs.input_notes,
            ),
        )
    except ReferenceCheckLLMError as exc:
        return result(ReferenceCheckStatus.ERROR, f"The model call failed: {exc}", target=target_info)

    settings = get_settings()
    rtol, atol = min(max(impl.rtol, 0.0), 1.0), min(max(impl.atol, 0.0), 1.0)
    run = run_harness(
        python=settings.reference_check_python or sys.executable,
        repo_root=Path(repo.clone_path),
        module_file=target.file_path,
        qualname=qualname,
        reference_code=impl.reference_code,
        inputs_code=inputs.inputs_code,
        instance_code=inputs.instance_code,
        rtol=rtol,
        atol=atol,
        timeout_seconds=settings.reference_check_timeout_seconds,
    )
    status, explanation, cases = summarise(run, qualname, settings.reference_check_timeout_seconds)
    return result(
        status, explanation,
        target=target_info,
        reference_code=impl.reference_code,
        inputs_code=inputs.inputs_code,
        instance_code=inputs.instance_code,
        assumptions=impl.assumptions,
        rtol=rtol, atol=atol,
        cases=cases,
        isolation=run.isolation,
    )


def summarise(
    run: SandboxRun, qualname: str, timeout_seconds: int,
) -> tuple[ReferenceCheckStatus, str, list[ReferenceCase]]:
    """Status, one-paragraph explanation and per-input cases for a harness run."""
    report = run.report
    if report is None:
        if run.timed_out:
            return ReferenceCheckStatus.ERROR, f"The run did not finish within {timeout_seconds}s.", []
        last = run.stderr.strip().splitlines()[-1:] or ["no output"]
        return ReferenceCheckStatus.ERROR, f"The test process exited without a report: {last[0]}", []
    if report.get("stage") == "import":
        return ReferenceCheckStatus.ERROR, f"Could not import {qualname}: {report.get('error')}", []
    if report.get("stage") != "done":
        return (
            ReferenceCheckStatus.ERROR,
            f"The generated inputs or reference did not run: {report.get('error')}",
            [],
        )

    cases = [ReferenceCase.model_validate(c) for c in report.get("cases", [])]
    compared = [c for c in cases if c.outcome in (CaseOutcome.MATCH, CaseOutcome.MISMATCH)]
    mismatched = [c for c in compared if c.outcome == CaseOutcome.MISMATCH]
    target_errors = sum(c.outcome == CaseOutcome.TARGET_ERROR for c in cases)
    reference_errors = sum(c.outcome == CaseOutcome.REFERENCE_ERROR for c in cases)

    errors = [
        f"{_inputs(n)} in {where}"
        for n, where in ((target_errors, qualname), (reference_errors, "the reference"))
        if n
    ]
    tail = f" Raised an error and not compared: {', '.join(errors)}." if errors else ""

    if not compared:
        first = next((c for c in cases if c.detail), None)
        detail = f" First error: {first.detail}" if first else ""
        return ReferenceCheckStatus.ERROR, f"No input could be compared.{detail}", cases
    if mismatched:
        first = mismatched[0]
        why = f": {first.detail}" if first.detail else ""
        return (
            ReferenceCheckStatus.MISMATCH,
            f"{len(mismatched)} of {_inputs(len(compared))} gave a different output from "
            f"{qualname} than from the reference. First difference, input {first.index + 1}{why}.{tail}",
            cases,
        )
    return (
        ReferenceCheckStatus.MATCH,
        f"All {_inputs(len(compared))} gave the same output from {qualname} and from the "
        f"reference written from the paper.{tail}",
        cases,
    )


def _inputs(n: int) -> str:
    return f"{n} input" if n == 1 else f"{n} inputs"


def _pick_target(
    db: Session, state: AnalysisState, claim: Claim, repository_id: UUID, symbol_id: UUID | None,
) -> tuple[RetrieveMatch, str] | None:
    """The first callable candidate, with the dotted name the harness imports."""
    from claim2code_api.verification.workflow import _hybrid_to_retrieve_matches, _retrieve

    if symbol_id is not None:
        ids = [symbol_id]
    else:
        ids = _cited_symbol_ids(state, claim.claim_id)
        try:
            ids += [m.symbol_id for m in _retrieve(db, repository_id, claim.text, top_k=_RETRIEVAL_K)]
        except Exception as exc:  # noqa: BLE001 — cited code may still be enough
            logger.warning("reference_check retrieval failed claim=%s: %s", claim.claim_id, exc)
        ids = list(dict.fromkeys(ids))
    if not ids:
        if symbol_id is not None:
            raise ReferenceCheckNotFoundError(f"symbol {symbol_id} not found")
        return None

    rows = {
        s.symbol_id: s
        for s in db.query(CodeSymbol)
        .filter(CodeSymbol.symbol_id.in_(ids), CodeSymbol.repository_id == repository_id)
        .all()
    }
    if symbol_id is not None and symbol_id not in rows:
        raise ReferenceCheckNotFoundError(f"symbol {symbol_id} not found in this repository")
    for sid in ids:
        sym = rows.get(sid)
        if sym is None:
            continue
        qualname = _qualname(db, sym)
        if qualname is None:
            continue
        matches = _hybrid_to_retrieve_matches(
            db=db, matches=[SimpleNamespace(symbol_id=sid, final_score=1.0)],
            repository_id=repository_id, claim=claim.text,
        )
        if matches and matches[0].source:
            return matches[0], qualname
    return None


def _qualname(db: Session, sym: CodeSymbol) -> str | None:
    """``name`` or ``Class.name`` for a function or a method of a top-level class."""
    if sym.language != "python" or sym.kind not in _CALLABLE_KINDS:
        return None
    if not sym.file_path.endswith(".py"):
        return None
    if sym.parent_symbol_id is None:
        return sym.name if sym.kind == CodeSymbolKind.FUNCTION.value else None
    parent = db.get(CodeSymbol, sym.parent_symbol_id)
    if parent is None or parent.kind != CodeSymbolKind.CLASS.value or parent.parent_symbol_id is not None:
        return None  # nested in a function, or a nested class
    return f"{parent.name}.{sym.name}"


def _cited_symbol_ids(state: AnalysisState, claim_id: str) -> list[UUID]:
    for entry in state.verifications or []:
        if entry.get("claim_id") != claim_id:
            continue
        out = []
        for ev in entry.get("code_evidence") or []:
            try:
                out.append(UUID(str(ev.get("symbol_id"))))
            except (TypeError, ValueError):
                continue
        return out
    return []


__all__ = ["run_reference_check", "summarise"]
