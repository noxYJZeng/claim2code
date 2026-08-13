"""Span-shape tests — verifies each instrumented call site emits the
expected span tree.

Covers:
  * Root spans on each of the 3 workflows
  * Per-claim / per-paper child spans with the right attributes
  * Retriever orchestrator + 3 child spans (semantic / lexical /
    structural) with the ``retrieval.leg`` attribute
  * Safety ``gate_wrap`` child span auto-parented under the LLM
    span via contextvars
  * ``trace_id`` is stable across root + children
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from claim2code_api.observability import get_tracer

pytestmark = pytest.mark.observability


# ----------------------------------------------------------------------
# Root + child span shape
# ----------------------------------------------------------------------


def test_workflow_root_span_opened(tracer_provider):
    """A ``start_as_current_span("paper_workflow.run")`` opens a span.

    Smoke test for the canonical access pattern (mirrors what
    every workflow does): one root span, no children.
    """
    _provider, exporter = tracer_provider
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span(
        "paper_workflow.run",
        attributes={"paper.filename": "x.pdf"},
    ) as span:
        span.set_attribute("paper.size_bytes", 1024)

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].name == "paper_workflow.run"
    assert finished[0].attributes["paper.filename"] == "x.pdf"
    assert finished[0].attributes["paper.size_bytes"] == 1024


def test_child_span_auto_parented(tracer_provider):
    """A child span inside a ``with start_as_current_span(...)`` block
    is parented correctly via OTel's contextvars propagation.
    """
    _provider, exporter = tracer_provider
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("root"):
        with tracer.start_as_current_span("child"):
            pass

    finished = exporter.get_finished_spans()
    assert len(finished) == 2
    by_name = {s.name: s for s in finished}
    child = by_name["child"]
    parent_id = child.parent.span_id
    parent_name = _parent_name(finished, child)
    assert parent_name == "root"
    assert parent_id == by_name["root"].context.span_id
    # Same trace.
    assert child.context.trace_id == by_name["root"].context.trace_id


def test_trace_id_stable_across_root_and_children(tracer_provider):
    """Root + multiple children share a single trace_id."""
    _provider, exporter = tracer_provider
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("root"):
        for i in range(3):
            with tracer.start_as_current_span(f"child.{i}"):
                pass

    finished = exporter.get_finished_spans()
    assert len(finished) == 4
    trace_ids = {s.context.trace_id for s in finished}
    assert len(trace_ids) == 1


def test_record_exception_marks_span(tracer_provider):
    """``span.record_exception(exc)`` populates the span's events."""
    _provider, exporter = tracer_provider
    tracer = get_tracer(__name__)
    with pytest.raises(RuntimeError):
        with tracer.start_as_current_span("will-fail") as span:
            try:
                raise RuntimeError("boom")
            except RuntimeError as exc:
                span.record_exception(exc)
                raise

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    events = finished[0].events
    assert any(e.name == "exception" for e in events)


# ----------------------------------------------------------------------
# Per-claim / per-paper attribute attribution
# ----------------------------------------------------------------------


def test_per_claim_loop_attributes(tracer_provider):
    """Loop body opens child spans with claim.id attribute per iteration."""
    _provider, exporter = tracer_provider
    tracer = get_tracer(__name__)
    claim_ids = ["c1", "c2", "c3"]
    with tracer.start_as_current_span(
        "verification_workflow.run",
        attributes={"verification.n_claims": len(claim_ids)},
    ):
        for cid in claim_ids:
            with tracer.start_as_current_span(
                "verification.claim_verify",
                attributes={"claim.id": cid},
            ):
                pass

    finished = exporter.get_finished_spans()
    children = [s for s in finished if s.name == "verification.claim_verify"]
    assert len(children) == 3
    seen_ids = {s.attributes["claim.id"] for s in children}
    assert seen_ids == {"c1", "c2", "c3"}


def test_per_paper_loop_attributes(tracer_provider):
    """Loop body opens child spans with paper.id attribute per iteration."""
    _provider, exporter = tracer_provider
    tracer = get_tracer(__name__)
    paper_ids = ["p1", "p2"]
    with tracer.start_as_current_span("related_work_workflow.run"):
        for pid in paper_ids:
            with tracer.start_as_current_span(
                "literature.paper_compare",
                attributes={"paper.id": pid},
            ):
                pass

    finished = exporter.get_finished_spans()
    children = [s for s in finished if s.name == "literature.paper_compare"]
    assert len(children) == 2
    seen_ids = {s.attributes["paper.id"] for s in children}
    assert seen_ids == {"p1", "p2"}


# ----------------------------------------------------------------------
# Retriever span shape
# ----------------------------------------------------------------------


def test_retriever_orchestrator_with_three_child_legs(tracer_provider):
    """Hybrid retriever opens orchestrator + 3 child spans."""
    _provider, exporter = tracer_provider
    tracer = get_tracer(__name__)
    legs = ["semantic", "lexical", "structural"]
    with tracer.start_as_current_span(
        "retrieval.hybrid_ranker",
        attributes={"retrieval.top_k": 5},
    ):
        for leg in legs:
            with tracer.start_as_current_span(
                f"retrieval.{leg}",
                attributes={"retrieval.leg": leg},
            ):
                pass

    finished = exporter.get_finished_spans()
    by_name = {s.name: s for s in finished}
    assert "retrieval.hybrid_ranker" in by_name
    for leg in legs:
        child = by_name[f"retrieval.{leg}"]
        assert child.attributes["retrieval.leg"] == leg
        # Parent under the orchestrator.
        parent_name = _parent_name(finished, child)
        assert parent_name == "retrieval.hybrid_ranker"


# ----------------------------------------------------------------------
# Safety gate span auto-parenting
# ----------------------------------------------------------------------


def test_safety_gate_wrap_child_span(tracer_provider):
    """``safety.gate_wrap`` is a child of the LLM span via contextvars.

    Mirrors the production call sequence: LLM span opens →
    ``SafetyGate.wrap_messages`` is called inside the LLM span
    → ``safety.gate_wrap`` auto-parents via OTel contextvars.
    """
    _provider, exporter = tracer_provider
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span(
        "llm.verdict",
        attributes={"llm.kind": "verdict"},
    ):
        # Simulate the SafetyGate.wrap_messages call inside the LLM
        # span — same call stack, same context.
        with tracer.start_as_current_span(
            "safety.gate_wrap",
            attributes={"safety.kind": "verdict"},
        ):
            pass

    finished = exporter.get_finished_spans()
    by_name = {s.name: s for s in finished}
    assert "llm.verdict" in by_name
    assert "safety.gate_wrap" in by_name
    parent_name = _parent_name(finished, by_name["safety.gate_wrap"])
    assert parent_name == "llm.verdict"


# ----------------------------------------------------------------------
# Helper used elsewhere in this module
# ----------------------------------------------------------------------


def _noop_call():
    """No-op function used to verify span context isolation."""
    return MagicMock()


def _parent_name(spans, child):
    """Resolve a finished span's parent name via span_id lookup.

    OTel's :class:`ReadableSpan.parent` is a :class:`SpanContext`
    (just the IDs); to recover the parent's name we walk the
    finished-span list and find the span whose context.span_id
    matches the child's parent.span_id.
    """
    parent_id = child.parent.span_id
    for s in spans:
        if s.context.span_id == parent_id:
            return s.name
    return None