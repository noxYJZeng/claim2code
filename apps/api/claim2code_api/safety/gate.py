"""SafetyGate — composes SecretRedactor + PromptInjectionDetector.

This is the single SDK-seam wrapper used by ``LLMClient``. Every
production LLM call (extract_claims, verdict_for_claim,
extract_comparison) flows through ``wrap_messages`` immediately
before ``chat.completions.create(...)``.

Contract:

* ``wrap_messages(messages, *, kind) -> GateResult``
* Walks every message's ``content`` field (string OR list-of-parts).
* Runs ``SecretRedactor.redact`` (mutates text) and
  ``PromptInjectionDetector.detect`` (read-only).
* Returns the redacted message list + a violations list (empty when
  clean). Never raises on detection — audit log only.

The ``kind`` parameter is the call-site label (``"extract_claims"``,
``"verdict"``, ``"compare"``); it threads through to the audit log so
operators can correlate violations with the producing LLM method.
"""
from __future__ import annotations

from typing import Any

from claim2code_api.observability import get_tracer
from claim2code_api.observability.meter import record_safety_violation
from claim2code_api.safety.detector import PromptInjectionDetector
from claim2code_api.safety.exceptions import (
    GateResult,
    SafetyViolation,
)
from claim2code_api.safety.redactor import SecretRedactor

MessageDict = dict[str, Any]


class SafetyGate:
    """Compose redactor + detector into a single SDK-seam wrapper."""

    def __init__(
        self,
        *,
        redactor: SecretRedactor | None = None,
        detector: PromptInjectionDetector | None = None,
    ) -> None:
        self._redactor = redactor or SecretRedactor()
        self._detector = detector or PromptInjectionDetector()

    def wrap_messages(
        self,
        messages: list[MessageDict],
        *,
        kind: str,
    ) -> GateResult:
        """Redact secrets in ``messages``; collect violations.

        ``messages`` is the OpenAI-style list of role/content dicts.
        The returned list is a deep-ish copy: the outer list is new,
        but message dicts are shallow-copied and the ``content`` field
        is replaced with the redacted text (so callers do not see
        surprise mutation of their input).

        ``kind`` is a free-form label for the audit log (e.g.
        ``"extract_claims"``).

        Phase 9: wraps the work in a ``safety.gate_wrap`` child span.
        The span auto-parents under the current LLM span (because
        ``wrap_messages`` is called immediately before
        ``chat.completions.create`` on the same call stack). Each
        detected violation is also recorded on the
        ``safety_violations_total`` counter, labelled by pattern.
        """
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span(
            "safety.gate_wrap",
            attributes={
                "safety.kind": kind,
                "safety.n_messages": len(messages),
            },
        ) as span:
            new_messages: list[MessageDict] = []
            violations: list[SafetyViolation] = []
            # Aggregate per-label so a single SafetyViolation carries
            # span_count = N (matches for the same pattern in the same
            # wrap_messages call).
            secret_per_label: dict[str, list[str]] = {}
            injection_per_label: dict[str, list[str]] = {}

            for msg in messages:
                new_msg = dict(msg)
                content = msg.get("content")
                if isinstance(content, str):
                    redacted, hits = self._redactor.redact(content)
                    for h in hits:
                        secret_per_label.setdefault(h.label, []).append(h.preview)
                    # Detector: no mutation.
                    for inj in self._detector.detect(content):
                        injection_per_label.setdefault(inj.label, []).append(inj.preview)
                    new_msg["content"] = redacted
                elif isinstance(content, list):
                    # Multimodal: each part may be a dict with a "text"
                    # field. Walk text parts only; non-text (image_url)
                    # pass through untouched.
                    new_parts: list[Any] = []
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and isinstance(part.get("text"), str)
                        ):
                            redacted, hits = self._redactor.redact(part["text"])
                            for h in hits:
                                secret_per_label.setdefault(h.label, []).append(h.preview)
                            for inj in self._detector.detect(part["text"]):
                                injection_per_label.setdefault(inj.label, []).append(
                                    inj.preview,
                                )
                            new_part = dict(part)
                            new_part["text"] = redacted
                            new_parts.append(new_part)
                        else:
                            new_parts.append(part)
                    new_msg["content"] = new_parts
                # If content is None or any other type, pass through.
                new_messages.append(new_msg)

            for label, previews in secret_per_label.items():
                span_count = len(previews)
                violations.append(
                    SafetyViolation(
                        kind="secret",
                        pattern_label=label,
                        span_count=span_count,
                        redacted_preview=previews[0],
                    ),
                )
                record_safety_violation(
                    pattern_label=label, span_count=span_count,
                )
            for label, previews in injection_per_label.items():
                span_count = len(previews)
                violations.append(
                    SafetyViolation(
                        kind="injection",
                        pattern_label=label,
                        span_count=span_count,
                        redacted_preview=previews[0],
                    ),
                )
                record_safety_violation(
                    pattern_label=label, span_count=span_count,
                )

            span.set_attribute("safety.violations.secret_count", len(secret_per_label))
            span.set_attribute(
                "safety.violations.injection_count", len(injection_per_label),
            )
            span.set_attribute("safety.violations.total", len(violations))

            return GateResult(messages=new_messages, violations=violations)