"""Unit tests for :mod:`claim2code_api.verification.verifier`
and the new ``LLMClient.verdict_for_claim`` method.

Coverage for ``compose_verification``:

  - Happy path: candidate + LLM VERIFIED + code_evidence → VERIFIED.
  - Structural guard: LLM VERIFIED + no code_evidence → UNVERIFIED.
  - Deterministic HIGH + LLM None + code evidence → CONTRADICTED
    (we still have code evidence here because the test passes
    llm_verdict with code_evidence when LLM was synthesized by
    the caller; the pure-function path tests this).
  - LLM VERIFIED + MEDIUM discrepancy → PARTIALLY_VERIFIED.

Coverage for ``LLMClient.verdict_for_claim``:

  - happy path: response validates into ``Verification``.
  - chat.completions.create called once with the verify schema
    (strict json_schema, name="verification").
  - claim_id mismatch → ``VerificationLLMError``.
  - OpenAI timeout → ``VerificationLLMError`` (not ClaimExtractionError).
  - Schema validation failure → ``VerificationLLMError``.
  - Empty claim_id in response → ``VerificationLLMError``.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
)

from claim2code_api.llm.client import LLMClient
from claim2code_api.papers.schemas import Claim, ClaimType, PaperEvidence
from claim2code_api.repos.schemas import CodeLanguage, CodeSymbolKind, RetrieveMatch, SourceSpan
from claim2code_api.verification.exceptions import VerificationLLMError
from claim2code_api.verification.schemas import (
    CodeEvidence,
    Discrepancy,
    DiscrepancySeverity,
    DiscrepancyType,
    Verification,
    VerificationStatus,
)
from claim2code_api.verification.verifier import (
    compose_verification,
    extract_verdict,
    run_deterministic_checkers,
)

# ============================================================
# Fixtures
# ============================================================


def _paper_evidence() -> PaperEvidence:
    return PaperEvidence(
        section_title="Method", page=1, char_start=0, char_end=20,
        quote="we use learning_rate=5e-5",
    )


def _claim() -> Claim:
    return Claim(
        claim_id="c1",
        claim_type=ClaimType.HYPERPARAMETER,
        text="AdamW, lr=5e-5",
        confidence=0.9,
        paper_evidence=_paper_evidence(),
    )


def _candidate(signature: str = "def train(learning_rate=5e-5): ...") -> RetrieveMatch:
    return RetrieveMatch(
        symbol_id=_sym_id(),
        kind=CodeSymbolKind.FUNCTION,
        language=CodeLanguage.PYTHON,
        file_path="m.py",
        start_line=1,
        start_col=0,
        end_line=10,
        end_col=0,
        name="train",
        signature=signature,
        docstring="Train loop.",
        similarity=0.95,
    )


def _sym_id():
    from uuid import uuid4
    return uuid4()


def _grounded_candidate(signature: str = "def train(learning_rate=5e-5): ...") -> RetrieveMatch:
    return _candidate(signature).model_copy(update={"source": signature, "end_line": 1})


def _llm_verdict(
    *,
    status: VerificationStatus = VerificationStatus.VERIFIED,
    code_evidence: list[CodeEvidence] | None = None,
    explanation: str = "matches",
) -> Verification:
    return Verification(
        claim_id="c1",
        status=status,
        confidence=0.85,
        explanation=explanation,
        paper_evidence=_paper_evidence(),
        code_evidence=code_evidence or [],
        discrepancies=[],
        hyperparameters=[],
    )


# ============================================================
# compose_verification
# ============================================================


def test_compose_verified_when_llm_verified_and_code_evidence() -> None:
    cand = _grounded_candidate()
    ce = [CodeEvidence(
        symbol_id=cand.symbol_id, file_path="m.py", start_line=1, end_line=1,
        quote="def train(learning_rate=5e-5): ...",
    )]
    out = compose_verification(
        claim=_claim(),
        candidates=[cand],
        deterministic_discrepancies=[],
        llm_verdict=_llm_verdict(code_evidence=ce),
    )
    assert out.status == VerificationStatus.VERIFIED
    assert out.confidence == 0.85


def test_compose_takes_the_file_path_from_the_cited_candidate() -> None:
    cand = _grounded_candidate()
    ce = [CodeEvidence(
        symbol_id=cand.symbol_id, file_path="m.py:1", start_line=1, end_line=1,
        quote="def train(learning_rate=5e-5): ...",
    )]
    out = compose_verification(
        claim=_claim(), candidates=[cand], deterministic_discrepancies=[],
        llm_verdict=_llm_verdict(code_evidence=ce),
    )
    assert out.code_evidence[0].file_path == "m.py"


@pytest.mark.parametrize("invalid", [
    "invented_quote", "missing_source", "truncated_source", "context_only",
    "blank_quote", "ambiguous",
])
def test_compose_rejects_citations_not_grounded_in_located_source(invalid: str) -> None:
    cand = _candidate().model_copy(update={
        "start_line": 10, "end_line": 12,
        "source": "def train():\n    lr = 5e-5\n    return lr",
    })
    citation = CodeEvidence(
        symbol_id=cand.symbol_id, file_path=cand.file_path,
        start_line=11, end_line=11, quote="lr = 5e-5",
    )
    if invalid == "invented_quote":
        citation.quote = "optimizer = ImaginaryOptimizer()"
    elif invalid == "missing_source":
        cand.source = None
    elif invalid == "truncated_source":
        cand.source = "def train():"
    elif invalid == "context_only":
        cand.class_context = "lr = 5e-5"
        cand.source = "def train():\n    pass\n    return None"
    elif invalid == "blank_quote":
        citation.quote = "   "
    elif invalid == "ambiguous":
        # Two equally near occurrences, neither at the stated line.
        cand.end_line = 14
        cand.source = "def train():\n    lr = 5e-5\n    pass\n    lr = 5e-5\n    return lr"
        citation.start_line = citation.end_line = 12
    out = compose_verification(
        claim=_claim(), candidates=[cand], deterministic_discrepancies=[],
        llm_verdict=_llm_verdict(code_evidence=[citation]),
    )
    assert out.status == VerificationStatus.UNVERIFIED
    assert out.code_evidence == []
    assert out.confidence == 0.0


@pytest.mark.parametrize("misplaced", ["unknown_symbol", "wrong_lines", "reversed_lines", "outside_symbol"])
def test_compose_reanchors_real_code_cited_at_the_wrong_place(misplaced: str) -> None:
    cand = _candidate().model_copy(update={
        "start_line": 10, "end_line": 12,
        "source": "def train():\n    lr = 5e-5\n    return lr",
    })
    citation = CodeEvidence(
        symbol_id=cand.symbol_id, file_path=cand.file_path,
        start_line=11, end_line=11, quote="lr = 5e-5",
    )
    if misplaced == "unknown_symbol":
        citation.symbol_id = _sym_id()
    elif misplaced == "wrong_lines":
        citation.start_line = citation.end_line = 10
    elif misplaced == "reversed_lines":
        citation.start_line, citation.end_line = 12, 11
    elif misplaced == "outside_symbol":
        citation.start_line, citation.end_line = 1, 999
    out = compose_verification(
        claim=_claim(), candidates=[cand], deterministic_discrepancies=[],
        llm_verdict=_llm_verdict(code_evidence=[citation]),
    )
    assert out.status == VerificationStatus.VERIFIED
    [kept] = out.code_evidence
    assert (kept.symbol_id, kept.file_path, kept.start_line, kept.end_line) == (
        cand.symbol_id, "m.py", 11, 11,
    )


def _span(name: str, file_path: str, start: int, source: str) -> SourceSpan:
    return SourceSpan(
        symbol_id=_sym_id(), file_path=file_path, start_line=start,
        end_line=start + source.count("\n"), name=name, source=source,
    )


def test_compose_keeps_context_code_at_its_real_location() -> None:
    """Code from a candidate's setup/callee context, cited under the
    candidate's symbol_id, is kept with the context symbol's location."""
    init = _span("__init__", "m.py", 589, "def __init__(self, h):\n    self.h = h\n    self.linears = clones(4)")
    cand = _candidate().model_copy(update={
        "start_line": 600, "end_line": 601, "source": "def forward(self, x):\n    return x",
        "context_spans": [init],
    })
    at_lines = CodeEvidence(
        symbol_id=cand.symbol_id, file_path="m.py", start_line=591, end_line=591,
        quote="self.linears = clones(4)",
    )
    elsewhere = at_lines.model_copy(update={"start_line": 700, "end_line": 702, "quote": "self.h = h"})
    out = compose_verification(
        claim=_claim(), candidates=[cand], deterministic_discrepancies=[],
        llm_verdict=_llm_verdict(code_evidence=[at_lines, elsewhere]),
    )
    assert [(e.symbol_id, e.start_line, e.end_line) for e in out.code_evidence] == [
        (init.symbol_id, 591, 591), (init.symbol_id, 590, 590),
    ]


def test_compose_reanchors_to_the_nearest_occurrence_in_the_cited_file() -> None:
    body = "def step():\n    loss.backward()\n    opt.step()"
    near = _span("step", "a.py", 40, body)
    far = _span("step", "a.py", 400, body)
    other_file = _span("step", "b.py", 84, body)
    cand = _candidate().model_copy(update={
        "file_path": "a.py", "source": None, "context_spans": [far, near, other_file],
    })
    citation = CodeEvidence(
        symbol_id=cand.symbol_id, file_path="a.py:82-83", start_line=82, end_line=83,
        quote="loss.backward()\n    opt.step()",
    )
    out = compose_verification(
        claim=_claim(), candidates=[cand], deterministic_discrepancies=[],
        llm_verdict=_llm_verdict(code_evidence=[citation]),
    )
    [kept] = out.code_evidence
    assert (kept.file_path, kept.start_line, kept.end_line) == ("a.py", 41, 42)


def test_compose_rejects_invented_citation_when_no_candidates_exist() -> None:
    citation = CodeEvidence(
        symbol_id=_sym_id(), file_path="does-not-exist.py",
        start_line=999, end_line=1000, quote="invented code",
    )
    out = compose_verification(
        claim=_claim(), candidates=[], deterministic_discrepancies=[],
        llm_verdict=_llm_verdict(code_evidence=[citation]),
    )
    assert out.status == VerificationStatus.UNVERIFIED
    assert not out.code_evidence


def test_compose_retains_only_valid_unique_citations() -> None:
    cand = _grounded_candidate()
    valid = CodeEvidence(
        symbol_id=cand.symbol_id, file_path="copied-wrong.py",
        start_line=1, end_line=1, quote="  def train(learning_rate=5e-5):   ... ",
    )
    unknown = valid.model_copy(update={"symbol_id": _sym_id()})
    out = compose_verification(
        claim=_claim(), candidates=[cand], deterministic_discrepancies=[],
        llm_verdict=_llm_verdict(code_evidence=[unknown, valid, valid]),
    )
    assert out.status == VerificationStatus.VERIFIED
    assert len(out.code_evidence) == 1
    assert out.code_evidence[0].file_path == cand.file_path


def test_compose_unverified_when_llm_verified_but_no_code_evidence() -> None:
    """Structural guard: LLM says VERIFIED but no code_evidence → UNVERIFIED."""
    out = compose_verification(
        claim=_claim(),
        candidates=[_candidate()],
        deterministic_discrepancies=[],
        llm_verdict=_llm_verdict(),
    )
    assert out.status == VerificationStatus.UNVERIFIED


def test_compose_contradicted_when_high_severity_and_code_evidence() -> None:
    cand = _grounded_candidate()
    ce = [CodeEvidence(
        symbol_id=cand.symbol_id, file_path="m.py", start_line=1, end_line=1,
        quote=cand.signature,
    )]
    high = Discrepancy(
        type=DiscrepancyType.HYPERPARAMETER_MISMATCH,
        severity=DiscrepancySeverity.HIGH,
        detail="lr mismatch",
    )
    out = compose_verification(
        claim=_claim(),
        candidates=[cand],
        deterministic_discrepancies=[high],
        llm_verdict=_llm_verdict(status=VerificationStatus.VERIFIED, code_evidence=ce),
    )
    assert out.status == VerificationStatus.CONTRADICTED
    assert out.discrepancies == [high]


def test_compose_partial_when_medium_severity_and_code_evidence() -> None:
    cand = _grounded_candidate()
    ce = [CodeEvidence(
        symbol_id=cand.symbol_id, file_path="m.py", start_line=1, end_line=1,
        quote=cand.signature,
    )]
    medium = Discrepancy(
        type=DiscrepancyType.DATASET_MISMATCH,
        severity=DiscrepancySeverity.MEDIUM,
        detail="dataset differs",
    )
    out = compose_verification(
        claim=_claim(),
        candidates=[cand],
        deterministic_discrepancies=[medium],
        llm_verdict=_llm_verdict(code_evidence=ce),
    )
    assert out.status == VerificationStatus.PARTIALLY_VERIFIED


def test_compose_drops_llm_discrepancies_without_a_candidate_quote() -> None:
    """An LLM discrepancy must quote conflicting candidate code to count."""
    cand = _grounded_candidate()
    ce = [CodeEvidence(
        symbol_id=cand.symbol_id, file_path="m.py", start_line=1, end_line=1,
        quote=cand.signature,
    )]
    quoted = Discrepancy(
        type=DiscrepancyType.HYPERPARAMETER_MISMATCH,
        severity=DiscrepancySeverity.MEDIUM,
        detail="paper says lr 1e-4, code defaults to 5e-5",
        code_quote="def train(learning_rate=5e-5):",
    )
    silent = Discrepancy(
        type=DiscrepancyType.CONFIGURATION_MISMATCH,
        severity=DiscrepancySeverity.MEDIUM,
        detail="the loss definition is not visible in the candidates",
        code_quote="",
    )
    invented = Discrepancy(
        type=DiscrepancyType.CONFIGURATION_MISMATCH,
        severity=DiscrepancySeverity.HIGH,
        detail="optimizer differs",
        code_quote="optimizer = SGD(lr=0.1)",
    )
    verdict = _llm_verdict(
        status=VerificationStatus.PARTIALLY_VERIFIED, code_evidence=ce,
    ).model_copy(update={"discrepancies": [quoted, silent, invented]})
    out = compose_verification(
        claim=_claim(),
        candidates=[cand],
        deterministic_discrepancies=[],
        llm_verdict=verdict,
    )
    assert out.discrepancies == [quoted]
    assert out.status == VerificationStatus.PARTIALLY_VERIFIED


def test_compose_verified_once_unquoted_discrepancies_drop() -> None:
    """PARTIALLY_VERIFIED needs a surviving discrepancy (phase-6-design §2.5)."""
    cand = _grounded_candidate()
    ce = [CodeEvidence(
        symbol_id=cand.symbol_id, file_path="m.py", start_line=1, end_line=1,
        quote=cand.signature,
    )]
    silent = Discrepancy(
        type=DiscrepancyType.CONFIGURATION_MISMATCH,
        severity=DiscrepancySeverity.MEDIUM,
        detail="no dataset count in the code",
    )
    verdict = _llm_verdict(
        status=VerificationStatus.PARTIALLY_VERIFIED, code_evidence=ce,
    ).model_copy(update={"discrepancies": [silent]})
    out = compose_verification(
        claim=_claim(),
        candidates=[cand],
        deterministic_discrepancies=[],
        llm_verdict=verdict,
    )
    assert out.discrepancies == []
    assert out.status == VerificationStatus.VERIFIED


def test_verdict_schema_requires_code_quote_on_discrepancies() -> None:
    fake = _fake_openai(_make_valid_verification_json())
    _build_client(fake).verdict_for_claim(
        claim=_claim(),
        paper_evidence=_paper_evidence(),
        candidates=[_candidate()],
        deterministic_discrepancies=[],
    )
    schema = fake.chat.completions.create.call_args.kwargs["response_format"]["json_schema"]["schema"]
    assert "code_quote" in schema["$defs"]["Discrepancy"]["required"]
    # The shared model's own schema stays unchanged.
    assert "code_quote" not in Discrepancy.model_json_schema()["required"]


def test_compose_unverified_when_no_llm_and_no_code_evidence() -> None:
    """LLM skipped (None) + no code_evidence → UNVERIFIED (default)."""
    out = compose_verification(
        claim=_claim(),
        candidates=[_candidate()],
        deterministic_discrepancies=[],
        llm_verdict=None,
    )
    assert out.status == VerificationStatus.UNVERIFIED


def test_compose_synthesises_explanation_when_llm_skipped() -> None:
    out = compose_verification(
        claim=_claim(),
        candidates=[_candidate()],
        deterministic_discrepancies=[],
        llm_verdict=None,
    )
    # When LLM is skipped and there are no discrepancies, the
    # synthesised explanation says "defaulting".
    assert "default" in out.explanation.lower()


# ============================================================
# extract_verdict
# ============================================================


def test_extract_verdict_runs_deterministic_checkers() -> None:
    """A high-severity hp mismatch in the deterministic checkers
    turns a VERIFIED LLM verdict into CONTRADICTED."""
    cand = _grounded_candidate(signature="def train(learning_rate=1e-3): ...")
    ce = [CodeEvidence(
        symbol_id=cand.symbol_id, file_path="m.py", start_line=1, end_line=1,
        quote=cand.signature,
    )]
    out = extract_verdict(
        claim=_claim(),
        # 1e-3 vs 5e-5 → 20× ratio, multiplicative regime fails.
        candidates=[cand],
        llm_verdict=_llm_verdict(code_evidence=ce),
    )
    assert any(d.severity == DiscrepancySeverity.HIGH for d in out.discrepancies)
    assert out.status == VerificationStatus.CONTRADICTED


# ============================================================
# run_deterministic_checkers (smoke)
# ============================================================


def test_run_deterministic_checkers_smoke() -> None:
    out = run_deterministic_checkers(claim=_claim(), candidates=[_candidate()])
    # No discrepancies should arise (paper and code agree on lr).
    assert out == []


# ============================================================
# LLMClient.verdict_for_claim
# ============================================================


def _fake_openai(content: str | None) -> MagicMock:
    fake = MagicMock()
    if content == "<RAISE_TIMEOUT>":
        fake.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
    elif content == "<RAISE_CONNECT>":
        fake.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock(), message="x"
        )
    elif content == "<RAISE_AUTH>":
        fake.chat.completions.create.side_effect = AuthenticationError(
            message="bad key", response=MagicMock(status_code=401), body=None
        )
    elif content is None or content == "":
        resp = MagicMock()
        resp.choices = []
        fake.chat.completions.create.return_value = resp
    else:
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        fake.chat.completions.create.return_value = resp
    return fake


def _build_client(fake_openai: MagicMock) -> LLMClient:
    client = LLMClient.__new__(LLMClient)
    import claim2code_api.app.config as cfg
    client._settings = cfg.get_settings()
    client._client = fake_openai
    client._model = client._settings.openai_model_chat
    # Phase 8: stub safety gate + callback.
    from claim2code_api.safety import SafetyGate
    client._gate = SafetyGate()
    client._on_violations = None
    return client


def _make_valid_verification_json() -> str:
    return json.dumps({
        "claim_id": "c1",
        "status": "VERIFIED",
        "confidence": 0.85,
        "explanation": "matches",
        "paper_evidence": {
            "section_title": "Method",
            "page": 1,
            "char_start": 0,
            "char_end": 20,
            "quote": "we use learning_rate=5e-5",
        },
        "code_evidence": [],
        "discrepancies": [],
        "hyperparameters": [],
    })


def test_verdict_for_claim_happy_path() -> None:
    fake = _fake_openai(_make_valid_verification_json())
    client = _build_client(fake)
    out = client.verdict_for_claim(
        claim=_claim(),
        paper_evidence=_paper_evidence(),
        candidates=[_candidate()],
        deterministic_discrepancies=[],
    )
    assert out.status == VerificationStatus.VERIFIED
    assert out.claim_id == "c1"
    fake.chat.completions.create.assert_called_once()
    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    assert kwargs["response_format"]["json_schema"]["name"] == "verification"


def test_verdict_for_claim_system_prompt_mentions_no_model_knowledge() -> None:
    fake = _fake_openai(_make_valid_verification_json())
    client = _build_client(fake)
    client.verdict_for_claim(
        claim=_claim(),
        paper_evidence=_paper_evidence(),
        candidates=[_candidate()],
        deterministic_discrepancies=[],
    )
    msgs = fake.chat.completions.create.call_args.kwargs["messages"]
    assert "no model knowledge" in msgs[0]["content"].lower()


def test_verdict_for_claim_user_payload_includes_deterministic_discrepancies() -> None:
    fake = _fake_openai(_make_valid_verification_json())
    client = _build_client(fake)
    disc = Discrepancy(
        type=DiscrepancyType.HYPERPARAMETER_MISMATCH,
        severity=DiscrepancySeverity.HIGH,
        detail="lr mismatch",
    )
    client.verdict_for_claim(
        claim=_claim(),
        paper_evidence=_paper_evidence(),
        candidates=[_candidate()],
        deterministic_discrepancies=[disc],
    )
    msgs = fake.chat.completions.create.call_args.kwargs["messages"]
    assert "HYPERPARAMETER_MISMATCH" in msgs[1]["content"]


def test_verdict_for_claim_user_payload_renders_candidate_signature() -> None:
    fake = _fake_openai(_make_valid_verification_json())
    client = _build_client(fake)
    client.verdict_for_claim(
        claim=_claim(),
        paper_evidence=_paper_evidence(),
        candidates=[_candidate(signature="def my_unique_func(x): pass")],
        deterministic_discrepancies=[],
    )
    msgs = fake.chat.completions.create.call_args.kwargs["messages"]
    assert "my_unique_func" in msgs[1]["content"]


def test_verdict_for_claim_timeout_raises_verification_llm_error() -> None:
    client = _build_client(_fake_openai("<RAISE_TIMEOUT>"))
    with pytest.raises(VerificationLLMError, match="connection/timeout"):
        client.verdict_for_claim(
            claim=_claim(), paper_evidence=_paper_evidence(),
            candidates=[], deterministic_discrepancies=[],
        )


def test_verdict_for_claim_connection_error_raises_verification_llm_error() -> None:
    client = _build_client(_fake_openai("<RAISE_CONNECT>"))
    with pytest.raises(VerificationLLMError):
        client.verdict_for_claim(
            claim=_claim(), paper_evidence=_paper_evidence(),
            candidates=[], deterministic_discrepancies=[],
        )


def test_verdict_for_claim_auth_error_raises_verification_llm_error() -> None:
    client = _build_client(_fake_openai("<RAISE_AUTH>"))
    with pytest.raises(VerificationLLMError, match="auth error"):
        client.verdict_for_claim(
            claim=_claim(), paper_evidence=_paper_evidence(),
            candidates=[], deterministic_discrepancies=[],
        )


def test_verdict_for_claim_schema_violation_raises_verification_llm_error() -> None:
    bad = json.dumps({
        "claim_id": "c1",
        "status": "VERIFIED",
        "confidence": 1.5,  # out of range
        "explanation": "x",
        "paper_evidence": {
            "section_title": "x", "page": 1, "char_start": 0, "char_end": 1, "quote": "x",
        },
    })
    client = _build_client(_fake_openai(bad))
    with pytest.raises(VerificationLLMError, match="schema validation"):
        client.verdict_for_claim(
            claim=_claim(), paper_evidence=_paper_evidence(),
            candidates=[], deterministic_discrepancies=[],
        )


def test_verdict_for_claim_claim_id_mismatch_raises() -> None:
    """LLM returns a different claim_id → ``VerificationLLMError``."""
    payload = json.loads(_make_valid_verification_json())
    payload["claim_id"] = "WRONG"
    client = _build_client(_fake_openai(json.dumps(payload)))
    with pytest.raises(VerificationLLMError, match="claim_id"):
        client.verdict_for_claim(
            claim=_claim(), paper_evidence=_paper_evidence(),
            candidates=[], deterministic_discrepancies=[],
        )


def test_verdict_for_claim_empty_claim_id_raises() -> None:
    """An empty claim_id in the LLM response fails Pydantic schema
    validation (min_length=1) before the explicit post-parse check
    fires — either path is acceptable; both raise VerificationLLMError.
    """
    payload = json.loads(_make_valid_verification_json())
    payload["claim_id"] = ""
    client = _build_client(_fake_openai(json.dumps(payload)))
    with pytest.raises(VerificationLLMError):
        client.verdict_for_claim(
            claim=_claim(), paper_evidence=_paper_evidence(),
            candidates=[], deterministic_discrepancies=[],
        )


def test_verdict_for_claim_empty_choices_raises() -> None:
    client = _build_client(_fake_openai(None))
    with pytest.raises(VerificationLLMError, match="no choices"):
        client.verdict_for_claim(
            claim=_claim(), paper_evidence=_paper_evidence(),
            candidates=[], deterministic_discrepancies=[],
        )

# ============================================================
# Verify payload: candidate source bodies
# ============================================================


def test_verify_payload_includes_candidate_source() -> None:
    """The verifier sees the code body, not just signature + docstring."""
    from claim2code_api.llm.client import _build_verify_payload

    cand = _candidate().model_copy(update={"source": "def train():\n    lr = 5e-5"})
    payload = _build_verify_payload(
        claim=_claim(), paper_evidence=_paper_evidence(),
        candidates=[cand], deterministic_discrepancies=[],
    )
    assert "source:\n```\ndef train():\n    lr = 5e-5\n```" in payload


def test_verify_payload_truncates_long_source() -> None:
    from claim2code_api.llm import client as llm_client_mod

    cap = llm_client_mod._VERIFY_SOURCE_CHAR_CAP
    cand = _candidate().model_copy(update={"source": "x" * (cap + 500)})
    payload = llm_client_mod._build_verify_payload(
        claim=_claim(), paper_evidence=_paper_evidence(),
        candidates=[cand], deterministic_discrepancies=[],
    )
    assert "x" * cap + "\n[... truncated ...]" in payload
    assert "x" * (cap + 1) not in payload


def test_verify_payload_omits_source_when_absent() -> None:
    from claim2code_api.llm.client import _build_verify_payload

    payload = _build_verify_payload(
        claim=_claim(), paper_evidence=_paper_evidence(),
        candidates=[_candidate()], deterministic_discrepancies=[],
    )
    assert "source:" not in payload


def test_compose_drops_llm_discrepancies_without_code_evidence() -> None:
    """"No code found" restated as a HIGH discrepancy is noise."""
    det = Discrepancy(
        type=DiscrepancyType.DATASET_MISMATCH,
        severity=DiscrepancySeverity.MEDIUM,
        detail="deterministic",
    )
    llm = _llm_verdict(status=VerificationStatus.UNVERIFIED).model_copy(
        update={"discrepancies": [Discrepancy(
            type=DiscrepancyType.METRIC_MISMATCH,
            severity=DiscrepancySeverity.HIGH,
            detail="No candidate implements the benchmark.",
        )]},
    )
    out = compose_verification(
        claim=_claim(),
        candidates=[_candidate()],
        deterministic_discrepancies=[det],
        llm_verdict=llm,
    )
    assert out.status == VerificationStatus.UNVERIFIED
    assert out.discrepancies == [det]


def test_verify_payload_includes_class_context() -> None:
    from claim2code_api.llm.client import _build_verify_payload

    cand = _candidate().model_copy(
        update={"class_context": "# reset_parameters (lines 5-6)\ninit_zeros(self.B)"},
    )
    payload = _build_verify_payload(
        claim=_claim(), paper_evidence=_paper_evidence(),
        candidates=[cand], deterministic_discrepancies=[],
    )
    assert "class_context (setup methods of its class):\n```\n# reset_parameters" in payload


def test_verify_payload_includes_callee_context() -> None:
    from claim2code_api.llm.client import _build_verify_payload

    cand = _candidate().model_copy(
        update={"callee_context": "# check_is_exhausted (env_history.py:23-24)\nreturn self._is_exhausted"},
    )
    payload = _build_verify_payload(
        claim=_claim(), paper_evidence=_paper_evidence(),
        candidates=[cand], deterministic_discrepancies=[],
    )
    assert "callee_context (repo functions it calls):\n```\n# check_is_exhausted" in payload
