"""Source budgets preserve useful implementation and the evidence boundary."""
from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from claim2code_api.llm.client import _build_verify_payload
from claim2code_api.papers.schemas import Claim, ClaimType, PaperEvidence
from claim2code_api.repos.schemas import CodeLanguage, CodeSymbolKind, RetrieveMatch, SourceSpan
from claim2code_api.verification.context import (
    CONTEXT_CHAR_CAP,
    MAX_CANDIDATES,
    MAX_VERIFY_PAYLOAD_CHAR_CAP,
    SOURCE_CHAR_CAP,
    TOTAL_METADATA_CHAR_CAP,
    TOTAL_SOURCE_CHAR_CAP,
    prepare_candidates,
)
from claim2code_api.verification.exceptions import VerificationLLMError
from claim2code_api.verification.schemas import CodeEvidence, Verification, VerificationStatus
from claim2code_api.verification.verifier import compose_verification


def _claim(text: str) -> Claim:
    return Claim(
        claim_id="context-test", text=text, claim_type=ClaimType.METHOD,
        confidence=1.0,
        paper_evidence=PaperEvidence(
            section_title="Method", page=1, char_start=0, char_end=len(text), quote=text,
        ),
    )


def _candidate(source: str, start: int = 10) -> RetrieveMatch:
    return RetrieveMatch(
        symbol_id=uuid4(), kind=CodeSymbolKind.CLASS, language=CodeLanguage.PYTHON,
        file_path="layers.py",
        start_line=start, end_line=start + len(source.splitlines()) - 1,
        start_col=0, end_col=0, name="Block", signature="class Block:",
        docstring=None, similarity=1.0, source=source,
    )


def test_long_body_keeps_relevant_implementation_with_original_lines() -> None:
    text = "The attention module scales query vectors."
    source = "class Block:\n" + "    padding_value = 0\n" * 180
    source += "    def forward(self, query):\n        return query * self.attention_scale\n"
    original = _candidate(source)
    [prepared] = prepare_candidates([original], text)
    assert prepared.source is not None
    assert len(prepared.source) <= SOURCE_CHAR_CAP
    assert prepared.end_line == prepared.start_line + len(prepared.source.splitlines()) - 1
    assert "return query * self.attention_scale" not in prepared.source
    [window] = [s for s in prepared.context_spans if "return query" in s.source]
    assert window.source == "\n".join(source.splitlines()[window.start_line - 10:window.end_line - 9])
    claim = _claim(text)
    citation = CodeEvidence(
        symbol_id=window.symbol_id, file_path=window.file_path,
        start_line=192, end_line=192, quote="return query * self.attention_scale",
    )
    proposed = Verification(
        claim_id=claim.claim_id, status=VerificationStatus.VERIFIED, confidence=1.0,
        explanation="The implementation scales queries.", paper_evidence=claim.paper_evidence,
        code_evidence=[citation],
    )
    result = compose_verification(
        claim=claim, candidates=[prepared], deterministic_discrepancies=[], llm_verdict=proposed,
    )
    assert result.status == VerificationStatus.VERIFIED
    assert result.code_evidence[0].start_line == 192


def test_hidden_tail_cannot_validate_a_citation() -> None:
    source = "class Block:\n" + "    padding_value = 0\n" * 180 + "    hidden_value = 999\n"
    [prepared] = prepare_candidates([_candidate(source)], "The module scales query vectors.")
    assert prepared.context_spans == []
    claim = _claim("The module scales query vectors.")
    proposed = Verification(
        claim_id=claim.claim_id, status=VerificationStatus.VERIFIED, confidence=1.0,
        explanation="A fabricated evidence chain.", paper_evidence=claim.paper_evidence,
        code_evidence=[CodeEvidence(
            symbol_id=prepared.symbol_id, file_path=prepared.file_path,
            start_line=191, end_line=191, quote="hidden_value = 999",
        )],
    )
    result = compose_verification(
        claim=claim, candidates=[prepared], deterministic_discrepancies=[], llm_verdict=proposed,
    )
    assert result.status == VerificationStatus.UNVERIFIED
    assert result.code_evidence == []


def test_located_context_renders_its_own_identity_and_budget() -> None:
    candidate = _candidate("class Block:\n    pass")
    candidate.context_spans = [SourceSpan(
        symbol_id=uuid4(), file_path="helpers.py", name=f"scale_{i}",
        start_line=i * 10 + 1, end_line=i * 10 + 2,
        source=f"def scale_{i}(query):\n    return query * 0.5",
    ) for i in range(80)]
    [prepared] = prepare_candidates([candidate], "The attention module scales query vectors.")
    assert len(prepared.context_spans) < 80
    assert sum(len(s.source) + len(s.file_path) + len(s.name) + 120 for s in prepared.context_spans) <= CONTEXT_CHAR_CAP
    claim = _claim("The attention module scales query vectors.")
    payload = _build_verify_payload(
        claim=claim, paper_evidence=claim.paper_evidence,
        candidates=[prepared], deterministic_discrepancies=[],
    )
    for span in prepared.context_spans:
        assert f"symbol_id: {span.symbol_id}" in payload
        assert f"file: {span.file_path}:{span.start_line}-{span.end_line}" in payload
        assert span.source in payload


def test_deterministic_mismatch_can_obtain_real_citations(monkeypatch) -> None:
    from claim2code_api.verification import workflow

    source = "def train(learning_rate=1e-3):\n    return learning_rate"
    candidate = _candidate(source, start=1).model_copy(update={
        "kind": CodeSymbolKind.FUNCTION, "name": "train", "signature": "def train(learning_rate=1e-3):",
    })
    claim = _claim("learning_rate=5e-5")
    llm = MagicMock()
    llm.verdict_for_claim.return_value = Verification(
        claim_id=claim.claim_id, status=VerificationStatus.CONTRADICTED, confidence=0.9,
        explanation="The configured value differs.", paper_evidence=claim.paper_evidence,
        code_evidence=[CodeEvidence(
            symbol_id=candidate.symbol_id, file_path=candidate.file_path,
            start_line=1, end_line=1, quote="def train(learning_rate=1e-3):",
        )],
    )
    monkeypatch.setattr(workflow, "hybrid_ranker", lambda **kwargs: [MagicMock()])
    monkeypatch.setattr(workflow, "_hybrid_to_retrieve_matches", lambda **kwargs: [candidate])
    monkeypatch.setattr(workflow, "get_embedding_client", MagicMock)
    monkeypatch.setattr(workflow, "get_llm_client", lambda: llm)
    result, error = workflow._verify_one_claim(
        db=MagicMock(), analysis_id=uuid4(), claim=claim, repository_id=uuid4(), top_k=8,
    )
    llm.verdict_for_claim.assert_called_once()
    assert error is None
    assert result.status == VerificationStatus.PARTIALLY_VERIFIED
    assert result.code_evidence[0].quote == "def train(learning_rate=1e-3):"


def _retry_fixture(monkeypatch, *, add_source: bool = True, fail_model: bool = False):
    from claim2code_api.verification import workflow

    claim = _claim("The attention module scales query vectors.")
    first = _candidate("class Block:\n    pass")
    added = _candidate("def attention(query):\n    return query * 0.5", start=1)
    added = added.model_copy(update={"file_path": "attention.py", "name": "attention"})
    model = MagicMock()
    unresolved = Verification(
        claim_id=claim.claim_id, status=VerificationStatus.UNVERIFIED, confidence=0.1,
        explanation="The implementation is not in the initial shortlist.",
        paper_evidence=claim.paper_evidence,
    )
    supported = Verification(
        claim_id=claim.claim_id, status=VerificationStatus.VERIFIED, confidence=0.9,
        explanation="The newly retrieved operation scales the query.",
        paper_evidence=claim.paper_evidence, code_evidence=[CodeEvidence(
            symbol_id=added.symbol_id, file_path=added.file_path,
            start_line=2, end_line=2, quote="return query * 0.5",
        )],
    )
    model.verdict_for_claim.side_effect = [
        RuntimeError("unavailable") if fail_model else unresolved, supported,
    ]
    retriever = MagicMock(side_effect=[[first.symbol_id], [first.symbol_id, added.symbol_id]])
    monkeypatch.setattr(workflow, "hybrid_ranker", retriever)
    monkeypatch.setattr(workflow, "get_embedding_client", MagicMock)
    monkeypatch.setattr(workflow, "get_llm_client", lambda: model)
    monkeypatch.setattr(
        workflow, "_hybrid_to_retrieve_matches",
        lambda **kwargs: [first, added] if add_source and added.symbol_id in kwargs["matches"] else [first],
    )
    result, error = workflow._verify_one_claim(
        db=MagicMock(), analysis_id=uuid4(), claim=claim, repository_id=uuid4(), top_k=5,
    )
    return result, error, model, retriever


def test_unverified_retries_once_when_expansion_finds_new_source(monkeypatch) -> None:
    result, error, model, retriever = _retry_fixture(monkeypatch)
    assert error is None
    assert result.status == VerificationStatus.VERIFIED
    assert model.verdict_for_claim.call_count == 2
    assert [c.kwargs["top_k"] for c in retriever.call_args_list] == [5, 12]
    assert len(model.verdict_for_claim.call_args_list[1].kwargs["candidates"]) == 2


def test_unchanged_evidence_does_not_resample_the_model(monkeypatch) -> None:
    result, error, model, retriever = _retry_fixture(monkeypatch, add_source=False)
    assert error is None
    assert result.status == VerificationStatus.UNVERIFIED
    assert retriever.call_count == 2
    model.verdict_for_claim.assert_called_once()


def test_model_failure_does_not_trigger_evidence_retry(monkeypatch) -> None:
    result, error, model, retriever = _retry_fixture(monkeypatch, fail_model=True)
    assert "unavailable" in error
    assert result.status == VerificationStatus.UNVERIFIED
    retriever.assert_called_once()
    model.verdict_for_claim.assert_called_once()


@pytest.mark.parametrize("count", [8, 12])
def test_request_budget_includes_every_candidate_and_metadata(count: int) -> None:
    """Initial and expanded shortlists preserve useful tails within one budget."""
    claim = _claim("The attention module scales query vectors.")
    candidates = []
    for i in range(count):
        source = "class Block:\n" + "    padding_value = 0\n" * 180
        source += f"    def forward(self, query):\n        return query * self.attention_scale_{i}\n"
        candidate = _candidate(source).model_copy(update={
            "file_path": f"layer_{i}.py", "signature": "s" * 100000,
            "docstring": "d" * 100000,
        })
        candidate.context_spans = [SourceSpan(
            symbol_id=uuid4(), file_path=f"helper_{i}.py", name=f"helper_{j}",
            start_line=j * 4 + 1, end_line=j * 4 + 2,
            source=f"def helper_{j}(x):\n    return x + {j}",
        ) for j in range(80)]
        candidates.append(candidate)
    prepared = prepare_candidates(candidates, claim.text)
    assert len(prepared) == count
    source_chars = sum(
        len(c.source or "") + sum(len(span.source) for span in c.context_spans)
        for c in prepared
    )
    metadata_chars = sum(
        len(c.file_path) + len(c.name) + len(c.signature) + len(c.docstring or "") + 256
        for c in prepared
    )
    assert source_chars <= TOTAL_SOURCE_CHAR_CAP
    assert metadata_chars <= TOTAL_METADATA_CHAR_CAP
    payload = _build_verify_payload(
        claim=claim, paper_evidence=claim.paper_evidence,
        candidates=prepared, deterministic_discrepancies=[],
    )
    assert len(payload) <= MAX_VERIFY_PAYLOAD_CHAR_CAP
    assert "s" * 1000 not in payload
    assert "d" * 1000 not in payload
    for i, candidate in enumerate(prepared):
        assert any(f"self.attention_scale_{i}" in span.source for span in candidate.context_spans)
        assert all(span.source in payload for span in candidate.context_spans)


def test_oversized_shortlist_is_bounded_before_model_and_validation() -> None:
    candidates = [_candidate("class Block:\n    pass") for _ in range(50)]
    prepared = prepare_candidates(candidates, "The module scales query vectors.")
    assert len(prepared) == MAX_CANDIDATES
    assert {c.symbol_id for c in prepared} == {c.symbol_id for c in candidates[:MAX_CANDIDATES]}


def test_oversized_claim_fails_instead_of_silently_truncating_evidence() -> None:
    claim = _claim("x" * MAX_VERIFY_PAYLOAD_CHAR_CAP)
    with pytest.raises(VerificationLLMError, match="bounded prompt budget"):
        _build_verify_payload(
            claim=claim, paper_evidence=claim.paper_evidence,
            candidates=prepare_candidates([_candidate("class Block:\n    pass")], claim.text),
            deterministic_discrepancies=[],
        )
