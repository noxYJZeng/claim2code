"""Unit tests for :mod:`claim2code_api.llm.client.extract_comparison`.

Coverage:

  - Happy path: OpenAI returns a valid Comparison; method returns it.
  - Connection error → LiteratureComparisonError.
  - Auth error → LiteratureComparisonError.
  - API status error → LiteratureComparisonError.
  - No choices / empty content → LiteratureComparisonError.
  - Schema validation failure → LiteratureComparisonError.
  - System prompt mentions "no model knowledge" rule.
  - Payload includes claim text + paper abstract verbatim.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
)

from claim2code_api.literature.exceptions import LiteratureComparisonError
from claim2code_api.literature.schemas import Comparison, ScholarPaper
from claim2code_api.llm.client import (
    _COMPARE_SYSTEM_PROMPT,
    LLMClient,
    _build_compare_payload,
)
from claim2code_api.papers.schemas import Claim, ClaimType, PaperEvidence


def _claim() -> Claim:
    ev = PaperEvidence(
        section_title="Method",
        page=3,
        char_start=0,
        char_end=30,
        quote="we use a transformer encoder",
    )
    return Claim(
        claim_id="c1",
        claim_type=ClaimType.METHOD,
        text="We use a transformer encoder.",
        confidence=0.9,
        paper_evidence=ev,
    )


def _paper() -> ScholarPaper:
    return ScholarPaper(
        paper_id="p1",
        title="BERT",
        authors=["Devlin"],
        year=2018,
        citation_count=100,
        abstract="We introduce a new language representation model called BERT.",
        tldr="Pre-trained transformer encoder",
    )


def _comparison_dict() -> dict[str, str]:
    return {
        "problem": "Language model pre-training",
        "method": "Bidirectional transformer",
        "dataset": "Wikipedia + BookCorpus",
        "metric": "GLUE benchmark",
        "difference": "We differ in being bidirectional",
        "agreement": "We agree on using transformers",
    }


def _make_settings() -> MagicMock:
    """A minimal Settings stub the LLMClient can use."""
    s = MagicMock(name="settings")
    s.openai_api_key = "sk-test"
    s.openai_base_url = "https://api.example.com/v1"
    s.openai_model_chat = "gpt-4o"
    return s


def _build_client() -> LLMClient:
    """Construct an LLMClient whose underlying OpenAI SDK is a stub.

    Bypasses ``__init__`` to skip real httpx/OpenAI instantiation
    (mirrors Phase 6's ``test_llm_client._build_client``). We only
    need ``_client`` and ``_settings``.
    """
    client = LLMClient.__new__(LLMClient)
    client._client = MagicMock(name="openai_sdk_stub")
    client._settings = _make_settings()
    client._model = client._settings.openai_model_chat
    # Phase 8: stub the safety gate + callback (mirrors test_llm_client).
    from claim2code_api.safety import SafetyGate
    client._gate = SafetyGate()
    client._on_violations = None
    return client


def _make_response(content_dict: dict[str, str] | None) -> MagicMock:
    """Build a fake chat.completions.create response with one choice."""
    response = MagicMock(name="response")
    choice = MagicMock(name="choice")
    if content_dict is None:
        choice.message.content = None
    else:
        choice.message.content = json.dumps(content_dict)
    response.choices = [choice]
    return response


# ============================================================
# 1. Happy path
# ============================================================


class TestExtractComparisonHappy:
    def test_returns_comparison(self) -> None:
        """Valid LLM output is parsed and returned as Comparison."""
        client = _build_client()
        client._client.chat.completions.create.return_value = _make_response(  # type: ignore[attr-defined]
            _comparison_dict(),
        )
        cmp = client.extract_comparison(claim=_claim(), paper=_paper())
        assert isinstance(cmp, Comparison)
        assert cmp.problem == "Language model pre-training"
        assert cmp.method == "Bidirectional transformer"

    def test_call_uses_chat_model_and_schema(self) -> None:
        """SDK call uses settings.openai_model_chat + response_format with strict=True."""
        client = _build_client()
        create = client._client.chat.completions.create  # type: ignore[attr-defined]
        create.return_value = _make_response(_comparison_dict())  # type: ignore[attr-defined]
        client.extract_comparison(claim=_claim(), paper=_paper())
        call_kwargs = create.call_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs["model"] == "gpt-4o"
        rf = call_kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["name"] == "comparison"


# ============================================================
# 2. Error mapping
# ============================================================


class TestExtractComparisonErrors:
    def test_connection_error(self) -> None:
        client = _build_client()
        client._client.chat.completions.create.side_effect = APIConnectionError(  # type: ignore[attr-defined]
            request=MagicMock(),
        )
        with pytest.raises(LiteratureComparisonError, match="connection"):
            client.extract_comparison(claim=_claim(), paper=_paper())

    def test_timeout_error(self) -> None:
        client = _build_client()
        client._client.chat.completions.create.side_effect = APITimeoutError(  # type: ignore[attr-defined]
            request=MagicMock(),
        )
        with pytest.raises(LiteratureComparisonError, match="connection"):
            client.extract_comparison(claim=_claim(), paper=_paper())

    def test_auth_error(self) -> None:
        client = _build_client()
        client._client.chat.completions.create.side_effect = AuthenticationError(  # type: ignore[attr-defined]
            message="bad key",
            response=MagicMock(status_code=401),
            body="bad key",
        )
        with pytest.raises(LiteratureComparisonError, match="auth"):
            client.extract_comparison(claim=_claim(), paper=_paper())

    def test_status_error(self) -> None:
        client = _build_client()
        client._client.chat.completions.create.side_effect = APIStatusError(  # type: ignore[attr-defined]
            message="bad gateway",
            response=MagicMock(status_code=502),
            body=None,
        )
        with pytest.raises(LiteratureComparisonError, match="502"):
            client.extract_comparison(claim=_claim(), paper=_paper())

    def test_empty_choices(self) -> None:
        client = _build_client()
        resp = MagicMock()
        resp.choices = []
        client._client.chat.completions.create.return_value = resp  # type: ignore[attr-defined]
        with pytest.raises(LiteratureComparisonError, match="no choices"):
            client.extract_comparison(claim=_claim(), paper=_paper())

    def test_empty_content(self) -> None:
        client = _build_client()
        client._client.chat.completions.create.return_value = _make_response(None)  # type: ignore[attr-defined]
        with pytest.raises(LiteratureComparisonError, match="empty content"):
            client.extract_comparison(claim=_claim(), paper=_paper())

    def test_non_json_content(self) -> None:
        client = _build_client()
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "not json"
        resp.choices = [choice]
        client._client.chat.completions.create.return_value = resp  # type: ignore[attr-defined]
        with pytest.raises(LiteratureComparisonError, match="non-JSON"):
            client.extract_comparison(claim=_claim(), paper=_paper())

    def test_schema_validation_failure(self) -> None:
        """LLM output missing a required field → LiteratureComparisonError."""
        client = _build_client()
        bad = {
            "problem": "p",
            # method missing
            "difference": "d",
            "agreement": "a",
        }
        client._client.chat.completions.create.return_value = _make_response(bad)  # type: ignore[attr-defined]
        with pytest.raises(LiteratureComparisonError, match="schema"):
            client.extract_comparison(claim=_claim(), paper=_paper())


# ============================================================
# 3. System prompt
# ============================================================


class TestCompareSystemPrompt:
    def test_mentions_no_model_knowledge(self) -> None:
        """The 'no model knowledge' rule is in the system prompt."""
        assert "No model knowledge" in _COMPARE_SYSTEM_PROMPT or "no model knowledge" in _COMPARE_SYSTEM_PROMPT

    def test_mentions_six_fields(self) -> None:
        """All 6 Comparison fields are listed in the prompt."""
        for field in ("problem", "method", "dataset", "metric", "difference", "agreement"):
            assert field in _COMPARE_SYSTEM_PROMPT


# ============================================================
# 4. Payload builder
# ============================================================


class TestBuildComparePayload:
    def test_includes_claim_text_and_paper_abstract(self) -> None:
        """The user message cites the claim text + paper abstract verbatim."""
        payload = _build_compare_payload(claim=_claim(), paper=_paper())
        assert "transformer encoder" in payload
        assert "BERT" in payload
        assert "Pre-trained transformer encoder" in payload  # TLDR
        assert "We introduce a new language representation model called BERT." in payload

    def test_uses_section_separators(self) -> None:
        """The payload has a # Claim + # Related paper split."""
        payload = _build_compare_payload(claim=_claim(), paper=_paper())
        assert "# Claim" in payload
        assert "# Related paper" in payload