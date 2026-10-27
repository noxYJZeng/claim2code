"""Unit tests for :mod:`claim2code_api.llm.client`.

These tests are CI-safe (no real OpenAI calls). We patch
``LLMClient._client`` with a fake whose ``chat.completions.create``
returns whatever the test wants — so the schema-validation, error
mapping, and request-shape paths are all exercised end-to-end without
touching the network.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
)

from claim2code_api.llm import client as llm_client_mod
from claim2code_api.llm.client import LLMClient
from claim2code_api.papers.exceptions import ClaimExtractionError
from claim2code_api.papers.schemas import ClaimType, Paper, PaperSection

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        filename="x.pdf",
        sha256="0" * 64,
        n_pages=2,
        n_chars=200,
        sections=[
            PaperSection(title="Method", page=1, char_start=0, char_end=100, text="x" * 100),
            PaperSection(title="Experiments", page=2, char_start=100, char_end=200, text="y" * 100),
        ],
    )


def _make_response(content: str) -> MagicMock:
    """Build a fake OpenAI ``chat.completions.create`` response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _make_valid_claims_json() -> str:
    payload = {
        "claims": [
            {
                "claim_id": "c1",
                "text": "The model uses β=0.1.",
                "claim_type": ClaimType.HYPERPARAMETER.value,
                "confidence": 0.9,
                "paper_evidence": {
                    "section_title": "Method",
                    "page": 1,
                    "char_start": 0,
                    "char_end": 10,
                    "quote": "we use beta=0.1",
                },
            },
            {
                "claim_id": "c2",
                "text": "The dataset has 50k examples.",
                "claim_type": ClaimType.DATASET.value,
                "confidence": 0.8,
                "paper_evidence": {
                    "section_title": "Experiments",
                    "page": 2,
                    "char_start": 0,
                    "char_end": 10,
                    "quote": "fifty thousand examples",
                },
            },
        ]
    }
    return json.dumps(payload)


def _client_with_response(content: str | None) -> MagicMock:
    """Build a fake OpenAI SDK whose chat.completions.create returns
    ``content`` (or raises if ``content`` is the sentinel ``"<RAISE_TIMEOUT>"``)."""
    fake = MagicMock()
    if content == "<RAISE_TIMEOUT>":
        # APITimeoutError requires a Request object as its first arg;
        # the original OpenAI SDK accepted a positional str message,
        # but the newer SDK is keyword-only.
        fake.chat.completions.create.side_effect = APITimeoutError(
            request=MagicMock()
        )
    elif content == "<RAISE_CONNECT>":
        fake.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock(), message="simulated connect error"
        )
    elif content == "<RAISE_AUTH>":
        fake.chat.completions.create.side_effect = AuthenticationError(
            message="bad key", response=MagicMock(status_code=401), body=None
        )
    elif content == "<RAISE_EMPTY_CHOICES>":
        resp = MagicMock()
        resp.choices = []
        fake.chat.completions.create.return_value = resp
    elif content == "<RAISE_EMPTY_CONTENT>":
        fake.chat.completions.create.return_value = _make_response("")
    else:
        fake.chat.completions.create.return_value = _make_response(content or "")
    return fake


def _build_client(fake_openai: MagicMock, monkeypatch: pytest.MonkeyPatch) -> LLMClient:
    """Construct an LLMClient whose underlying OpenAI SDK is the fake."""
    client = LLMClient.__new__(LLMClient)
    # Stub the heavy init — we only need _client + _settings.
    import claim2code_api.app.config as cfg

    client._settings = cfg.get_settings()
    client._client = fake_openai
    # Phase 8: also stub the safety gate + callback so existing tests
    # continue to work after the SDK-seam wiring. Production code
    # sets these in LLMClient.__init__; test_llm_client bypasses init.
    from claim2code_api.safety import SafetyGate
    client._gate = SafetyGate()
    client._on_violations = None
    return client


# ============================================================
# extract_claims — happy path
# ============================================================


def test_extract_claims_parses_valid_response(sample_paper: Paper) -> None:
    fake = _client_with_response(_make_valid_claims_json())
    client = _build_client(fake, pytest.MonkeyPatch())

    claims = client.extract_claims(sample_paper)

    assert len(claims) == 2
    assert claims[0].claim_type == ClaimType.HYPERPARAMETER
    assert claims[1].claim_type == ClaimType.DATASET
    # The SDK was called exactly once with the chat.completions endpoint.
    fake.chat.completions.create.assert_called_once()
    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o"
    # response_format carries the strict json_schema envelope.
    rf = kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "claim_list"
    # The schema sent is ClaimList.model_json_schema() (has minItems/maxItems).
    assert rf["json_schema"]["schema"]["properties"]["claims"]["minItems"] == 1
    assert rf["json_schema"]["schema"]["properties"]["claims"]["maxItems"] == 20


def test_extract_claims_renumbers_claim_ids(sample_paper: Paper) -> None:
    payload = json.loads(_make_valid_claims_json())
    payload["claims"][0]["claim_id"] = "}]}"
    payload["claims"][1]["claim_id"] = "c001"
    client = _build_client(_client_with_response(json.dumps(payload)), pytest.MonkeyPatch())
    assert [c.claim_id for c in client.extract_claims(sample_paper)] == ["C1", "C2"]


def _wrapped_claims_json() -> str:
    """A schema-valid response whose one claim holds the real answer."""
    payload = json.loads(_make_valid_claims_json())
    payload["claims"] = payload["claims"][:1]
    payload["claims"][0]["text"] = (
        'Here are the claims:```json\n[\n  {\n    "claim": "LoRA freezes W0."\n  }\n]\n```'
    )
    return json.dumps(payload)


def test_extract_claims_retries_wrapped_output(sample_paper: Paper) -> None:
    fake = _client_with_response(None)
    fake.chat.completions.create.side_effect = [
        _make_response(_wrapped_claims_json()),
        _make_response(_make_valid_claims_json()),
    ]
    client = _build_client(fake, pytest.MonkeyPatch())
    claims = client.extract_claims(sample_paper)
    assert fake.chat.completions.create.call_count == 2
    assert claims[0].text == "The model uses β=0.1."


def test_extract_claims_raises_when_retry_still_wrapped(sample_paper: Paper) -> None:
    fake = _client_with_response(_wrapped_claims_json())
    client = _build_client(fake, pytest.MonkeyPatch())
    with pytest.raises(ClaimExtractionError, match="two attempts"):
        client.extract_claims(sample_paper)


def test_wrapped_output_detector_leaves_real_claims_alone() -> None:
    assert not llm_client_mod._is_wrapped_output("LoRA scales ΔWx by α/r, with r ∈ {1, 2, 4, 8}.")
    assert not llm_client_mod._is_wrapped_output('The config sets "beta": 0.1 for DPO.')
    assert llm_client_mod._is_wrapped_output('[{"claim": "x"}]')


def test_extract_claims_sends_system_and_user_messages(sample_paper: Paper) -> None:
    fake = _client_with_response(_make_valid_claims_json())
    client = _build_client(fake, pytest.MonkeyPatch())
    client.extract_claims(sample_paper)
    messages = fake.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "claim extractor" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    # The user payload includes both section headings.
    assert "Method" in messages[1]["content"]
    assert "Experiments" in messages[1]["content"]


# ============================================================
# extract_claims — error mapping
# ============================================================


def test_user_payload_skips_references_section() -> None:
    paper = Paper(
        filename="x.pdf",
        sha256="0" * 64,
        n_pages=2,
        n_chars=200,
        sections=[
            PaperSection(title="Method", page=1, char_start=0, char_end=100, text="we use beta"),
            PaperSection(title="References", page=2, char_start=100, char_end=200, text="[1] Foo"),
            PaperSection(title="A Appendix", page=2, char_start=200, char_end=300, text="lr 1e-6"),
        ],
    )
    payload = llm_client_mod._build_user_payload(paper)
    assert "we use beta" in payload
    assert "lr 1e-6" in payload
    assert "[1] Foo" not in payload


def test_extract_claims_raises_on_timeout(sample_paper: Paper) -> None:
    client = _build_client(_client_with_response("<RAISE_TIMEOUT>"), pytest.MonkeyPatch())
    with pytest.raises(ClaimExtractionError, match="connection/timeout"):
        client.extract_claims(sample_paper)


def test_extract_claims_raises_on_connection_error(sample_paper: Paper) -> None:
    client = _build_client(_client_with_response("<RAISE_CONNECT>"), pytest.MonkeyPatch())
    with pytest.raises(ClaimExtractionError, match="connection/timeout"):
        client.extract_claims(sample_paper)


def test_extract_claims_raises_on_auth_error(sample_paper: Paper) -> None:
    client = _build_client(_client_with_response("<RAISE_AUTH>"), pytest.MonkeyPatch())
    with pytest.raises(ClaimExtractionError, match="auth error"):
        client.extract_claims(sample_paper)


def test_extract_claims_raises_on_empty_choices(sample_paper: Paper) -> None:
    client = _build_client(_client_with_response("<RAISE_EMPTY_CHOICES>"), pytest.MonkeyPatch())
    with pytest.raises(ClaimExtractionError, match="no choices"):
        client.extract_claims(sample_paper)


def test_extract_claims_raises_on_empty_content(sample_paper: Paper) -> None:
    client = _build_client(_client_with_response("<RAISE_EMPTY_CONTENT>"), pytest.MonkeyPatch())
    with pytest.raises(ClaimExtractionError, match="empty content"):
        client.extract_claims(sample_paper)


def test_extract_claims_raises_on_non_json(sample_paper: Paper) -> None:
    client = _build_client(_client_with_response("not json at all"), pytest.MonkeyPatch())
    with pytest.raises(ClaimExtractionError, match="non-JSON"):
        client.extract_claims(sample_paper)


def test_extract_claims_raises_on_schema_violation(sample_paper: Paper) -> None:
    # Schema violation: confidence > 1.0.
    bad = {
        "claims": [
            {
                "claim_id": "c1",
                "text": "x",
                "claim_type": "METHOD",
                "confidence": 2.5,  # out of range
                "paper_evidence": {
                    "section_title": "Method",
                    "page": 1,
                    "char_start": 0,
                    "char_end": 1,
                    "quote": "x",
                },
            }
        ]
    }
    client = _build_client(_client_with_response(json.dumps(bad)), pytest.MonkeyPatch())
    with pytest.raises(ClaimExtractionError, match="schema validation"):
        client.extract_claims(sample_paper)


def test_extract_claims_raises_on_empty_claim_list(sample_paper: Paper) -> None:
    # Schema allows min_length=1; sending 0 must fail.
    bad: dict[str, list[dict[str, object]]] = {"claims": []}
    client = _build_client(_client_with_response(json.dumps(bad)), pytest.MonkeyPatch())
    with pytest.raises(ClaimExtractionError):
        client.extract_claims(sample_paper)


# ============================================================
# check_ready
# ============================================================


def test_check_ready_returns_ok_on_success() -> None:
    fake = MagicMock()
    fake.chat.completions.create.return_value = MagicMock()
    client = _build_client(fake, pytest.MonkeyPatch())
    ok, err = client.check_ready()
    assert ok is True
    assert err is None


def test_check_ready_returns_error_on_failure() -> None:
    fake = MagicMock()
    fake.chat.completions.create.side_effect = APITimeoutError(
        request=MagicMock()
    )
    client = _build_client(fake, pytest.MonkeyPatch())
    ok, err = client.check_ready()
    assert ok is False
    assert err is not None
    assert "APITimeoutError" in err


# ============================================================
# Module-level lazy singleton
# ============================================================


def test_get_llm_client_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module-level client is not built until first call."""
    # Stub the OpenAI SDK so the real httpx proxy resolution doesn't run
    # (CI env may publish socks:// proxies that httpx 0.28 rejects).
    monkeypatch.setattr("claim2code_api.llm.client.OpenAI", MagicMock())

    llm_client_mod.reset_llm_for_tests()
    # Patch LLMClient.__init__ to count calls without constructing the
    # real OpenAI SDK (the patched constructor above is a no-op MagicMock).
    calls: list[Any] = []

    def counting_init(self: Any, settings: Any) -> None:
        calls.append(settings)
        self._settings = settings
        self._client = MagicMock()
        # Phase 8: safety gate + callback stubs.
        from claim2code_api.safety import SafetyGate
        self._gate = SafetyGate()
        self._on_violations = None

    monkeypatch.setattr(LLMClient, "__init__", counting_init)
    assert llm_client_mod._client is None
    _ = llm_client_mod.get_llm_client()
    assert len(calls) == 1
    # Second call returns the same instance (no new init).
    _ = llm_client_mod.get_llm_client()
    assert len(calls) == 1


def test_reset_llm_for_tests_rebuilds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch __init__ so the real OpenAI SDK isn't constructed.
    monkeypatch.setattr("claim2code_api.llm.client.OpenAI", MagicMock())

    def fake_init(self: Any, settings: Any) -> None:
        self._settings = settings
        self._client = MagicMock()
        # Phase 8: safety gate + callback stubs.
        from claim2code_api.safety import SafetyGate
        self._gate = SafetyGate()
        self._on_violations = None

    monkeypatch.setattr(LLMClient, "__init__", fake_init)
    llm_client_mod.reset_llm_for_tests()
    _ = llm_client_mod.get_llm_client()
    assert llm_client_mod._client is not None
    llm_client_mod.reset_llm_for_tests()
    assert llm_client_mod._client is None