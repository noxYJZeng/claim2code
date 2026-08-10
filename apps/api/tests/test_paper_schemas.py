"""Unit tests for :mod:`claim2code_api.papers.schemas`.

These tests are CI-safe (no DB, no LLM, no PDF parser). They verify:

  - All Pydantic models validate / reject as expected.
  - :class:`ClaimList` enforces the 1-20 claims range that OpenAI's
    Structured Outputs also enforce.
  - ``model_json_schema()`` produces a valid JSON schema dict that
    can be sent to OpenAI's ``response_format`` parameter.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from claim2code_api.papers.schemas import (
    Claim,
    ClaimList,
    ClaimType,
    PaperEvidence,
)


def _make_evidence() -> PaperEvidence:
    return PaperEvidence(
        section_title="Method",
        page=3,
        char_start=10,
        char_end=50,
        quote="We use β=0.1.",
    )


def _make_claim() -> Claim:
    return Claim(
        claim_id=str(uuid4()),
        text="The model uses β=0.1.",
        claim_type=ClaimType.HYPERPARAMETER,
        confidence=0.92,
        paper_evidence=_make_evidence(),
    )


def test_evidence_rejects_missing_quote() -> None:
    with pytest.raises(ValidationError) as ei:
        PaperEvidence(
            section_title="Method",
            page=3,
            char_start=0,
            char_end=10,
            quote="",
        )
    assert "quote" in str(ei.value)


def test_evidence_rejects_unknown_extra() -> None:
    """``extra="forbid"`` rejects unknown keys fed via ``model_validate``."""
    with pytest.raises(ValidationError) as ei:
        PaperEvidence.model_validate(
            {
                "section_title": "Method",
                "page": 3,
                "char_start": 0,
                "char_end": 10,
                "quote": "x",
                "bogus": "nope",
            }
        )
    assert "bogus" in str(ei.value)


def test_claim_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        Claim(
            claim_id=str(uuid4()),
            text="x",
            claim_type=ClaimType.METHOD,
            confidence=1.5,
            paper_evidence=_make_evidence(),
        )


def test_claim_accepts_valid_input() -> None:
    c = _make_claim()
    assert c.claim_type == ClaimType.HYPERPARAMETER
    assert 0.0 <= c.confidence <= 1.0


def test_claim_list_enforces_min_length() -> None:
    with pytest.raises(ValidationError):
        ClaimList(claims=[])


def test_claim_list_enforces_max_length() -> None:
    claims = [_make_claim() for _ in range(21)]
    with pytest.raises(ValidationError):
        ClaimList(claims=claims)


def test_claim_list_accepts_within_range() -> None:
    claims = [_make_claim() for _ in range(7)]
    cl = ClaimList(claims=claims)
    assert len(cl.claims) == 7


def test_claim_list_json_schema_has_min_max_items() -> None:
    """The schema we send to OpenAI must carry ``minItems`` / ``maxItems``.

    Pydantic emits the array constraints at the **root** schema's
    ``properties.claims`` (not nested under ``$defs``).
    """
    schema = ClaimList.model_json_schema()
    claims_schema = schema["properties"]["claims"]
    assert claims_schema["minItems"] == 1
    assert claims_schema["maxItems"] == 20


def test_claim_list_json_schema_root_has_no_extras() -> None:
    """``extra="forbid"`` produces ``additionalProperties: false`` at the root."""
    schema = ClaimList.model_json_schema()
    assert schema["additionalProperties"] is False


def test_claim_type_is_str_enum() -> None:
    """ClaimType must serialise as the string value, not an int."""
    assert ClaimType.METHOD == "METHOD"
    assert ClaimType.METHOD.value == "METHOD"
    # JSON serialisation emits the *string* value, not the int ordinal.
    import json

    payload = json.dumps({"t": ClaimType.METHOD})
    loaded = json.loads(payload)
    assert loaded == {"t": "METHOD"}
    assert isinstance(ClaimType.METHOD, str)
