"""Glue between :mod:`claim2code_api.llm.client` and the parsed :class:`Paper`.

Kept thin on purpose: the router only needs to ask "extract
structured claims from this parsed paper" — it doesn't care about
the SDK details. Future phases will add a
``extract_verdict(claim, paper_evidence, code_evidence)`` method
on the same module.

Why a wrapper instead of calling ``LLMClient.extract_claims`` directly?

  - **Type narrowing for callers**: the router imports a function
    that takes a parsed ``Paper`` and returns ``list[Claim]`` — no
    need to know about the SDK client / settings / mockability.
  - **Single seam for tests**: patching
    ``claim2code_api.papers.extractor.extract_claims_from_paper``
    is the one place every endpoint test mocks to skip the LLM.
"""
from __future__ import annotations

from claim2code_api.llm.client import LLMClient
from claim2code_api.papers.schemas import Claim, Paper


def extract_claims_from_paper(llm: LLMClient, paper: Paper) -> list[Claim]:
    """Extract structured claims from a parsed :class:`Paper`.

    Delegates to :meth:`LLMClient.extract_claims` (which raises
    :class:`claim2code_api.papers.exceptions.ClaimExtractionError`
    on any SDK / schema / parse failure — those are mapped to HTTP
    responses by the router's exception handler).
    """
    return llm.extract_claims(paper)