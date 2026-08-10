"""LLM client layer — OpenAI Structured Outputs for claim extraction.

Phase 2 ships the :class:`LLMClient` that calls OpenAI's
``response_format=json_schema`` endpoint with the JSON schema derived
from :class:`claim2code_api.papers.schemas.ClaimList`. The same client
will be reused for Phase 6's verifier and Phase 8's embedding-based
retrieval ranking.

The module follows the same lazy-init pattern as
``db/session.py``: a module-level ``_client`` + ``get_llm_client()``
so the API process can boot even when the LLM endpoint is briefly
unreachable.
"""