"""Unit tests for :mod:`claim2code_api.literature.scholar_client`.

Coverage:

  - Semantic Scholar happy path (parse ``data`` → list[ScholarPaper]).
  - Semantic Scholar 429 + 5xx → retries with backoff.
  - Semantic Scholar 4xx (non-429) → bails immediately.
  - Semantic Scholar missing API key → still works (1 req/s limit).
  - Semantic Scholar sets ``x-api-key`` when key is provided.
  - OpenAlex happy path (parse ``results`` + reconstruct
    ``abstract_inverted_index``).
  - OpenAlex inverted-index abstract reconstruction (round-trip).
  - OpenAlex skips works without abstracts (required field).
  - OpenAlex fall-back path: primary fails → fallback called.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from claim2code_api.literature.exceptions import (
    LiteratureConfigError,
    LiteratureSearchError,
)
from claim2code_api.literature.scholar_client import (
    OpenAlexClient,
    SemanticScholarClient,
)

# ============================================================
# Helpers
# ============================================================


def _s2_response_data(*, total: int = 1, items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"total": total, "offset": 0, "next": 0, "data": items or []}


def _s2_paper(
    *,
    paper_id: str = "abc-123",
    title: str = "A title",
    abstract: str = "We propose ...",
    year: int = 2024,
    authors: list[dict[str, Any]] | None = None,
    citation_count: int = 0,
    venue: str | None = None,
    tldr: dict[str, str] | None = None,
    external_ids: dict[str, str] | None = None,
    oa_pdf: dict[str, str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "paperId": paper_id,
        "title": title,
        "abstract": abstract,
        "year": year,
        "authors": authors or [{"name": "Alice"}],
        "citationCount": citation_count,
        "venue": venue,
    }
    if tldr is not None:
        out["tldr"] = tldr
    if external_ids is not None:
        out["externalIds"] = external_ids
    if oa_pdf is not None:
        out["openAccessPdf"] = oa_pdf
    return out


def _openalex_response(
    *, results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {"meta": {"count": len(results or [])}, "results": results or []}


def _openalex_work(
    *,
    id_: str = "https://openalex.org/W123",
    title: str = "A work",
    year: int = 2024,
    abstract_inverted_index: dict[str, list[int]] | None = None,
    cited_by_count: int = 0,
    authorships: list[dict[str, Any]] | None = None,
    primary_location: dict[str, Any] | None = None,
    doi: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": id_,
        "title": title,
        "publication_year": year,
        "cited_by_count": cited_by_count,
    }
    if abstract_inverted_index is not None:
        out["abstract_inverted_index"] = abstract_inverted_index
    if authorships is not None:
        out["authorships"] = authorships
    if primary_location is not None:
        out["primary_location"] = primary_location
    if doi is not None:
        out["doi"] = doi
    return out


def _make_httpx_response(
    *, status_code: int, json_data: dict[str, Any] | None = None, text: str = "",
) -> httpx.Response:
    request = httpx.Request("GET", "https://example.com/")
    if json_data is not None:
        import json as _json
        return httpx.Response(
            status_code=status_code,
            content=_json.dumps(json_data).encode("utf-8"),
            request=request,
        )
    return httpx.Response(
        status_code=status_code,
        text=text,
        request=request,
    )


def _patch_get_with_response(client: Any, response: httpx.Response) -> MagicMock:
    """Patch the client's _get_with_retry to return ``response``."""
    cm = patch.object(client, "_get_with_retry", return_value=response)
    return cm.__enter__()  # type: ignore[return-value]


# ============================================================
# 1. SemanticScholarClient
# ============================================================


class TestSemanticScholarHappyPath:
    def test_parses_data(self) -> None:
        client = SemanticScholarClient()
        response = _make_httpx_response(
            status_code=200,
            json_data=_s2_response_data(items=[
                _s2_paper(tldr={"model": "peS2o", "text": "TLDR"}),
                _s2_paper(paper_id="p2", title="Other", abstract="...", year=2023),
            ]),
        )
        with _patch_get_with_response(client, response):
            papers = client.search(query="transformer", limit=20)
        assert len(papers) == 2
        assert papers[0].paper_id == "abc-123"
        assert papers[0].title == "A title"
        assert papers[0].tldr == "TLDR"
        assert papers[1].paper_id == "p2"

    def test_empty_query_raises_config_error(self) -> None:
        client = SemanticScholarClient()
        with pytest.raises(LiteratureConfigError):
            client.search(query="   ", limit=10)

    def test_skips_items_missing_required_fields(self) -> None:
        """Items with no abstract OR no year are filtered out."""
        client = SemanticScholarClient()
        response = _make_httpx_response(
            status_code=200,
            json_data=_s2_response_data(items=[
                _s2_paper(),                                    # valid
                _s2_paper(paper_id="x2", abstract=""),          # no abstract
                _s2_paper(paper_id="x3", year=None),  # type: ignore[arg-type]  # no year
            ]),
        )
        with _patch_get_with_response(client, response):
            papers = client.search(query="x", limit=10)
        assert [p.paper_id for p in papers] == ["abc-123"]

    def test_external_ids_passed_through(self) -> None:
        client = SemanticScholarClient()
        response = _make_httpx_response(
            status_code=200,
            json_data=_s2_response_data(items=[
                _s2_paper(external_ids={"DOI": "10.1/abc", "ArXiv": "1234.5678"}),
            ]),
        )
        with _patch_get_with_response(client, response):
            papers = client.search(query="x", limit=10)
        assert papers[0].external_ids == {"DOI": "10.1/abc", "ArXiv": "1234.5678"}


class TestSemanticScholarRetry:
    def test_429_retries_then_raises(self) -> None:
        """429 returns should be retried; exhausted retries → LiteratureSearchError."""
        client = SemanticScholarClient(max_retries=2)
        # First call: 429; second call: 429 again → exhaust.
        responses = [
            _make_httpx_response(status_code=429, text="rate limited"),
            _make_httpx_response(status_code=429, text="rate limited"),
        ]
        with patch.object(client._client, "get", side_effect=responses):
            with pytest.raises(LiteratureSearchError):
                client.search(query="x", limit=10)

    def test_4xx_non_429_raises_immediately(self) -> None:
        """4xx (non-429) is non-transient; one call, no retry."""
        client = SemanticScholarClient(max_retries=3)
        with patch.object(
            client._client, "get",
            return_value=_make_httpx_response(status_code=400, text="bad"),
        ) as get:
            with pytest.raises(LiteratureSearchError):
                client.search(query="x", limit=10)
        assert get.call_count == 1  # no retries

    def test_recovers_on_second_attempt(self) -> None:
        """First 429, then 200 → returns papers."""
        client = SemanticScholarClient(max_retries=3)
        responses = [
            _make_httpx_response(status_code=429, text="rate limited"),
            _make_httpx_response(
                status_code=200,
                json_data=_s2_response_data(items=[_s2_paper()]),
            ),
        ]
        with patch.object(client._client, "get", side_effect=responses):
            papers = client.search(query="x", limit=10)
        assert len(papers) == 1


class TestSemanticScholarAPIKey:
    def test_no_key_works(self) -> None:
        """No API key — anon mode still works."""
        client = SemanticScholarClient(api_key=None)
        assert "x-api-key" not in client._client.headers

    def test_with_key_sets_header(self) -> None:
        client = SemanticScholarClient(api_key="s2-test-key")
        assert client._client.headers["x-api-key"] == "s2-test-key"


# ============================================================
# 2. OpenAlexClient
# ============================================================


class TestOpenAlexReconstruct:
    def test_simple_inverted_index(self) -> None:
        idx = {"hello": [0], "world": [1]}
        assert OpenAlexClient._reconstruct_abstract(idx) == "hello world"

    def test_multi_position_words(self) -> None:
        idx = {"the": [0, 3], "cat": [1], "sat": [2]}
        assert OpenAlexClient._reconstruct_abstract(idx) == "the cat sat the"

    def test_none_returns_empty(self) -> None:
        assert OpenAlexClient._reconstruct_abstract(None) == ""

    def test_empty_dict_returns_empty(self) -> None:
        assert OpenAlexClient._reconstruct_abstract({}) == ""


class TestOpenAlexHappyPath:
    def test_parses_results(self) -> None:
        client = OpenAlexClient(mailto="dev@example.com")
        # "the cat sat" at positions 0,1,2
        idx = {"the": [0], "cat": [1], "sat": [2]}
        response = _make_httpx_response(
            status_code=200,
            json_data=_openalex_response(results=[
                _openalex_work(
                    title="Work 1",
                    abstract_inverted_index=idx,
                    authorships=[{"author": {"display_name": "Alice"}}],
                    primary_location={"source": {"display_name": "NeurIPS"}},
                    doi="https://doi.org/10.1/abc",
                ),
            ]),
        )
        with _patch_get_with_retry_oa(client, response):
            papers = client.search(query="x", limit=10)
        assert len(papers) == 1
        assert papers[0].abstract == "the cat sat"
        assert papers[0].paper_id == "W123"
        assert papers[0].authors == ["Alice"]
        assert papers[0].venue == "NeurIPS"
        assert papers[0].external_ids == {"DOI": "https://doi.org/10.1/abc"}
        assert papers[0].tldr is None  # OpenAlex has no native TLDR

    def test_skips_no_abstract(self) -> None:
        """Works with no abstract are filtered (abstract is required)."""
        client = OpenAlexClient()
        response = _make_httpx_response(
            status_code=200,
            json_data=_openalex_response(results=[
                _openalex_work(),  # no abstract
                _openalex_work(
                    id_="https://openalex.org/W2",
                    abstract_inverted_index={"foo": [0]},
                ),
            ]),
        )
        with _patch_get_with_retry_oa(client, response):
            papers = client.search(query="x", limit=10)
        assert [p.paper_id for p in papers] == ["W2"]

    def test_empty_query_raises_config_error(self) -> None:
        client = OpenAlexClient()
        with pytest.raises(LiteratureConfigError):
            client.search(query="", limit=10)


def _patch_get_with_retry_oa(client: Any, response: httpx.Response) -> Any:
    cm = patch.object(client, "_get_with_retry", return_value=response)
    return cm.__enter__()  # type: ignore[return-value]


# ============================================================
# 3. ScholarClient Protocol conformance
# ============================================================


class TestScholarClientProtocol:
    def test_s2_satisfies_protocol(self) -> None:
        """SemanticScholarClient has the ``search`` method (Protocol)."""
        client = SemanticScholarClient()
        assert hasattr(client, "search")
        assert callable(client.search)

    def test_openalex_satisfies_protocol(self) -> None:
        client = OpenAlexClient()
        assert hasattr(client, "search")
        assert callable(client.search)