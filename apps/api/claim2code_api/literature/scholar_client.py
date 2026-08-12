"""Scholarly paper search clients.

Two HTTP-based clients behind a single ``ScholarClient``
protocol:

  - :class:`SemanticScholarClient` — primary; uses Semantic
    Scholar's ``/graph/v1/paper/search`` endpoint with
    SPECTER-embedding-backed relevance ranking, the ``tldr``
    field (a pre-trained model summary), and rich metadata
    (citationCount, year, venue, externalIds, openAccessPdf,
    fieldsOfStudy).
  - :class:`OpenAlexClient` — fallback; uses OpenAlex's
    ``/works`` endpoint with cursor pagination and inverted-index
    abstract reconstruction. Higher rate limits in the polite
    pool (``mailto=``) than S2's anonymous tier.

Both clients share:

  - ``httpx.Client(proxy=None, trust_env=False)`` workaround
    for httpx 0.28+ rejecting ``socks://`` schemes that
    corporate proxies publish (memory
    ``claim2code-local-env.md``).
  - 429 retry with exponential backoff (3 tries, 1s/2s/4s).
  - Map 4xx/5xx to :class:`LiteratureSearchError` with the
    status code + URL in the message for observability.

The protocol lets the workflow inject either client (or a fixture
in tests) without coupling to the concrete class.

Usage::

    client = SemanticScholarClient(api_key=settings.semantic_scholar_api_key)
    papers = client.search(
        query="transformer fine-tuning on SQuAD",
        limit=20,
    )
"""
from __future__ import annotations

import logging
import time
from typing import Any, Protocol

import httpx

from claim2code_api.literature.exceptions import (
    LiteratureConfigError,
    LiteratureSearchError,
)
from claim2code_api.literature.schemas import ScholarPaper

_logger = logging.getLogger(__name__)

# Default fields requested from Semantic Scholar's paper-search.
# ``tldr`` is the biggest quality lever for the LLM comparison
# step (pre-trained model summary, problem/method-shaped).
DEFAULT_S2_FIELDS: list[str] = [
    "title",
    "abstract",
    "year",
    "authors",
    "citationCount",
    "venue",
    "publicationVenue",
    "externalIds",
    "openAccessPdf",
    "tldr",
    "fieldsOfStudy",
]

# Default fields requested from OpenAlex's works endpoint.
DEFAULT_OPENALEX_SELECT: list[str] = [
    "id",
    "title",
    "abstract_inverted_index",
    "publication_year",
    "authorships",
    "cited_by_count",
    "primary_location",
    "doi",
    "open_access",
]


class ScholarClient(Protocol):
    """Protocol for a scholarly paper search client.

    The workflow uses ``ScholarClient`` as the type hint; the
    concrete implementation (SemanticScholar / OpenAlex) is
    injected at construction time.
    """

    def search(self, *, query: str, limit: int) -> list[ScholarPaper]:
        """Search for papers matching ``query``; return up to ``limit``."""
        ...


# =================================================================================
# 1. Semantic Scholar
# =================================================================================


class SemanticScholarClient:
    """Semantic Scholar Graph API paper-search client.

    Endpoint: ``GET https://api.semanticscholar.org/graph/v1/paper/search``
    Docs: https://api.semanticscholar.org/api-docs/

    Rate limits: 1 req/s anonymous; 100 req/5 min with an API key
    (header ``x-api-key: …``). The key is optional — the client
    works without one but is throttled.

    Response shape::

        {
          "total": 1234,
          "offset": 0,
          "next": 25,
          "data": [
            {
              "paperId": "uuid",
              "title": "...",
              "abstract": "...",
              "year": 2024,
              "authors": [{"name": "..."}, ...],
              "citationCount": 42,
              "venue": "...",
              "publicationVenue": {"name": "..."},
              "externalIds": {"DOI": "...", "ArXiv": "..."},
              "openAccessPdf": {"url": "..."},
              "tldr": {"model": "...", "text": "..."},
              "fieldsOfStudy": ["Computer Science", ...]
            },
            ...
          ]
        }
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._max_retries = max_retries
        # Same pattern as Phase 4/6 — proxy workaround for httpx
        # 0.28+ in corporate proxy environments. ``trust_env=False``
        # disables reading ``HTTP_PROXY`` / ``HTTPS_PROXY`` /
        # ``all_proxy`` from the environment.
        headers = {"User-Agent": "claim2code/0.1"}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.Client(
            proxy=None,
            trust_env=False,
            timeout=timeout,
            headers=headers,
        )

    def search(self, *, query: str, limit: int) -> list[ScholarPaper]:
        """Search S2 for papers matching ``query``.

        :param query: search query (typically a claim text).
        :param limit: max papers to return (S2 caps at 100 per call;
            we over-fetch and let the ranker pick the top-K).
        :returns: list of :class:`ScholarPaper`.
        :raises LiteratureSearchError: API returned 4xx/5xx after
            retries exhausted.
        :raises LiteratureConfigError: API key is malformed.
        """
        if not query.strip():
            raise LiteratureConfigError("SemanticScholar search query is empty")

        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": str(min(limit, 100)),
            "fields": ",".join(DEFAULT_S2_FIELDS),
            "offset": "0",
        }

        response = self._get_with_retry(url, params=params)
        data = response.json().get("data", [])
        parsed = [self._parse_paper(item) for item in data]
        return [p for p in parsed if p is not None]  # type: ignore[misc]

    def _get_with_retry(self, url: str, *, params: dict[str, str]) -> httpx.Response:
        """GET with exponential backoff on 429 / 5xx."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt >= self._max_retries - 1:
                    break
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                last_exc = LiteratureSearchError(
                    f"SemanticScholar returned {response.status_code}: {response.text[:200]}"
                )
                if attempt >= self._max_retries - 1:
                    break
                _logger.warning(
                    "SemanticScholar %s (attempt %d); backing off",
                    response.status_code,
                    attempt + 1,
                )
                self._sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                # 4xx (non-429) is non-transient — bail.
                raise LiteratureSearchError(
                    f"SemanticScholar returned {response.status_code}: "
                    f"{response.text[:200]} (url={url})"
                )

            return response

        # Exhausted retries.
        msg = f"SemanticScholar call failed after {self._max_retries} attempts"
        if last_exc is not None:
            msg += f": {last_exc.__class__.__name__}: {last_exc}"
        raise LiteratureSearchError(msg)

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        """Sleep ``min(2**attempt, 8)`` seconds between retries."""
        time.sleep(min(2**attempt, 8))

    @staticmethod
    def _parse_paper(item: dict[str, Any]) -> ScholarPaper | None:
        """Parse one S2 paper-search item into :class:`ScholarPaper`.

        Returns ``None`` if the item is missing required fields
        (title / abstract / year) — S2 sometimes returns partial
        records that don't satisfy our schema.
        """
        try:
            paper_id = str(item["paperId"])
            title = str(item["title"])
            abstract = str(item.get("abstract") or "")
            year = int(item["year"])
        except (KeyError, TypeError, ValueError):
            return None

        if not title or not abstract or not year:
            return None

        authors_raw = item.get("authors") or []
        authors = [
            str(a.get("name", "")) for a in authors_raw if isinstance(a, dict) and a.get("name")
        ]

        venue_raw = item.get("venue")
        venue = str(venue_raw) if venue_raw else None
        pub_venue = item.get("publicationVenue")
        if isinstance(pub_venue, dict):
            venue = str(pub_venue.get("name")) or venue

        citation_count = int(item.get("citationCount") or 0)

        tldr_raw = item.get("tldr")
        tldr = (
            str(tldr_raw.get("text"))
            if isinstance(tldr_raw, dict) and tldr_raw.get("text")
            else None
        )

        oa_pdf = item.get("openAccessPdf")
        pdf_url = (
            str(oa_pdf.get("url"))
            if isinstance(oa_pdf, dict) and oa_pdf.get("url")
            else None
        )

        external_ids_raw = item.get("externalIds") or {}
        external_ids = {
            str(k): str(v)
            for k, v in external_ids_raw.items()
            if isinstance(k, str) and v is not None
        }

        return ScholarPaper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            citation_count=citation_count,
            tldr=tldr,
            abstract=abstract,
            pdf_url=pdf_url,
            external_ids=external_ids,
        )


# =================================================================================
# 2. OpenAlex
# =================================================================================


class OpenAlexClient:
    """OpenAlex works search client.

    Endpoint: ``GET https://api.openalex.org/works?search=…``
    Docs: https://docs.openalex.org/api-entities/works

    Rate limits: ~10 req/s anonymous; ~100 req/s in the "polite
    pool" with ``mailto=`` query parameter. The client always
    sends ``mailto=`` if configured.

    Response shape::

        {
          "meta": {"count": 1234, "next_cursor": "..."},
          "results": [
            {
              "id": "https://openalex.org/W123",
              "title": "...",
              "abstract_inverted_index": {"word": [0, 5, 12], ...},
              "publication_year": 2024,
              "authorships": [{"author": {"display_name": "..."}}, ...],
              "cited_by_count": 42,
              "primary_location": {
                "source": {"display_name": "..."},
                "pdf_url": "..."
              },
              "doi": "https://doi.org/...",
              "open_access": {"is_oa": true, "oa_url": "..."}
            },
            ...
          ]
        }

    OpenAlex stores abstracts as inverted indexes that need
    reconstruction. The :meth:`_reconstruct_abstract` method
    rebuilds the original abstract by sorting the positions of all
    words in the index.
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(
        self,
        *,
        mailto: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._mailto = mailto or "claim2code-dev@example.com"
        self._max_retries = max_retries
        self._client = httpx.Client(
            proxy=None,
            trust_env=False,
            timeout=timeout,
            headers={"User-Agent": f"claim2code/0.1 (mailto:{self._mailto})"},
        )

    def search(self, *, query: str, limit: int) -> list[ScholarPaper]:
        """Search OpenAlex for papers matching ``query``.

        :param query: search query (typically a claim text).
        :param limit: max papers to return (OpenAlex caps per-page
            at 200; we use per-page pagination in v1 and let the
            ranker pick the top-K).
        :returns: list of :class:`ScholarPaper`.
        :raises LiteratureSearchError: API returned 4xx/5xx after
            retries exhausted.
        """
        if not query.strip():
            raise LiteratureConfigError("OpenAlex search query is empty")

        url = f"{self.BASE_URL}/works"
        params = {
            "search": query,
            "per-page": str(min(limit, 200)),
            "select": ",".join(DEFAULT_OPENALEX_SELECT),
            "mailto": self._mailto,
        }

        response = self._get_with_retry(url, params=params)
        results = response.json().get("results", [])
        parsed = [self._parse_work(item) for item in results]
        return [p for p in parsed if p is not None]  # type: ignore[misc]

    def _get_with_retry(self, url: str, *, params: dict[str, str]) -> httpx.Response:
        """GET with exponential backoff on 429 / 5xx."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt >= self._max_retries - 1:
                    break
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                last_exc = LiteratureSearchError(
                    f"OpenAlex returned {response.status_code}: {response.text[:200]}"
                )
                if attempt >= self._max_retries - 1:
                    break
                _logger.warning(
                    "OpenAlex %s (attempt %d); backing off",
                    response.status_code,
                    attempt + 1,
                )
                self._sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                raise LiteratureSearchError(
                    f"OpenAlex returned {response.status_code}: "
                    f"{response.text[:200]} (url={url})"
                )

            return response

        msg = f"OpenAlex call failed after {self._max_retries} attempts"
        if last_exc is not None:
            msg += f": {last_exc.__class__.__name__}: {last_exc}"
        raise LiteratureSearchError(msg)

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        """Sleep ``min(2**attempt, 8)`` seconds between retries."""
        time.sleep(min(2**attempt, 8))

    @staticmethod
    def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
        """Rebuild an abstract from OpenAlex's inverted-index representation.

        OpenAlex stores abstracts as ``{word: [position1, position2, …]}``
        to avoid legal/copyright concerns about hosting full abstracts.
        We sort all (position, word) pairs and join them into a string.
        Returns an empty string if the input is ``None`` or empty.
        """
        if not inverted_index:
            return ""
        word_positions: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        if not word_positions:
            return ""
        word_positions.sort(key=lambda p: p[0])
        return " ".join(w for _, w in word_positions)

    @staticmethod
    def _parse_work(item: dict[str, Any]) -> ScholarPaper | None:
        """Parse one OpenAlex work into :class:`ScholarPaper`."""
        try:
            full_id = str(item["id"])
            paper_id = full_id.rsplit("/", 1)[-1]  # "W123" from URL
            title = str(item["title"])
            year = int(item["publication_year"])
        except (KeyError, TypeError, ValueError, IndexError):
            return None

        if not title or not year:
            return None

        abstract = OpenAlexClient._reconstruct_abstract(
            item.get("abstract_inverted_index")
        )
        if not abstract:
            return None  # Required field; skip works with no abstract.

        authorships_raw = item.get("authorships") or []
        authors = [
            str(a.get("author", {}).get("display_name", ""))
            for a in authorships_raw
            if isinstance(a, dict)
        ]
        authors = [a for a in authors if a]

        citation_count = int(item.get("cited_by_count") or 0)

        primary_location = item.get("primary_location") or {}
        venue: str | None = None
        pdf_url: str | None = None
        if isinstance(primary_location, dict):
            source = primary_location.get("source")
            if isinstance(source, dict):
                venue = str(source.get("display_name")) or None
            raw_pdf = primary_location.get("pdf_url")
            if isinstance(raw_pdf, str):
                pdf_url = raw_pdf

        doi_raw = item.get("doi")
        external_ids: dict[str, str] = {}
        if isinstance(doi_raw, str) and doi_raw:
            external_ids["DOI"] = doi_raw

        return ScholarPaper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            citation_count=citation_count,
            tldr=None,  # OpenAlex has no native TLDR.
            abstract=abstract,
            pdf_url=str(pdf_url) if pdf_url else None,
            external_ids=external_ids,
        )


__all__ = [
    "ScholarClient",
    "SemanticScholarClient",
    "OpenAlexClient",
    "DEFAULT_S2_FIELDS",
    "DEFAULT_OPENALEX_SELECT",
]