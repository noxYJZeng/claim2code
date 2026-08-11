"""Provider-agnostic embedding interface + the OpenAI adapter.

Phase 4 ships two pieces:

  1. ``EmbeddingClient`` — a ``Protocol`` declaring the canonical
     ``embed(texts: list[str]) -> list[list[float]]`` method. The
     rest of the indexing layer talks to this interface, so
     swapping OpenAI for a local sentence-transformers model
     (Phase 4.5+) is a one-file change.

  2. ``OpenAIEmbeddingClient`` — the v1 implementation. It drives
     the OpenAI SDK's ``embeddings.create`` endpoint, batches
     large inputs into ``Settings.embed_batch_size`` chunks
     (the API accepts up to 2 048 inputs per call), and
     transparently retries on transient errors up to
     ``Settings.embed_max_retries`` times.

Error mapping
-------------
SDK failures map to :class:`EmbeddingError` so the router can
return HTTP 500 without leaking SDK internals. The wrapper also
guards against dimension drift (a real risk when the user
changes ``openai_model_embed`` without restarting): any vector
whose length differs from ``Settings.openai_embed_dim`` raises
:class:`EmbeddingDimMismatchError`.

httpx proxy workaround
----------------------
Like :mod:`claim2code_api.llm.client`, we pass
``http_client=httpx.Client(proxy=None)`` so the ``all_proxy``
environment variable doesn't break httpx 0.28+.
"""
from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Protocol, runtime_checkable

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
)

from claim2code_api.app.config import Settings, get_settings
from claim2code_api.indexing.exceptions import (
    EmbeddingDimMismatchError,
    EmbeddingError,
)

logger = logging.getLogger(__name__)

# Per-request timeout for an OpenAI embedding call. Keep tight so
# a single hung call doesn't burn the entire workflow budget.
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0


@runtime_checkable
class EmbeddingClient(Protocol):
    """Provider-agnostic embedding interface.

    Implementations must:

      - Accept a list of strings and return one vector per string.
      - Return vectors of length ``Settings.openai_embed_dim``
        (the rest of the indexing layer assumes a fixed dim).
      - Raise :class:`EmbeddingError` on any unrecoverable failure
        (network, auth, schema) and retry transient errors internally.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` → ``[v_0, v_1, ...]`` where each
        ``v_i`` is a length-``openai_embed_dim`` float list."""
        ...


class OpenAIEmbeddingClient:
    """OpenAI ``text-embedding-3-small`` adapter (default 1536 dims).

    Why a class — not just a module-level helper:

      - Tests instantiate it directly with a hand-rolled Settings
        so they can patch ``_client.embeddings.create`` without
        touching the cached singleton.
      - The protocol check above lets the rest of the indexing
        layer call ``client.embed(...)`` without a type-ignore.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # ``proxy=None`` disables proxying via env vars
        # (``all_proxy``/``HTTPS_PROXY``) — same workaround as
        # :mod:`claim2code_api.llm.client`. httpx 0.28+ rejects
        # ``socks://`` schemes, which corporate proxies publish.
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            # ``proxy=None`` + ``trust_env=False`` disables proxying
            # via env vars (``all_proxy``/``HTTPS_PROXY``). Necessary
            # because httpx 0.28+ rejects the ``socks://`` scheme
            # corporate proxies publish — see module docstring.
            http_client=httpx.Client(proxy=None, trust_env=False),
            timeout=_DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` via OpenAI ``embeddings.create``.

        Batches the input into ``Settings.embed_batch_size`` chunks.
        Retries transient errors with exponential backoff up to
        ``Settings.embed_max_retries`` times. Returns the
        concatenated vectors in the original order.

        :raises EmbeddingError: the SDK call failed irrecoverably
            (auth, schema, repeated timeouts).
        :raises EmbeddingDimMismatchError: a returned vector has
            the wrong length — almost always a settings typo.
        """
        if not texts:
            return []

        out: list[list[float]] = []
        batch_size = self._settings.embed_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = self._embed_one_batch(batch)
            out.extend(vectors)
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed_one_batch(self, batch: list[str]) -> list[list[float]]:
        """Embed one batch with retries; returns vectors in input order."""
        last_exc: Exception | None = None
        for attempt in range(self._settings.embed_max_retries + 1):
            try:
                response = self._client.embeddings.create(
                    model=self._settings.openai_model_embed,
                    input=batch,
                )
            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                if attempt >= self._settings.embed_max_retries:
                    break
                # Exponential backoff: 1s, 2s, 4s ... capped at 8s.
                self._sleep_backoff(attempt)
                continue
            except AuthenticationError as exc:
                # Auth errors are not transient — bail immediately.
                raise EmbeddingError(
                    f"Embedding auth error (check OPENAI_API_KEY): {exc}"
                ) from exc
            except APIStatusError as exc:
                last_exc = exc
                # 5xx counts as transient; 4xx does not.
                if exc.status_code < 500 or attempt >= self._settings.embed_max_retries:
                    raise EmbeddingError(
                        f"Embedding API returned {exc.status_code}: {exc.message}"
                    ) from exc
                self._sleep_backoff(attempt)
                continue
            else:
                return self._extract_vectors(response, len(batch))

        # Exhausted retries.
        assert last_exc is not None  # noqa: S101 — defensive
        raise EmbeddingError(
            f"Embedding call failed after {self._settings.embed_max_retries + 1} "
            f"attempts: {last_exc.__class__.__name__}: {last_exc}"
        ) from last_exc

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        """Sleep ``min(2**attempt, 8)`` seconds between retries."""
        delay = min(2 ** attempt, 8)
        time.sleep(delay)

    def _extract_vectors(self, response: object, expected_count: int) -> list[list[float]]:
        """Pull the embedding vectors out of the SDK response.

        The OpenAI SDK returns ``response.data`` as a list of
        objects whose ``index`` field preserves the input order;
        we sort by ``index`` defensively (the API usually returns
        them in order, but we don't want to depend on it).

        :raises EmbeddingError: the response is missing fields.
        :raises EmbeddingDimMismatchError: a vector has the wrong
            length for ``Settings.openai_embed_dim``.
        """
        data = getattr(response, "data", None)
        if not data or len(data) != expected_count:
            raise EmbeddingError(
                f"Embedding response returned {len(data) if data else 0} vectors, "
                f"expected {expected_count}"
            )

        # Sort by ``index`` so the output matches the input order.
        try:
            ordered = sorted(data, key=lambda d: d.index)
        except AttributeError as exc:
            raise EmbeddingError(
                f"Embedding response missing 'index' field: {exc}"
            ) from exc

        expected_dim = self._settings.openai_embed_dim
        vectors: list[list[float]] = []
        for vec in ordered:
            raw = getattr(vec, "embedding", None)
            if raw is None:
                raise EmbeddingError("Embedding response missing 'embedding' field")
            if len(raw) != expected_dim:
                raise EmbeddingDimMismatchError(expected=expected_dim, actual=len(raw))
            vectors.append(list(raw))
        return vectors


# ----------------------------------------------------------------------
# Module-level lazy singleton (mirrors llm/client.py)
# ----------------------------------------------------------------------

_client: OpenAIEmbeddingClient | None = None
_client_lock = Lock()


def get_embedding_client() -> OpenAIEmbeddingClient:
    """Return the lazily-initialised process-wide ``OpenAIEmbeddingClient``.

    Tests use :func:`reset_embedding_client_for_tests` to drop the
    cache so the next call picks up changes to env vars.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = OpenAIEmbeddingClient(get_settings())
    return _client


def reset_embedding_client_for_tests() -> None:
    """Drop the cached client so the next ``get_embedding_client`` rebuilds it."""
    global _client
    with _client_lock:
        _client = None
