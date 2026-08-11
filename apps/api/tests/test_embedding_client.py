"""Unit tests for :mod:`claim2code_api.indexing.embedding_client`.

CI-safe — no real OpenAI calls. We patch the SDK client
``OpenAIEmbeddingClient._client`` with a ``MagicMock`` whose
``embeddings.create`` returns whatever the test wants.

Coverage:

  - ``embed`` returns one vector per input, in order.
  - Batching splits a large input into ``Settings.embed_batch_size``
    chunks and concatenates the results.
  - Dimension mismatch raises :class:`EmbeddingDimMismatchError`.
  - Empty input short-circuits without calling the SDK.
  - Per-batch retry on ``APITimeoutError`` + ``APIConnectionError``
    succeeds on the second attempt.
  - Per-batch retry exhausts ``embed_max_retries`` and raises
    :class:`EmbeddingError`.
  - ``AuthenticationError`` is non-transient — single attempt,
    raised as :class:`EmbeddingError`.
  - ``APIStatusError`` 5xx is retried; 4xx is not.
  - ``reset_embedding_client_for_tests`` drops the cached singleton.
"""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
)

from claim2code_api import indexing
from claim2code_api.app.config import Settings
from claim2code_api.indexing import embedding_client as embed_mod
from claim2code_api.indexing.embedding_client import (
    EmbeddingClient,
    OpenAIEmbeddingClient,
    get_embedding_client,
    reset_embedding_client_for_tests,
)
from claim2code_api.indexing.exceptions import (
    EmbeddingDimMismatchError,
    EmbeddingError,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def settings() -> Settings:
    """A Settings with embed_batch_size=2 + embed_max_retries=2.

    Tiny batch size so the multi-batch tests don't need 100+
    inputs; small retry count so the exhaustion test is fast.
    """
    return Settings(
        openai_api_key="sk-test",
        openai_base_url="https://api.openai.com/v1",
        openai_model_embed="text-embedding-3-small",
        openai_embed_dim=4,
        embed_batch_size=2,
        embed_max_retries=2,
    )


def _make_response(vectors: list[list[float]]) -> MagicMock:
    """Build a fake OpenAI ``embeddings.create`` response."""
    response = MagicMock()
    response.data = []
    for i, vec in enumerate(vectors):
        item = MagicMock()
        item.index = i
        item.embedding = vec
        response.data.append(item)
    return response


def _make_4d_vec(seed: int) -> list[float]:
    """Deterministic 4-dim vector for tests (length = openai_embed_dim).

    Doesn't matter what the values are — we just need a repeatable
    length-4 vector the test can compare against.
    """
    return [float(seed + j) / 10.0 for j in range(4)]


# ============================================================
# Interface / construction
# ============================================================


def test_openai_client_satisfies_protocol(settings: Settings) -> None:
    """OpenAIEmbeddingClient implements the EmbeddingClient protocol."""
    client = OpenAIEmbeddingClient(settings)
    assert isinstance(client, EmbeddingClient)


def test_openai_client_constructs_sdk(settings: Settings) -> None:
    """The constructor wires the OpenAI SDK with settings."""
    client = OpenAIEmbeddingClient(settings)
    assert client._settings is settings
    assert client._client is not None


# ============================================================
# embed — happy path
# ============================================================


def test_embed_returns_one_vector_per_input(settings: Settings) -> None:
    """Three inputs with batch_size=2 → 2 SDK calls (2 + 1)."""
    client = OpenAIEmbeddingClient(settings)
    inputs = ["alpha", "beta", "gamma"]
    with patch.object(client._client.embeddings, "create") as fake_create:
        fake_create.side_effect = [
            _make_response([_make_4d_vec(1), _make_4d_vec(2)]),
            _make_response([_make_4d_vec(3)]),
        ]
        out = client.embed(inputs)

    assert len(out) == 3
    assert all(len(v) == 4 for v in out)
    # Order preserved: input "alpha" → first vector (seed=1).
    assert out[0] == _make_4d_vec(1)
    assert out[1] == _make_4d_vec(2)
    assert out[2] == _make_4d_vec(3)


def test_embed_empty_input_short_circuits(settings: Settings) -> None:
    """``[]`` in → ``[]`` out, no SDK call."""
    client = OpenAIEmbeddingClient(settings)
    with patch.object(client._client.embeddings, "create") as fake_create:
        out = client.embed([])

    assert out == []
    fake_create.assert_not_called()


def test_embed_sends_correct_model_setting(settings: Settings) -> None:
    client = OpenAIEmbeddingClient(settings)
    with patch.object(client._client.embeddings, "create") as fake_create:
        fake_create.return_value = _make_response([_make_4d_vec(0)])
        client.embed(["only"])

    assert fake_create.call_args.kwargs["model"] == "text-embedding-3-small"
    assert fake_create.call_args.kwargs["input"] == ["only"]


# ============================================================
# embed — batching
# ============================================================


def test_embed_batches_according_to_settings(settings: Settings) -> None:
    """5 inputs with batch_size=2 → 3 SDK calls (2 + 2 + 1)."""
    client = OpenAIEmbeddingClient(settings)
    inputs = [f"text-{i}" for i in range(5)]

    with patch.object(client._client.embeddings, "create") as fake_create:
        fake_create.side_effect = [
            _make_response([_make_4d_vec(0), _make_4d_vec(1)]),
            _make_response([_make_4d_vec(2), _make_4d_vec(3)]),
            _make_response([_make_4d_vec(4)]),
        ]
        out = client.embed(inputs)

    assert len(out) == 5
    assert fake_create.call_count == 3
    # Each call's batch size matches embed_batch_size=2 (last is 1).
    sizes = [len(c.kwargs["input"]) for c in fake_create.call_args_list]
    assert sizes == [2, 2, 1]


def test_embed_batches_concatenate_in_order(settings: Settings) -> None:
    """Batch B's vectors come after batch A's in the output."""
    client = OpenAIEmbeddingClient(settings)
    inputs = [f"text-{i}" for i in range(4)]

    with patch.object(client._client.embeddings, "create") as fake_create:
        fake_create.side_effect = [
            _make_response([_make_4d_vec(10), _make_4d_vec(11)]),
            _make_response([_make_4d_vec(20), _make_4d_vec(21)]),
        ]
        out = client.embed(inputs)

    assert out[0] == _make_4d_vec(10)
    assert out[1] == _make_4d_vec(11)
    assert out[2] == _make_4d_vec(20)
    assert out[3] == _make_4d_vec(21)


# ============================================================
# embed — dimension guarding
# ============================================================


def test_embed_raises_on_dim_mismatch(settings: Settings) -> None:
    """A returned vector of the wrong length trips EmbeddingDimMismatchError."""
    client = OpenAIEmbeddingClient(settings)
    with patch.object(client._client.embeddings, "create") as fake_create:
        # openai_embed_dim=4 but the SDK returns a 3-dim vector.
        fake_create.return_value = _make_response([[0.1, 0.2, 0.3]])
        with pytest.raises(EmbeddingDimMismatchError) as excinfo:
            client.embed(["hello"])

    assert excinfo.value.expected == 4
    assert excinfo.value.actual == 3


def test_embed_raises_on_short_response(settings: Settings) -> None:
    """If the response has fewer vectors than inputs, surface EmbeddingError."""
    client = OpenAIEmbeddingClient(settings)
    with patch.object(client._client.embeddings, "create") as fake_create:
        fake_create.return_value._mock_children.clear()
        # 2 inputs, but only 1 vector returned.
        resp = _make_response([_make_4d_vec(0)])
        fake_create.return_value = resp
        with pytest.raises(EmbeddingError, match=r"returned 1 vectors"):
            client.embed(["a", "b"])


# ============================================================
# embed — retry / error mapping
# ============================================================


def test_embed_retries_on_timeout_then_succeeds(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """APITimeoutError on attempt 1 → retry on attempt 2 → succeed."""
    client = OpenAIEmbeddingClient(settings)
    monkeypatch.setattr(embed_mod.OpenAIEmbeddingClient, "_sleep_backoff", lambda self, n: None)

    with patch.object(client._client.embeddings, "create") as fake_create:
        fake_create.side_effect = [
            APITimeoutError(request=MagicMock()),
            _make_response([_make_4d_vec(0)]),
        ]
        out = client.embed(["only"])

    assert out == [_make_4d_vec(0)]
    assert fake_create.call_count == 2


def test_embed_retries_on_connection_error_then_succeeds(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """APIConnectionError is in the transient bucket."""
    client = OpenAIEmbeddingClient(settings)
    monkeypatch.setattr(embed_mod.OpenAIEmbeddingClient, "_sleep_backoff", lambda self, n: None)

    with patch.object(client._client.embeddings, "create") as fake_create:
        fake_create.side_effect = [
            APIConnectionError(request=MagicMock()),
            APIConnectionError(request=MagicMock()),
            _make_response([_make_4d_vec(0)]),
        ]
        out = client.embed(["only"])

    assert out == [_make_4d_vec(0)]
    assert fake_create.call_count == 3


def test_embed_exhausts_retries_on_persistent_timeout(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """embed_max_retries=2 → 1 + 2 = 3 attempts, then EmbeddingError."""
    client = OpenAIEmbeddingClient(settings)
    monkeypatch.setattr(embed_mod.OpenAIEmbeddingClient, "_sleep_backoff", lambda self, n: None)

    with patch.object(client._client.embeddings, "create") as fake_create:
        fake_create.side_effect = APITimeoutError(request=MagicMock())
        with pytest.raises(EmbeddingError, match=r"3 attempts"):
            client.embed(["only"])

    # 1 initial + 2 retries = 3
    assert fake_create.call_count == 3


def test_embed_does_not_retry_on_auth_error(settings: Settings) -> None:
    """AuthenticationError is non-transient — single attempt, raise EmbeddingError."""
    client = OpenAIEmbeddingClient(settings)

    with patch.object(client._client.embeddings, "create") as fake_create:
        fake_create.side_effect = AuthenticationError(
            message="bad key",
            response=MagicMock(status_code=401),
            body=None,
        )
        with pytest.raises(EmbeddingError, match=r"auth error"):
            client.embed(["only"])

    assert fake_create.call_count == 1


def test_embed_retries_on_5xx(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """APIStatusError 503 is transient — retry, then succeed."""
    client = OpenAIEmbeddingClient(settings)
    monkeypatch.setattr(embed_mod.OpenAIEmbeddingClient, "_sleep_backoff", lambda self, n: None)

    def fake_503_then_ok(*args: Any, **kwargs: Any) -> Any:
        if fake_503_then_ok.attempts == 0:  # type: ignore[attr-defined]
            fake_503_then_ok.attempts = 1  # type: ignore[attr-defined]
            raise APIStatusError(
                "boom",
                response=MagicMock(status_code=503),
                body=None,
            )
        return _make_response([_make_4d_vec(0)])

    fake_503_then_ok.attempts = 0  # type: ignore[attr-defined]

    with patch.object(client._client.embeddings, "create", side_effect=fake_503_then_ok):
        out = client.embed(["only"])

    assert out == [_make_4d_vec(0)]


def test_embed_does_not_retry_on_4xx(settings: Settings) -> None:
    """APIStatusError 400 is non-transient — single attempt, raise EmbeddingError."""
    client = OpenAIEmbeddingClient(settings)

    with patch.object(client._client.embeddings, "create") as fake_create:
        fake_create.side_effect = APIStatusError(
            "bad input",
            response=MagicMock(status_code=400),
            body=None,
        )
        with pytest.raises(EmbeddingError, match=r"400"):
            client.embed(["only"])

    assert fake_create.call_count == 1


# ============================================================
# Backoff
# ============================================================


def test_sleep_backoff_caps_at_eight_seconds() -> None:
    """The backoff grows 1, 2, 4, 8, 8, 8, ... (cap at 8s)."""
    with patch.object(time, "sleep") as fake_sleep:
        for attempt in range(8):
            OpenAIEmbeddingClient._sleep_backoff(attempt)
    delays = [c.args[0] for c in fake_sleep.call_args_list]
    assert delays == [1, 2, 4, 8, 8, 8, 8, 8]


# ============================================================
# Singleton
# ============================================================


def test_get_embedding_client_returns_singleton() -> None:
    a = get_embedding_client()
    b = get_embedding_client()
    assert a is b


def test_reset_embedding_client_for_tests_replaces_singleton() -> None:
    a = get_embedding_client()
    reset_embedding_client_for_tests()
    b = get_embedding_client()
    assert a is not b


# ============================================================
# Package surface
# ============================================================


def test_indexing_package_exposes_submodules() -> None:
    """Grep-style smoke check that the package surface is intact."""
    assert hasattr(indexing, "embedding_client")
