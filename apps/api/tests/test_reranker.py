"""LLM reranking of hybrid candidates (retrieval/reranker.py, LLMClient.rank_candidates)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from openai import APITimeoutError

from claim2code_api.app.config import get_settings
from claim2code_api.llm.client import LLMClient, RerankLLMError
from claim2code_api.retrieval import reranker
from claim2code_api.retrieval.hybrid_ranker import HybridMatch
from claim2code_api.verification import workflow

# ---------------------------------------------------------------- client


def _client(create: MagicMock) -> LLMClient:
    from claim2code_api.safety import SafetyGate

    client = LLMClient.__new__(LLMClient)
    client._settings = get_settings()
    client._client = MagicMock()
    client._client.chat.completions.create = create
    client._model = "m"
    client._gate = SafetyGate()
    client._on_violations = None
    return client


def _response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.usage = None
    return resp


def test_rank_candidates_keeps_distinct_in_range_numbers_in_order() -> None:
    create = MagicMock(return_value=_response(json.dumps({"ranking": [3, 0, 3, 9, 1, "2", 2]})))
    assert _client(create).rank_candidates(claim="c", candidates="[1] ...", n_candidates=5) == [3, 1, 2]
    kwargs = create.call_args.kwargs
    assert kwargs["temperature"] == 0
    assert kwargs["response_format"]["json_schema"]["name"] == "ranking"
    assert kwargs["messages"][1]["content"].startswith("Claim: c\n\nCandidates:")


@pytest.mark.parametrize("failure", ["timeout", "not_json", "no_key", "not_list"])
def test_rank_candidates_raises_rerank_errors(failure: str) -> None:
    create = MagicMock()
    if failure == "timeout":
        create.side_effect = APITimeoutError(request=MagicMock())
    else:
        create.return_value = _response(
            {"not_json": "ranking: 1", "no_key": "{}", "not_list": '{"ranking": 3}'}[failure],
        )
    with pytest.raises(RerankLLMError):
        _client(create).rank_candidates(claim="c", candidates="x", n_candidates=3)


# ---------------------------------------------------------------- rerank


def _symbols(n: int):
    return [
        SimpleNamespace(
            symbol_id=uuid4(), kind="function", name=f"f{i}", file_path=f"pkg/m{i}.py",
            start_line=10 * i + 1, end_line=10 * i + 5, docstring=f"Doc {i}.\n  more",
        )
        for i in range(n)
    ]


def _db(symbols, clone_path: str | None = None) -> MagicMock:
    db = MagicMock()
    db.get.return_value = SimpleNamespace(clone_path=clone_path)
    db.query.return_value.filter.return_value.all.return_value = symbols
    return db


class _Ranker:
    def __init__(self, picks=None, error: Exception | None = None) -> None:
        self.picks, self.error = picks or [], error
        self.calls: list[tuple[str, str, int]] = []

    def rank_candidates(self, *, claim: str, candidates: str, n_candidates: int) -> list[int]:
        self.calls.append((claim, candidates, n_candidates))
        if self.error:
            raise self.error
        return self.picks


def test_rerank_puts_the_models_picks_first_and_keeps_the_rest_in_order() -> None:
    symbols = _symbols(5)
    matches = [HybridMatch(symbol_id=s.symbol_id, final_score=1.0 - i / 10) for i, s in enumerate(symbols)]
    ranker = _Ranker(picks=[4, 2])
    out = reranker.rerank(_db(symbols), uuid4(), "claim", matches, ranker=ranker)
    assert [m.symbol_id for m in out] == [symbols[i].symbol_id for i in (3, 1, 0, 2, 4)]
    assert [m.rank_rerank for m in out] == [1, 2, None, None, None]
    _, listing, n = ranker.calls[0]
    assert n == 5
    assert "[4] function f3 — pkg/m3.py:31-35\ndoc: Doc 3. more" in listing


def test_rerank_shows_source_heads_from_the_clone(tmp_path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m0.py").write_text("def f0(x):\n    return loss(x)\n" + "\n" * 3)
    symbols = _symbols(2)
    symbols[0].start_line, symbols[0].end_line = 1, 2
    matches = [HybridMatch(symbol_id=s.symbol_id, final_score=1.0) for s in symbols]
    ranker = _Ranker(picks=[1])
    reranker.rerank(_db(symbols, str(tmp_path)), uuid4(), "claim", matches, ranker=ranker)
    assert "def f0(x):\n    return loss(x)" in ranker.calls[0][1]


@pytest.mark.parametrize("error", [RerankLLMError("down"), RuntimeError("anything")])
def test_rerank_failure_keeps_the_hybrid_order(error: Exception) -> None:
    symbols = _symbols(3)
    matches = [HybridMatch(symbol_id=s.symbol_id, final_score=1.0) for s in symbols]
    out = reranker.rerank(_db(symbols), uuid4(), "claim", matches, ranker=_Ranker(error=error))
    assert out == matches


def test_workflow_reranks_a_wider_pool_then_cuts(monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANK", "true")
    get_settings.cache_clear()
    symbols = _symbols(reranker.POOL_SIZE)
    pool = [HybridMatch(symbol_id=s.symbol_id, final_score=1.0) for s in symbols]
    sizes: list[int] = []

    def fake_hybrid(**kwargs):
        sizes.append(kwargs["top_k"])
        return pool

    monkeypatch.setattr(workflow, "hybrid_ranker", fake_hybrid)
    monkeypatch.setattr(workflow, "get_embedding_client", MagicMock())
    monkeypatch.setattr(workflow, "get_llm_client", lambda: _Ranker(picks=[45, 7]))
    out = workflow._retrieve(_db(symbols), uuid4(), "claim", top_k=8)
    assert sizes == [reranker.POOL_SIZE]
    assert len(out) == 8
    assert [m.symbol_id for m in out[:3]] == [symbols[44].symbol_id, symbols[6].symbol_id, symbols[0].symbol_id]


def test_workflow_without_rerank_asks_hybrid_for_top_k(monkeypatch) -> None:
    sizes: list[int] = []
    def fake_hybrid(**kwargs):
        sizes.append(kwargs["top_k"])
        return []

    monkeypatch.setattr(workflow, "hybrid_ranker", fake_hybrid)
    monkeypatch.setattr(workflow, "get_embedding_client", MagicMock())
    assert workflow._retrieve(MagicMock(), uuid4(), "claim", top_k=8) == []
    assert sizes == [8]
