"""Identifier recall and repository isolation regressions."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from claim2code_api.indexing.identifiers import identifier_words
from claim2code_api.indexing.lexical import identifier_search
from claim2code_api.indexing.structural import structural_boost
from claim2code_api.retrieval.hybrid_ranker import hybrid_ranker


@pytest.mark.parametrize(("name", "expected"), [
    ("reset_parameters", {"reset", "parameters"}),
    ("MultiHeadAttention", {"multiheadattention", "multi", "head", "attention"}),
    ("HTTPResponseParser", {"http", "response", "parser"}),
    ("LoRALayer", {"lora", "layer"}),
])
def test_identifier_words_recovers_prose_terms(name, expected):
    assert expected <= identifier_words(name, min_length=3)


def test_identifier_query_is_parameterized_and_bounded():
    db = MagicMock()
    sid = uuid4()
    db.execute.return_value.fetchall.return_value = [SimpleNamespace(symbol_id=sid, score=0.5)]
    repo_id = uuid4()
    claim = "MultiHeadAttention'; DROP TABLE code_symbols; --"
    result = identifier_search(db, repo_id, claim, top_k=500)
    sql, params = db.execute.call_args.args
    assert claim not in str(sql)
    assert params["repository_id"] == str(repo_id)
    assert params["top_k"] == 50
    assert {"multi", "head", "attention"} <= set(params["claim"].split())
    assert result == [(sid, 0.5)]


def test_identifier_leg_short_tokens_do_not_query_database():
    db = MagicMock()
    assert identifier_search(db, uuid4(), "x r a") == []
    db.execute.assert_not_called()


def test_hybrid_recovers_undocumented_symbol_from_identifier_leg():
    sid = uuid4()
    db = MagicMock()
    # No dependency names; the only ORM fetch is the final symbol pool.
    db.query.return_value.filter.return_value.all.side_effect = [
        [], [SimpleNamespace(symbol_id=sid, kind="method", file_path="pkg/layers.py")],
    ]
    with (
        patch("claim2code_api.retrieval.hybrid_ranker.retrieve_semantic", return_value=[]),
        patch("claim2code_api.retrieval.hybrid_ranker.lexical_search", return_value=[]),
        patch("claim2code_api.retrieval.hybrid_ranker.identifier_search", return_value=[(sid, 0.8)]),
        patch("claim2code_api.retrieval.hybrid_ranker.structural_boost", return_value={sid: 1.0}),
    ):
        results = hybrid_ranker(db, uuid4(), "reset parameters")
    assert [r.symbol_id for r in results] == [sid]
    assert results[0].rank_structural == 1
    assert results[0].final_score == pytest.approx(0.2)


def test_structural_imports_cannot_leak_from_another_repository():
    own, foreign, sid = uuid4(), uuid4(), uuid4()
    candidate = SimpleNamespace(symbol_id=sid, repository_id=own, file_path="pkg/runner.py")
    other_import = SimpleNamespace(
        repository_id=foreign, file_path="pkg/runner.py", signature="torch", name="torch",
    )
    db = MagicMock()
    candidates_query, imports_query = MagicMock(), MagicMock()
    candidates_query.filter.return_value.all.return_value = [candidate]
    imports_query.filter.return_value = imports_query
    # Even a stale/mock session returning the foreign row must not boost it.
    imports_query.all.return_value = [other_import]
    db.query.side_effect = [candidates_query, imports_query]
    assert structural_boost(db, [sid], "torch computes predictions") == {sid: 1.0}


def test_structural_path_tokens_split_snake_case():
    sid, repo = uuid4(), uuid4()
    db = MagicMock()
    candidate = SimpleNamespace(
        symbol_id=sid, repository_id=repo, file_path="pkg/masked_attention.py",
    )
    candidates_query, imports_query = MagicMock(), MagicMock()
    candidates_query.filter.return_value.all.return_value = [candidate]
    imports_query.filter.return_value = imports_query
    imports_query.all.return_value = []
    db.query.side_effect = [candidates_query, imports_query]
    assert structural_boost(db, [sid], "masked attention") == {sid: pytest.approx(1.1)}
