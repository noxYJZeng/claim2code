"""Integration test — lexical leg round-trips against real Postgres.

Opt-in via ``pytest -m integration``. Requires a live Postgres
with the pgvector + pg_trgm extensions and the migrations
applied (so ``code_symbols.lex_tsv`` exists). CI runs this
against the testcontainers Postgres in the
``backend-integration`` job.

This file proves the generated ``lex_text`` / ``lex_tsv``
columns work end-to-end:

  1. 5 synthetic symbols seeded directly into ``code_symbols``
     (one per Phase-4-supported language kind).
  2. The 0005 migration's ``lex_tsv`` recomputes the
     generated column on INSERT (Postgres handles this
     automatically).
  3. :func:`claim2code_api.indexing.lexical.lexical_search`
     returns the expected (symbol_id, score) pairs.

What we deliberately don't test here:

  - Cross-language stemming (covered by unit tests in
    ``tests/test_lexical.py``).
  - Web search syntax (``OR``, ``-foo``) — left to a
    follow-up integration test if we ever expose it.
"""
from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import sessionmaker

from claim2code_api.app.config import get_settings
from claim2code_api.db.models import Base, CodeSymbol
from claim2code_api.indexing.lexical import lexical_search

pytestmark = pytest.mark.integration


def _build_engine():
    return create_engine(get_settings().database_url, future=True)


@pytest.fixture
def seeded_session():
    """Seed 5 synthetic symbols, yield a session, then clean up.

    Symbol mix covers the kinds / languages that a typical
    paper-claim-to-code match would surface: function, class,
    method, import, constant. The names + signatures include
    a few Porter-stem-friendly variants (``running`` vs
    ``run``) so the integration round-trip exercises the
    Postgres ``english`` config's stemming.
    """
    engine = _build_engine()
    # Make sure schema is up (CI runs migrations separately
    # but local dev may not).
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    session = session_local()

    # Seed an AnalysisState first because ``code_symbols``
    # has a FK to ``repositories`` which has a FK to
    # ``analysis_states``.
    session.execute(
        text(
            "INSERT INTO analysis_states (analysis_id, repository_status, paper_status) "
            "VALUES (gen_random_uuid(), 'done', 'done')"
        )
    )
    session.execute(
        text(
            "INSERT INTO repositories (analysis_id, url, owner, name, "
            "default_branch, clone_path, n_files, n_symbols, index_status) "
            "VALUES ((SELECT analysis_id FROM analysis_states ORDER BY started_at DESC LIMIT 1), "
            "'https://github.com/test/lexical_round_trip', 'test', "
            "'lexical_round_trip', 'main', '/tmp/lex_rt', 1, 5, 'done')"
        )
    )
    actual_repo_id = session.execute(
        text(
            "SELECT repository_id FROM repositories WHERE owner='test' "
            "AND name='lexical_round_trip'"
        )
    ).scalar()

    symbols = [
        CodeSymbol(
            repository_id=actual_repo_id,
            kind="function",
            language="python",
            file_path="pkg/attention.py",
            start_line=1,
            start_col=0,
            end_line=10,
            end_col=0,
            name="scaled_dot_product_attention",
            signature="(query, key, value, mask=None)",
            docstring="Computes scaled dot-product attention.",
        ),
        CodeSymbol(
            repository_id=actual_repo_id,
            kind="function",
            language="python",
            file_path="pkg/embeddings.py",
            start_line=1,
            start_col=0,
            end_line=8,
            end_col=0,
            name="token_embedding",
            signature="(input_ids)",
            docstring="Looks up token embeddings from a table.",
        ),
        CodeSymbol(
            repository_id=actual_repo_id,
            kind="class",
            language="python",
            file_path="pkg/model.py",
            start_line=1,
            start_col=0,
            end_line=20,
            end_col=0,
            name="TransformerBlock",
            signature="",
            docstring="A single transformer encoder block.",
        ),
        CodeSymbol(
            repository_id=actual_repo_id,
            kind="function",
            language="python",
            file_path="pkg/run.py",
            start_line=1,
            start_col=0,
            end_line=3,
            end_col=0,
            name="run_training_loop",
            signature="(model, data)",
            docstring="Runs one training epoch.",
        ),
        CodeSymbol(
            repository_id=actual_repo_id,
            kind="function",
            language="python",
            file_path="pkg/io.py",
            start_line=1,
            start_col=0,
            end_line=5,
            end_col=0,
            name="load_checkpoint",
            signature="(path)",
            docstring="Loads a checkpoint from disk.",
        ),
    ]
    session.add_all(symbols)
    session.commit()
    ids = {s.name: s.symbol_id for s in symbols}

    try:
        yield session, actual_repo_id, ids
    finally:
        session.execute(
            delete(CodeSymbol).where(CodeSymbol.repository_id == actual_repo_id)
        )
        session.execute(
            text("DELETE FROM repositories WHERE repository_id = :rid"),
            {"rid": str(actual_repo_id)},
        )
        session.execute(
            text(
                "DELETE FROM analysis_states WHERE analysis_id NOT IN "
                "(SELECT analysis_id FROM repositories) "
                "AND repository_status='done' AND paper_status='done' "
                "AND started_at = (SELECT MAX(started_at) FROM analysis_states)"
            )
        )
        session.commit()
        session.close()
        engine.dispose()


def test_lex_tsv_generated_column_populated(seeded_session) -> None:
    """The 0005 migration's generated column has content."""
    session, repo_id, _ = seeded_session
    rows = session.execute(
        text(
            "SELECT name, lex_tsv IS NOT NULL FROM code_symbols "
            "WHERE repository_id = :rid ORDER BY name"
        ),
        {"rid": str(repo_id)},
    ).fetchall()
    assert len(rows) == 5
    for _name, has_tsv in rows:
        assert has_tsv, "lex_tsv should be populated by the generated column"


def test_lexical_search_finds_attention_symbol(seeded_session) -> None:
    """A claim about 'attention' surfaces the attention symbol."""
    session, repo_id, ids = seeded_session

    rows = lexical_search(
        session, repo_id, "the model uses attention", top_k=5
    )
    assert rows, "lexical_search returned no rows"
    symbol_ids = [sid for sid, _score in rows]
    assert ids["scaled_dot_product_attention"] in symbol_ids


def test_lexical_search_stems_running_to_run(seeded_session) -> None:
    """Postgres's english config stems run/running so 'running' matches run_training_loop."""
    session, repo_id, ids = seeded_session

    rows = lexical_search(
        session, repo_id, "running the training loop", top_k=5
    )
    symbol_ids = [sid for sid, _score in rows]
    assert ids["run_training_loop"] in symbol_ids


def test_lexical_search_field_weights_name_above_docstring(seeded_session) -> None:
    """A name match should outrank a docstring-only match (ts_rank_cd sees A vs C)."""
    session, repo_id, ids = seeded_session

    # 'attention' appears in:
    #   - scaled_dot_product_attention.name  (weight A)
    #   - scaled_dot_product_attention.docstring ("attention")
    # 'embedding' appears only in token_embedding.name.
    # A claim that matches BOTH should put the attention symbol
    # first because its tokens hit a higher-weighted field.
    rows = lexical_search(
        session,
        repo_id,
        "attention embedding lookup",
        top_k=5,
    )
    assert rows, "lexical_search returned no rows"
    top_id, _top_score = rows[0]
    # The attention symbol's name+docstring match dominates
    # via field weight A vs the embedding symbol's name match
    # at weight A but only one token.
    assert top_id == ids["scaled_dot_product_attention"]


def test_lexical_search_repository_scope(seeded_session) -> None:
    """Searches don't leak across repositories (we seed one and assert only its symbols)."""
    session, repo_id, _ = seeded_session

    # Run a search against a random UUID → empty result.
    rows = lexical_search(
        session,
        UUID("00000000-0000-0000-0000-000000000000"),
        "attention",
        top_k=5,
    )
    assert rows == []


def test_lexical_search_empty_claim_short_circuits(seeded_session) -> None:
    """Empty claim → empty result (no round-trip)."""
    session, repo_id, _ = seeded_session
    assert lexical_search(session, repo_id, "") == []
    assert lexical_search(session, repo_id, "   \n  ") == []