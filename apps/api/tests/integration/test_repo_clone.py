"""Real-``git clone`` integration test for the repository workflow.

The Phase 3 design doc says (§4 CI considerations):

  > Integration test (real ``git clone``) is opt-in:
  > ``@pytest.mark.skipif(not os.getenv("GITHUB_TOKEN"))`` so CI
  > collects and skips without a token.

This test follows that rule: it requires both

  * ``GITHUB_TOKEN`` env var (raised unauthenticated GitHub rate
    limit from 60/h to 5000/h)
  * a real Postgres+pgvector (the ``DATABASE_URL`` env var)

Run with::

    GITHUB_TOKEN=ghp_... pytest -m integration tests/integration/test_repo_clone.py

The default target is a tiny public repo (``octocat/Hello-World``,
four files, < 10 KB on disk). The test runs the full
``run_repo_workflow`` synchronously and asserts every DB row
materialises.
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from claim2code_api.app.config import get_settings
from claim2code_api.db.models import AnalysisState, Repository
from claim2code_api.repos.workflow import run_repo_workflow

# Combine markers: ``integration`` opts the whole module in under
# ``pytest -m integration``; ``skipif`` deselects the module unless
# ``GITHUB_TOKEN`` is set (without it, GitHub caps unauthenticated
# clones at 60/h, which is fine for a one-off but not for nightly CI).
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("GITHUB_TOKEN"),
        reason="GITHUB_TOKEN not set; skipping real git clone integration",
    ),
]


@pytest.fixture
def db_session():
    engine = create_engine(get_settings().database_url, future=True)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    s = session_local()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def analysis_id(db_session) -> str:
    a = AnalysisState(
        repository_status="pending",
        paper_status="pending",
    )
    db_session.add(a)
    db_session.commit()
    return str(a.analysis_id)


def test_real_clone_hello_world_persists_rows(
    db_session, analysis_id: str
) -> None:
    """End-to-end against ``octocat/Hello-World``: clone, walk,
    parse, persist, assert the DB has at least one repo row +
    (possibly) some symbols / configs.

    ``octocat/Hello-World`` is a demo README-only repo. The
    assertion is intentionally tolerant (``>= 1`` files indexed)
    so we don't break if upstream refactors. We DO assert the
    repository row exists and is in a terminal state.
    """
    result = run_repo_workflow(
        db=db_session,
        analysis_id=uuid.UUID(analysis_id),
        url="https://github.com/octocat/Hello-World",
    )
    assert result.index_status == "done"
    assert result.n_files >= 1

    repo_row = (
        db_session.query(Repository)
        .filter(Repository.repository_id == result.repository_id)
        .one()
    )
    assert repo_row.owner == "octocat"
    assert repo_row.name == "Hello-World"
    assert repo_row.index_status == "done"
    assert repo_row.n_files >= 1

    # The analysis row is updated by the workflow.
    state = (
        db_session.query(AnalysisState)
        .filter(AnalysisState.analysis_id == uuid.UUID(analysis_id))
        .one()
    )
    assert state.repository_status == "done"
    assert state.completed_at is not None

    # Cleanup so repeated runs don't accumulate rows.
    db_session.delete(repo_row)
    db_session.delete(state)
    db_session.commit()


def test_real_clone_unknown_repo_raises_clone_error(
    db_session, analysis_id: str
) -> None:
    """``git clone`` against a 404 returns a non-zero exit code; the
    cloner wraps that in ``RepoCloneError`` and the workflow
    rolls back + records an error."""
    from claim2code_api.repos.exceptions import RepoCloneError

    with pytest.raises(RepoCloneError):
        run_repo_workflow(
            db=db_session,
            analysis_id=uuid.UUID(analysis_id),
            url="https://github.com/octocat/this-repo-does-not-exist-1234567890",
        )

    state = (
        db_session.query(AnalysisState)
        .filter(AnalysisState.analysis_id == uuid.UUID(analysis_id))
        .one()
    )
    assert state.repository_status == "error"
    assert state.errors
    assert state.errors[0]["type"] == "RepoCloneError"

    # Cleanup
    db_session.delete(state)
    db_session.commit()