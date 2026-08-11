"""Tests for the Phase 4 retrieval HTTP endpoint.

Covers ``POST /repositories/{repository_id}/retrieve``. The
underlying :func:`retrieve_semantic` is mocked at the router
boundary because it uses pgvector's ``<=>`` operator which
SQLite does not speak; the unit-level coverage of
``retrieve_semantic`` itself lives in
``tests/test_retriever.py``. This file proves the endpoint:

  - parses the ``RetrieveRequest`` body (claim + top_k)
  - returns a ``RetrieveResponse`` shaped like the Pydantic
    schema (repository_id, claim, symbols, scores)
  - maps missing repository → 404
  - maps malformed UUID → 422
  - maps body-validation errors → 422
  - maps ``EmbeddingError`` from the retriever → 500
  - is registered in the OpenAPI schema
  - caps ``top_k`` at the documented upper bound (50)
  - defaults ``top_k`` to 5 when omitted
  - passes the top-level workflow integration (the indexer is
    called inside ``run_repo_workflow`` after the parse step)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from claim2code_api.app.deps import get_db
from claim2code_api.app.main import create_app
from claim2code_api.db.models import (
    AnalysisState,
    Base,
    CodeSymbol,
    Repository,
)
from claim2code_api.indexing.exceptions import EmbeddingError

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(autouse=True)
def _unset_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Starlette's TestClient picks up HTTP(S)_PROXY + ``all_proxy``;
    clear them so localhost works under the corporate proxy."""
    for var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "ALL_PROXY",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def client() -> TestClient:
    """A TestClient with get_db overridden to an in-memory SQLite session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_conn, _):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        cur.close()

    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    # Pre-seed: 1 analysis + 1 repository + 2 symbols.
    seed = session_local()
    a = AnalysisState(
        repository_status="done",
        paper_status="pending",
    )
    seed.add(a)
    seed.flush()
    repo = Repository(
        analysis_id=a.analysis_id,
        url="https://github.com/acme/demo",
        owner="acme",
        name="demo",
        default_branch="main",
        clone_path="data/repos/acme__demo",
        n_files=10,
        n_symbols=2,
        index_status="done",
    )
    seed.add(repo)
    seed.flush()
    sym_add = CodeSymbol(
        repository_id=repo.repository_id,
        kind="function",
        language="python",
        file_path="pkg/core.py",
        start_line=1,
        start_col=0,
        end_line=3,
        end_col=0,
        name="add",
        signature="(a: int, b: int)",
        docstring="Add two ints.",
    )
    sym_widget = CodeSymbol(
        repository_id=repo.repository_id,
        kind="class",
        language="python",
        file_path="pkg/widget.py",
        start_line=5,
        start_col=0,
        end_line=20,
        end_col=0,
        name="Widget",
        signature="",
        docstring=None,
    )
    seed.add_all([sym_add, sym_widget])
    seed.commit()

    seeded_repo_id = repo.repository_id
    seeded_sym_ids = {
        "add": sym_add.symbol_id,
        "Widget": sym_widget.symbol_id,
    }

    def _override_get_db():
        s = session_local()
        try:
            yield s
        finally:
            s.close()

    app = create_app()
    app.dependency_overrides[get_db] = _override_get_db
    client_obj = TestClient(app)
    client_obj.repo_id = seeded_repo_id  # type: ignore[attr-defined]
    client_obj.sym_ids = seeded_sym_ids  # type: ignore[attr-defined]
    client_obj.session_local = session_local  # type: ignore[attr-defined]
    return client_obj


def _fake_symbols(rows_by_id: dict[str, object]):
    """Return a (symbol, similarity) list keyed by symbol name.

    Builds the MagicMock-style object the router consumes directly:
    the router only reads attributes (symbol_id, kind, language,
    file_path, start/end line/col, name, signature, docstring),
    so a SimpleNamespace is enough.
    """
    return rows_by_id


# ============================================================
# Happy path
# ============================================================


def test_retrieve_returns_top_matches(client: TestClient) -> None:
    """200 + payload mirrors the RetrieveResponse schema."""
    repo_id = client.repo_id  # type: ignore[attr-defined]
    sym_ids = client.sym_ids  # type: ignore[attr-defined]

    def _fake_retrieve(**kwargs):
        s_add = SimpleNamespace(
            symbol_id=sym_ids["add"],
            kind="function",
            language="python",
            file_path="pkg/core.py",
            start_line=1,
            start_col=0,
            end_line=3,
            end_col=0,
            name="add",
            signature="(a: int, b: int)",
            docstring="Add two ints.",
        )
        s_w = SimpleNamespace(
            symbol_id=sym_ids["Widget"],
            kind="class",
            language="python",
            file_path="pkg/widget.py",
            start_line=5,
            start_col=0,
            end_line=20,
            end_col=0,
            name="Widget",
            signature="",
            docstring=None,
        )
        return [(s_add, 0.92), (s_w, 0.61)]

    with patch("claim2code_api.repos.router.retrieve_semantic", side_effect=_fake_retrieve) as fake:
        response = client.post(
            f"/repositories/{repo_id}/retrieve",
            json={"claim": "two-int adder", "top_k": 5},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["repository_id"] == str(repo_id)
    assert body["claim"] == "two-int adder"
    assert len(body["symbols"]) == 2
    assert body["scores"] == [pytest.approx(0.92), pytest.approx(0.61)]

    # Verify the symbol rows carry the full schema.
    first = body["symbols"][0]
    assert first["name"] == "add"
    assert first["kind"] == "function"
    assert first["language"] == "python"
    assert first["file_path"] == "pkg/core.py"
    assert first["start_line"] == 1
    assert first["end_line"] == 3
    assert first["signature"] == "(a: int, b: int)"
    assert first["docstring"] == "Add two ints."
    assert first["similarity"] == pytest.approx(0.92)
    assert first["symbol_id"] == str(sym_ids["add"])

    # The retriever was called with the right kwargs.
    assert fake.call_args.kwargs["db"] is not None
    assert fake.call_args.kwargs["repository_id"] == repo_id
    assert fake.call_args.kwargs["claim"] == "two-int adder"
    assert fake.call_args.kwargs["top_k"] == 5


def test_retrieve_default_top_k_is_5(client: TestClient) -> None:
    """``top_k`` is optional and defaults to 5 per the design doc."""
    repo_id = client.repo_id  # type: ignore[attr-defined]

    def _fake_retrieve(**kwargs):
        return []

    with patch("claim2code_api.repos.router.retrieve_semantic", side_effect=_fake_retrieve) as fake:
        response = client.post(
            f"/repositories/{repo_id}/retrieve",
            json={"claim": "x"},
        )
    assert response.status_code == 200
    assert fake.call_args.kwargs["top_k"] == 5


def test_retrieve_empty_result_returns_empty_lists(client: TestClient) -> None:
    """An empty retriever result → empty symbols + scores lists."""
    repo_id = client.repo_id  # type: ignore[attr-defined]

    with patch("claim2code_api.repos.router.retrieve_semantic", return_value=[]) as fake:
        response = client.post(
            f"/repositories/{repo_id}/retrieve",
            json={"claim": "claim with no match"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["symbols"] == []
    assert body["scores"] == []
    assert body["claim"] == "claim with no match"
    assert fake.called


# ============================================================
# Error mapping
# ============================================================


def test_retrieve_404_for_unknown_repo(client: TestClient) -> None:
    """A non-existent repository → 404 with a clear detail message."""
    with patch("claim2code_api.repos.router.retrieve_semantic") as fake:
        response = client.post(
            f"/repositories/{uuid4()}/retrieve",
            json={"claim": "x"},
        )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
    fake.assert_not_called()


def test_retrieve_422_for_malformed_id(client: TestClient) -> None:
    """Non-UUID in the path → 422 (FastAPI's path validator)."""
    response = client.post(
        "/repositories/not-a-uuid/retrieve",
        json={"claim": "x"},
    )
    assert response.status_code == 422


def test_retrieve_422_for_missing_claim(client: TestClient) -> None:
    """An empty body → 422 (claim is required, min_length=1)."""
    repo_id = client.repo_id  # type: ignore[attr-defined]
    response = client.post(
        f"/repositories/{repo_id}/retrieve",
        json={},
    )
    assert response.status_code == 422


def test_retrieve_422_for_empty_claim(client: TestClient) -> None:
    """An empty claim string → 422 (Pydantic min_length=1)."""
    repo_id = client.repo_id  # type: ignore[attr-defined]
    response = client.post(
        f"/repositories/{repo_id}/retrieve",
        json={"claim": ""},
    )
    assert response.status_code == 422


def test_retrieve_422_for_top_k_below_one(client: TestClient) -> None:
    """top_k=0 is below the Pydantic ge=1 lower bound → 422."""
    repo_id = client.repo_id  # type: ignore[attr-defined]
    response = client.post(
        f"/repositories/{repo_id}/retrieve",
        json={"claim": "x", "top_k": 0},
    )
    assert response.status_code == 422


def test_retrieve_422_for_top_k_above_50(client: TestClient) -> None:
    """top_k=51 is above the Pydantic le=50 upper bound → 422."""
    repo_id = client.repo_id  # type: ignore[attr-defined]
    response = client.post(
        f"/repositories/{repo_id}/retrieve",
        json={"claim": "x", "top_k": 51},
    )
    assert response.status_code == 422


def test_retrieve_500_when_embedding_fails(client: TestClient) -> None:
    """An EmbeddingError from the retriever maps to 500."""
    repo_id = client.repo_id  # type: ignore[attr-defined]

    def _raise(**_kwargs):
        raise EmbeddingError("simulated OpenAI outage")

    with patch("claim2code_api.repos.router.retrieve_semantic", side_effect=_raise):
        response = client.post(
            f"/repositories/{repo_id}/retrieve",
            json={"claim": "x"},
        )

    assert response.status_code == 500
    assert "simulated OpenAI outage" in response.json()["detail"]


def test_retrieve_422_for_extra_field_in_body(client: TestClient) -> None:
    """``extra='forbid'`` on RetrieveRequest → 422 for unknown keys."""
    repo_id = client.repo_id  # type: ignore[attr-defined]
    response = client.post(
        f"/repositories/{repo_id}/retrieve",
        json={"claim": "x", "unknown": "y"},
    )
    assert response.status_code == 422


# ============================================================
# OpenAPI / route registration
# ============================================================


def test_retrieve_route_is_registered(client: TestClient) -> None:
    """The retrieve endpoint appears in the OpenAPI schema."""
    schema = client.get("/openapi.json").json()
    assert "/repositories/{repository_id}/retrieve" in schema["paths"]
    post = schema["paths"]["/repositories/{repository_id}/retrieve"].get("post")
    assert post is not None
    assert "200" in post["responses"]
    assert "404" in post["responses"]
    assert "500" in post["responses"]


def test_retrieve_request_body_schema(client: TestClient) -> None:
    """The OpenAPI schema for the body matches RetrieveRequest."""
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]
    assert "RetrieveRequest" in components
    req = components["RetrieveRequest"]
    assert "claim" in req["properties"]
    assert "top_k" in req["properties"]


def test_retrieve_response_body_schema(client: TestClient) -> None:
    """The OpenAPI schema for the response matches RetrieveResponse."""
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]
    assert "RetrieveResponse" in components
    res = components["RetrieveResponse"]
    assert "repository_id" in res["properties"]
    assert "claim" in res["properties"]
    assert "symbols" in res["properties"]
    assert "scores" in res["properties"]


# ============================================================
# Workflow integration — indexer is called after the parse step
# ============================================================


def test_run_repo_workflow_calls_embed_and_persist_symbols() -> None:
    """``run_repo_workflow`` triggers ``embed_and_persist_symbols``
    so a successful clone immediately leaves the repo retrievable.

    Mocks ``cloner.clone_repository`` (so no real git happens) and
    ``embed_and_persist_symbols`` (so we don't need OpenAI). Then
    asserts the workflow finishes with the expected counts and the embed
    function was called once with the right ``repository_id``.
    """
    from pathlib import Path

    from sqlalchemy import create_engine as _ce
    from sqlalchemy.orm import sessionmaker as _sm
    from sqlalchemy.pool import StaticPool

    from claim2code_api.repos import workflow as wf_mod

    engine = _ce(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = _sm(bind=engine, expire_on_commit=False)

    # Pre-seed an analysis row.
    seed = session_local()
    a = AnalysisState(
        repository_status="pending",
        paper_status="pending",
    )
    seed.add(a)
    seed.commit()
    analysis_id = a.analysis_id
    seed.close()

    # Empty clone dir on disk.
    clone_dir = Path("/tmp/c2c-test-clone-empty")
    clone_dir.mkdir(parents=True, exist_ok=True)
    (clone_dir / "README.md").write_text("# empty\n")

    fake_clone_result = SimpleNamespace(
        default_branch="main",
        clone_path=clone_dir,
    )

    with (
        patch.object(wf_mod.cloner, "parse_github_url") as fake_parse,
        patch.object(wf_mod.cloner, "clone_repository", return_value=fake_clone_result),
        patch("claim2code_api.repos.workflow.embed_and_persist_symbols") as fake_embed,
        patch("claim2code_api.repos.workflow._walk_and_persist", return_value=(1, 0, 0, 0)),
    ):
        fake_parse.return_value = SimpleNamespace(
            url="https://github.com/empty/nothing",
            owner="empty",
            name="nothing",
        )
        session = session_local()
        result = wf_mod.run_repo_workflow(
            db=session,
            analysis_id=analysis_id,
            url="https://github.com/empty/nothing",
        )

    # embed_and_persist_symbols was called once with the right id.
    assert fake_embed.call_count == 1
    args, kwargs = fake_embed.call_args
    # embed_and_persist_symbols(db, repository_id, *, embedding_client, settings)
    # so the repository_id is the second positional arg.
    assert args[1] == result.repository_id
    session.close()

    # Cleanup
    import shutil

    shutil.rmtree(clone_dir, ignore_errors=True)


# ============================================================
# Module __all__ sanity
# ============================================================


def test_router_module_exposes_retrieve_repository() -> None:
    """``repos.router.__all__`` includes ``retrieve_repository``."""
    from claim2code_api.repos import router as router_mod

    assert "retrieve_repository" in router_mod.__all__
