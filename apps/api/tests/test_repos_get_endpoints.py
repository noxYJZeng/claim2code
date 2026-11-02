"""Tests for the read-side repository HTTP endpoints.

Mirrors ``test_papers_get_endpoints.py`` — uses an in-memory SQLite
DB so the tests exercise the full ORM ↔ schema round-trip without
needing a real Postgres. ``StaticPool`` keeps the seeded data
visible to the dependency-overridden ``get_db``.
"""

from __future__ import annotations

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
    RepositoryConfig,
    RepositoryDependency,
)

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
    # SQLite ships with foreign-key enforcement OFF by default; turn
    # it on so the CASCADE deletes on ``repositories`` actually wipe
    # child rows during the DELETE endpoint tests. (Postgres enables
    # FKs unconditionally.)
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_conn, _):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        cur.close()

    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    # Pre-seed: 1 analysis + 1 repository + 2 symbols + 1 config + 2 deps.
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
    seed.add_all(
        [
            CodeSymbol(
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
            ),
            CodeSymbol(
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
            ),
        ]
    )
    seed.add(
        RepositoryConfig(
            repository_id=repo.repository_id,
            file_path="config.yaml",
            format="yaml",
            size_bytes=42,
            content_json={"name": "demo", "version": "0.1"},
            error=None,
        )
    )
    seed.add_all(
        [
            RepositoryDependency(
                repository_id=repo.repository_id,
                ecosystem="pypi",
                name="requests",
                version_spec="==2.31.0",
                source_file="requirements.txt",
            ),
            RepositoryDependency(
                repository_id=repo.repository_id,
                ecosystem="pypi",
                name="numpy",
                version_spec=">=1.26",
                source_file="requirements.txt",
            ),
        ]
    )
    seed.commit()

    seeded_repo_id = repo.repository_id

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
    client_obj.session_local = session_local  # type: ignore[attr-defined]
    return client_obj


# ============================================================
# GET /repositories/{repository_id}
# ============================================================


def test_get_repository_returns_metadata(client: TestClient) -> None:
    response = client.get(f"/repositories/{client.repo_id}")  # type: ignore[attr-defined]
    assert response.status_code == 200
    body = response.json()
    assert body["owner"] == "acme"
    assert body["name"] == "demo"
    assert body["url"] == "https://github.com/acme/demo"
    assert body["default_branch"] == "main"
    assert body["clone_path"] == "data/repos/acme__demo"
    assert body["n_files"] == 10
    assert body["n_symbols"] == 2
    assert body["index_status"] == "done"
    assert body["cloned_at"] is not None
    assert body["indexed_at"] is None  # not set by fixture
    assert body["error"] is None


def test_get_repository_404_for_unknown_id(client: TestClient) -> None:
    response = client.get(f"/repositories/{uuid4()}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_repository_422_for_malformed_id(client: TestClient) -> None:
    response = client.get("/repositories/not-a-uuid")
    assert response.status_code == 422


def test_get_analysis_repository_returns_the_bound_repo(client: TestClient) -> None:
    with client.session_local() as s:  # type: ignore[attr-defined]
        analysis_id = s.get(Repository, client.repo_id).analysis_id  # type: ignore[attr-defined]
    response = client.get(f"/analyses/{analysis_id}/repository")
    assert response.status_code == 200
    assert response.json()["repository_id"] == str(client.repo_id)  # type: ignore[attr-defined]
    assert client.get(f"/analyses/{uuid4()}/repository").status_code == 404


def test_get_repository_route_is_registered(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/repositories/{repository_id}" in schema["paths"]


# ============================================================
# GET /repositories/{repository_id}/symbols
# ============================================================


def test_get_symbols_returns_all_rows(client: TestClient) -> None:
    response = client.get(f"/repositories/{client.repo_id}/symbols")  # type: ignore[attr-defined]
    assert response.status_code == 200
    body = response.json()
    assert body["repository_id"] == str(client.repo_id)  # type: ignore[attr-defined]
    names = {s["name"] for s in body["symbols"]}
    assert names == {"add", "Widget"}
    assert len(body["symbols"]) == 2


def test_get_symbols_carries_full_schema(client: TestClient) -> None:
    response = client.get(f"/repositories/{client.repo_id}/symbols")  # type: ignore[attr-defined]
    sym = next(s for s in response.json()["symbols"] if s["name"] == "add")
    assert sym["kind"] == "function"
    assert sym["language"] == "python"
    assert sym["file_path"] == "pkg/core.py"
    assert sym["start_line"] == 1
    assert sym["end_line"] == 3
    assert sym["signature"] == "(a: int, b: int)"
    assert sym["docstring"] == "Add two ints."


def test_get_symbols_returns_empty_list_for_repo_without_symbols(
    client: TestClient,
) -> None:
    """A freshly-cloned repo with no source files returns []. Add a
    second repo with no symbols attached."""
    session = client.session_local()  # type: ignore[attr-defined]
    a = (
        session.query(AnalysisState)
        .filter(AnalysisState.analysis_id == AnalysisState.analysis_id)
        .first()
    )
    empty_repo = Repository(
        analysis_id=a.analysis_id,
        url="https://github.com/empty/nope",
        owner="empty",
        name="nope",
        default_branch="main",
        clone_path="data/repos/empty__nope",
        n_files=0,
        n_symbols=0,
        index_status="done",
    )
    session.add(empty_repo)
    session.commit()
    empty_id = empty_repo.repository_id
    session.close()

    response = client.get(f"/repositories/{empty_id}/symbols")
    assert response.status_code == 200
    assert response.json()["symbols"] == []


def test_get_symbols_404_for_unknown_repo(client: TestClient) -> None:
    response = client.get(f"/repositories/{uuid4()}/symbols")
    assert response.status_code == 404


def test_get_symbols_route_is_registered(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/repositories/{repository_id}/symbols" in schema["paths"]


# ============================================================
# GET /repositories/{repository_id}/configs
# ============================================================


def test_get_configs_returns_all_rows(client: TestClient) -> None:
    response = client.get(f"/repositories/{client.repo_id}/configs")  # type: ignore[attr-defined]
    assert response.status_code == 200
    body = response.json()
    assert body["repository_id"] == str(client.repo_id)  # type: ignore[attr-defined]
    assert len(body["configs"]) == 1
    cfg = body["configs"][0]
    assert cfg["file_path"] == "config.yaml"
    assert cfg["format"] == "yaml"
    assert cfg["size_bytes"] == 42
    assert cfg["content_json"] == {"name": "demo", "version": "0.1"}
    assert cfg["error"] is None


def test_get_configs_404_for_unknown_repo(client: TestClient) -> None:
    response = client.get(f"/repositories/{uuid4()}/configs")
    assert response.status_code == 404


def test_get_configs_route_is_registered(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/repositories/{repository_id}/configs" in schema["paths"]


# ============================================================
# GET /repositories/{repository_id}/dependencies
# ============================================================


def test_get_dependencies_returns_all_rows(client: TestClient) -> None:
    response = client.get(
        f"/repositories/{client.repo_id}/dependencies"  # type: ignore[attr-defined]
    )
    assert response.status_code == 200
    body = response.json()
    by_name = {d["name"]: d for d in body["dependencies"]}
    assert "requests" in by_name
    assert "numpy" in by_name
    assert by_name["requests"]["ecosystem"] == "pypi"
    assert by_name["requests"]["version_spec"] == "==2.31.0"
    assert by_name["requests"]["source_file"] == "requirements.txt"
    assert by_name["numpy"]["version_spec"] == ">=1.26"


def test_get_dependencies_404_for_unknown_repo(client: TestClient) -> None:
    response = client.get(f"/repositories/{uuid4()}/dependencies")
    assert response.status_code == 404


def test_get_dependencies_route_is_registered(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/repositories/{repository_id}/dependencies" in schema["paths"]


# ============================================================
# DELETE /repositories/{repository_id}
# ============================================================


def test_delete_repository_removes_row(client: TestClient) -> None:
    """A real clone directory under ``data/repos/`` is removed
    best-effort; the DB row is gone afterwards. We stub
    ``cloner.remove_clone`` so the test doesn't actually touch the
    real on-disk clone (which doesn't exist in this fixture)."""
    repo_id = client.repo_id  # type: ignore[attr-defined]

    with patch("claim2code_api.repos.router.cloner.remove_clone") as fake_rm:
        response = client.delete(f"/repositories/{repo_id}")

    assert response.status_code == 204
    assert response.content == b""  # No Content
    fake_rm.assert_called_once()

    # Subsequent GET should now 404.
    assert client.get(f"/repositories/{repo_id}").status_code == 404
    assert client.get(f"/repositories/{repo_id}/symbols").status_code == 404
    assert client.get(f"/repositories/{repo_id}/configs").status_code == 404
    assert client.get(f"/repositories/{repo_id}/dependencies").status_code == 404


def test_delete_repository_cascades_symbols_and_configs(
    client: TestClient,
) -> None:
    """After DELETE, every child endpoint returns 404 for the same
    repository id. The actual CASCADE behaviour (Postgres wipes
    child rows via ON DELETE CASCADE) is verified in the
    integration migration test — here we just confirm the app
    routes consistently report "gone" once the row is deleted."""
    repo_id = client.repo_id  # type: ignore[attr-defined]

    with patch("claim2code_api.repos.router.cloner.remove_clone"):
        response = client.delete(f"/repositories/{repo_id}")
    assert response.status_code == 204

    # Every read endpoint now reports 404 for the same id.
    assert client.get(f"/repositories/{repo_id}").status_code == 404
    assert client.get(f"/repositories/{repo_id}/symbols").status_code == 404
    assert client.get(f"/repositories/{repo_id}/configs").status_code == 404
    assert client.get(f"/repositories/{repo_id}/dependencies").status_code == 404


def test_delete_repository_404_for_unknown_id(client: TestClient) -> None:
    response = client.delete(f"/repositories/{uuid4()}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_delete_repository_logs_disk_cleanup_failure(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """When ``cloner.remove_clone`` raises OSError, the DB row is
    still deleted and the failure is logged."""
    repo_id = client.repo_id  # type: ignore[attr-defined]

    with patch(
        "claim2code_api.repos.router.cloner.remove_clone",
        side_effect=OSError("permission denied"),
    ):
        with caplog.at_level("WARNING", logger="claim2code_api.repos.router"):
            response = client.delete(f"/repositories/{repo_id}")

    assert response.status_code == 204
    assert any("clone cleanup failed" in m for m in caplog.messages)
    # DB row is gone.
    assert client.get(f"/repositories/{repo_id}").status_code == 404


def test_delete_repository_route_is_registered(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/repositories/{repository_id}" in schema["paths"]
    assert "delete" in schema["paths"]["/repositories/{repository_id}"]


# ============================================================
# 404 shape — all endpoints use the same RepoErrorResponse body
# ============================================================


def test_404_detail_format_is_consistent_across_endpoints(client: TestClient) -> None:
    unknown = uuid4()
    for endpoint in ("", "/symbols", "/configs", "/dependencies"):
        response = client.get(f"/repositories/{unknown}{endpoint}")
        assert response.status_code == 404, endpoint
        body = response.json()
        assert "detail" in body
        assert str(unknown) in body["detail"]
        assert "not found" in body["detail"]


# Touch imports so ruff doesn't drop them.
_uuid = uuid4()
_ = _uuid
