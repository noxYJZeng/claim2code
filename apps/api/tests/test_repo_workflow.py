"""Unit tests for :mod:`claim2code_api.repos.workflow` + the
``POST /repositories`` endpoint.

These tests mirror the Phase 2 ``test_paper_workflow.py`` pattern:
the workflow's disk + git + parse dependencies are mocked, but the
DB session is a real SQLAlchemy ``Session`` against a per-test
in-memory SQLite (via the ``session`` fixture) so we exercise the
ORM columns, the FK CASCADE, and the JSONB-shaped serialisation
that the real Postgres deployment will see.

The HTTP layer tests additionally mock ``run_repo_workflow`` so the
router is verified independently of the orchestrator.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from claim2code_api.app.config import get_settings
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
from claim2code_api.repos import cloner
from claim2code_api.repos.exceptions import (
    RepoAlreadyExistsError,
    RepoCloneError,
    RepoInvalidUrlError,
    RepoTooLargeError,
)
from claim2code_api.repos.workflow import (
    RepoWorkflowResult,
    run_repo_workflow,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def engine():
    """In-memory SQLite engine with the Phase 3 schema migrated.

    We re-create the engine per test so session isolation is automatic.
    Foreign-key enforcement is on (default for SQLite ≥3.6). StaticPool
    so all connections in a test share the same in-memory DB (without
    it, each new connection gets a fresh, empty database).
    """
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    s = session_local()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def analysis_id(session) -> UUID:
    """Insert one AnalysisState row and return its id."""
    a = AnalysisState(
        repository_status="pending",
        paper_status="pending",
    )
    session.add(a)
    session.commit()
    return a.analysis_id


@pytest.fixture
def fake_clone_dir(tmp_path: Path) -> Path:
    """Pre-populate a clone directory so :func:`walk_repo_files`
    has files to operate on. Two Python files + one YAML config +
    one requirements file = a representative tiny repo."""
    root = tmp_path / "data" / "repos" / "acme__demo"
    root.mkdir(parents=True)
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text('"""Demo package."""\n')
    (root / "pkg" / "core.py").write_text(
        'def add(a: int, b: int) -> int:\n    """Add two ints."""\n    return a + b\n'
    )
    (root / "config.yaml").write_text("name: demo\nversion: 0.1\n")
    (root / "requirements.txt").write_text("requests==2.31.0\n# comment\nnumpy>=1.26\n")
    return root


# ============================================================
# run_repo_workflow — happy path
# ============================================================


def test_workflow_happy_path_inserts_row_and_symbols(
    session, analysis_id, fake_clone_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full pipeline: URL parse, fake clone, walk, parse, persist,
    mark done. Asserts every row count + status transitions."""
    # Stub clone_repository so we don't actually shell out.
    fake_result = MagicMock()
    fake_result.clone_path = fake_clone_dir
    fake_result.default_branch = "main"
    monkeypatch.setattr(cloner, "clone_repository", lambda url: fake_result)

    result = run_repo_workflow(
        db=session,
        analysis_id=analysis_id,
        url="https://github.com/acme/demo",
    )

    assert isinstance(result, RepoWorkflowResult)
    assert result.index_status == "done"
    assert result.n_files >= 1
    assert result.n_symbols >= 1  # at least one function in core.py
    assert result.n_configs >= 1  # config.yaml
    assert result.n_dependencies >= 1  # requirements.txt

    # The Repository row should now be 'done' with non-zero counts.
    repo_row = (
        session.query(Repository).filter(Repository.repository_id == result.repository_id).one()
    )
    assert repo_row.index_status == "done"
    assert repo_row.owner == "acme"
    assert repo_row.name == "demo"
    assert repo_row.n_files == result.n_files
    assert repo_row.n_symbols == result.n_symbols
    assert repo_row.indexed_at is not None

    # And the analysis row is marked done with completed_at set.
    state = session.query(AnalysisState).filter(AnalysisState.analysis_id == analysis_id).one()
    assert state.repository_status == "done"
    assert state.completed_at is not None


def test_workflow_records_nonzero_symbols_for_demo_repo(
    session, analysis_id, fake_clone_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `add` function in core.py must show up as a CodeSymbol."""
    fake_result = MagicMock()
    fake_result.clone_path = fake_clone_dir
    fake_result.default_branch = "main"
    monkeypatch.setattr(cloner, "clone_repository", lambda url: fake_result)

    result = run_repo_workflow(
        db=session,
        analysis_id=analysis_id,
        url="https://github.com/acme/demo",
    )

    symbols = (
        session.query(CodeSymbol).filter(CodeSymbol.repository_id == result.repository_id).all()
    )
    names = {s.name for s in symbols}
    assert "add" in names
    # The function lives in pkg/core.py (relative).
    add_sym = next(s for s in symbols if s.name == "add")
    assert add_sym.kind == "function"
    assert add_sym.language == "python"
    assert add_sym.file_path == "pkg/core.py"
    assert add_sym.signature.startswith("(a: int")


def test_workflow_persists_config_row(
    session, analysis_id, fake_clone_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_result = MagicMock()
    fake_result.clone_path = fake_clone_dir
    fake_result.default_branch = "main"
    monkeypatch.setattr(cloner, "clone_repository", lambda url: fake_result)

    result = run_repo_workflow(
        db=session,
        analysis_id=analysis_id,
        url="https://github.com/acme/demo",
    )
    cfgs = (
        session.query(RepositoryConfig)
        .filter(RepositoryConfig.repository_id == result.repository_id)
        .all()
    )
    paths = {c.file_path for c in cfgs}
    assert "config.yaml" in paths


def test_workflow_persists_dependency_rows(
    session, analysis_id, fake_clone_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_result = MagicMock()
    fake_result.clone_path = fake_clone_dir
    fake_result.default_branch = "main"
    monkeypatch.setattr(cloner, "clone_repository", lambda url: fake_result)

    result = run_repo_workflow(
        db=session,
        analysis_id=analysis_id,
        url="https://github.com/acme/demo",
    )
    deps = (
        session.query(RepositoryDependency)
        .filter(RepositoryDependency.repository_id == result.repository_id)
        .all()
    )
    by_name = {d.name: d for d in deps}
    assert "requests" in by_name
    assert "numpy" in by_name
    assert by_name["requests"].version_spec == "==2.31.0"
    assert by_name["numpy"].version_spec == ">=1.26"


# ============================================================
# run_repo_workflow — error paths
# ============================================================


def test_workflow_rejects_invalid_url_before_writing(session, analysis_id) -> None:
    with pytest.raises(RepoInvalidUrlError):
        run_repo_workflow(
            db=session,
            analysis_id=analysis_id,
            url="not-a-github-url",
        )
    # Nothing should have been inserted.
    assert session.query(Repository).count() == 0


def test_workflow_rejects_too_large_before_writing(
    session, analysis_id, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_clone(url):
        raise RepoTooLargeError("repo is 500MB > cap")

    monkeypatch.setattr(cloner, "clone_repository", _fake_clone)

    # parse_github_url succeeds → row gets inserted then the clone
    # raises; the workflow rolls back, marks error, re-raises.
    with pytest.raises(RepoTooLargeError):
        run_repo_workflow(
            db=session,
            analysis_id=analysis_id,
            url="https://github.com/acme/demo",
        )

    repos = session.query(Repository).all()
    assert len(repos) == 1
    assert repos[0].index_status == "error"
    assert "500MB" in (repos[0].error or "")


def test_workflow_records_error_when_clone_raises_clone_error(
    session, analysis_id, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_clone(url):
        raise RepoCloneError("git clone failed: timeout")

    monkeypatch.setattr(cloner, "clone_repository", _fake_clone)

    with pytest.raises(RepoCloneError):
        run_repo_workflow(
            db=session,
            analysis_id=analysis_id,
            url="https://github.com/acme/demo",
        )

    state = session.query(AnalysisState).filter(AnalysisState.analysis_id == analysis_id).one()
    assert state.repository_status == "error"
    err_blob = state.errors
    assert isinstance(err_blob, list) and err_blob
    assert err_blob[0]["where"] == "repo_workflow"
    assert err_blob[0]["type"] == "RepoCloneError"


def test_workflow_retry_replaces_row_left_by_failed_ingest(
    session, analysis_id, fake_clone_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed ingest leaves an ``error`` row; (owner, name) is UNIQUE,
    so the retry must replace it rather than die on the INSERT."""

    def _fail(url):
        raise RepoCloneError("git clone failed: timeout")

    monkeypatch.setattr(cloner, "clone_repository", _fail)
    with pytest.raises(RepoCloneError):
        run_repo_workflow(db=session, analysis_id=analysis_id, url="https://github.com/acme/demo")
    (failed,) = session.query(Repository).all()
    assert failed.index_status == "error"
    failed_id = failed.repository_id

    removed: list[Path] = []
    monkeypatch.setattr(cloner, "remove_clone", removed.append)
    fake_result = MagicMock(clone_path=fake_clone_dir, default_branch="main")
    monkeypatch.setattr(cloner, "clone_repository", lambda url: fake_result)
    result = run_repo_workflow(
        db=session, analysis_id=analysis_id, url="https://github.com/acme/demo"
    )

    assert result.index_status == "done"
    (row,) = session.query(Repository).all()
    assert row.repository_id == result.repository_id != failed_id
    assert removed == [Path(failed.clone_path)]


def test_workflow_refuses_to_reingest_a_done_repository(
    session, analysis_id, fake_clone_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_result = MagicMock(clone_path=fake_clone_dir, default_branch="main")
    monkeypatch.setattr(cloner, "clone_repository", lambda url: fake_result)
    first = run_repo_workflow(
        db=session, analysis_id=analysis_id, url="https://github.com/acme/demo"
    )

    with pytest.raises(RepoAlreadyExistsError) as excinfo:
        run_repo_workflow(db=session, analysis_id=analysis_id, url="https://github.com/acme/demo")

    assert excinfo.value.repository_id == first.repository_id
    assert session.query(Repository).count() == 1
    assert session.query(CodeSymbol).count() == first.n_symbols


# ============================================================
# HTTP layer — POST /repositories
# ============================================================


@pytest.fixture(autouse=True)
def _unset_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Starlette's TestClient picks up HTTP(S)_PROXY and ``all_proxy``.

    The corporate environment sets ``all_proxy=socks://...`` which
    httpx can't parse — clearing the env vars for the duration of
    each test prevents the test client from trying to dial the
    proxy on every request.
    """
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
def mock_workflow() -> Iterator[MagicMock]:
    with patch("claim2code_api.repos.router.run_repo_workflow") as m:
        m.return_value = RepoWorkflowResult(
            repository_id=uuid4(),
            analysis_id=uuid4(),
            index_status="done",
            n_files=42,
            n_symbols=300,
            n_configs=5,
            n_dependencies=10,
        )
        yield m


def _wire_workflow_to_insert_row(m: MagicMock, session_factory) -> None:
    """Replace ``m.return_value`` with a side_effect that also writes
    a Repository row to ``session_factory()`` so the router's
    fetch-back logic has something to read.

    The router does ``db.query(Repository).filter(...).one()`` to
    populate owner / name / url on the response — if no row exists
    the test fails with NoResultFound. Use this for the happy-path
    HTTP test only.
    """
    repository_id = uuid4()

    def _side_effect(
        *,
        db,
        analysis_id: UUID,
        url: str,
        exclude_paths: list[str] | None = None,
    ) -> RepoWorkflowResult:
        row = Repository(
            repository_id=repository_id,
            analysis_id=analysis_id,
            url=url,
            owner="acme",
            name="demo",
            default_branch="main",
            clone_path="data/repos/acme__demo",
            n_files=42,
            n_symbols=300,
            index_status="done",
        )
        db.add(row)
        db.commit()
        return RepoWorkflowResult(
            repository_id=repository_id,
            analysis_id=analysis_id,
            index_status="done",
            n_files=42,
            n_symbols=300,
            n_configs=5,
            n_dependencies=10,
        )

    m.side_effect = _side_effect


@pytest.fixture
def repo_client() -> TestClient:
    """A TestClient with get_db overridden to an in-memory session.

    Because the unit tests for the router want to verify the row
    fetch-back logic, we provide a fully functional in-memory DB
    rather than mocking the SQLAlchemy session. The single seeded
    analysis row's id is captured so dependent tests can pass it
    in the request body.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    # Pre-seed an analysis so POST /repositories finds it.
    seed = session_local()
    a = AnalysisState(
        repository_status="pending",
        paper_status="pending",
    )
    seed.add(a)
    seed.commit()
    seeded_id = a.analysis_id
    seed.close()

    def _override_get_db():
        s = session_local()
        try:
            yield s
        finally:
            s.close()

    app = create_app()
    app.dependency_overrides[get_db] = _override_get_db
    client_obj = TestClient(app)
    # Attach the seeded id + session factory so dependent tests can
    # either reuse the seeded analysis or wire the workflow mock to
    # insert a Repository row.
    client_obj.seed_analysis_id = seeded_id  # type: ignore[attr-defined]
    client_obj.session_factory = session_local  # type: ignore[attr-defined]
    return client_obj


def test_post_repositories_route_is_registered(repo_client: TestClient) -> None:
    schema = repo_client.get("/openapi.json").json()
    assert "/repositories" in schema["paths"]
    assert "post" in schema["paths"]["/repositories"]


def test_post_repositories_returns_summary(
    repo_client: TestClient, mock_workflow: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The router fetches the Repository row after workflow returns;
    # we wire the mock to insert a row so the fetch-back succeeds.
    _wire_workflow_to_insert_row(
        mock_workflow,
        repo_client.session_factory,  # type: ignore[attr-defined]
    )
    seeded = repo_client.seed_analysis_id  # type: ignore[attr-defined]
    response = repo_client.post(
        "/repositories",
        json={
            "url": "https://github.com/acme/demo",
            "analysis_id": str(seeded),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) >= {
        "repository_id",
        "analysis_id",
        "url",
        "owner",
        "name",
        "n_files",
        "n_symbols",
        "n_configs",
        "n_dependencies",
        "index_status",
    }
    assert body["index_status"] == "done"
    assert body["owner"] == "acme"
    assert body["name"] == "demo"
    assert body["n_files"] == 42
    assert body["n_symbols"] == 300
    assert body["n_configs"] == 5
    assert body["n_dependencies"] == 10


def test_post_repositories_returns_404_when_analysis_missing(repo_client: TestClient) -> None:
    response = repo_client.post(
        "/repositories",
        json={
            "url": "https://github.com/acme/demo",
            "analysis_id": str(uuid4()),  # not in our seeded DB
        },
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_post_repositories_maps_invalid_url_to_422(
    repo_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the workflow raises RepoInvalidUrlError, the router
    returns 422. We patch the router's reference (not the source
    module) since the router does ``from workflow import ...``."""
    import claim2code_api.repos.router as router_mod

    monkeypatch.setattr(
        router_mod,
        "run_repo_workflow",
        MagicMock(side_effect=RepoInvalidUrlError("not a github url")),
    )
    seeded = repo_client.seed_analysis_id  # type: ignore[attr-defined]
    response = repo_client.post(
        "/repositories",
        json={
            "url": "https://github.com/acme/demo",
            "analysis_id": str(seeded),
        },
    )
    assert response.status_code == 422
    assert "not a github url" in response.json()["detail"]


def test_post_repositories_maps_too_large_to_413(
    repo_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import claim2code_api.repos.router as router_mod

    monkeypatch.setattr(
        router_mod,
        "run_repo_workflow",
        MagicMock(side_effect=RepoTooLargeError("repo is 500MB > cap")),
    )
    seeded = repo_client.seed_analysis_id  # type: ignore[attr-defined]
    response = repo_client.post(
        "/repositories",
        json={
            "url": "https://github.com/acme/demo",
            "analysis_id": str(seeded),
        },
    )
    assert response.status_code == 413
    assert "500MB" in response.json()["detail"]


def test_post_repositories_maps_clone_error_to_500(
    repo_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import claim2code_api.repos.router as router_mod

    monkeypatch.setattr(
        router_mod,
        "run_repo_workflow",
        MagicMock(side_effect=RepoCloneError("git clone failed: timeout")),
    )
    seeded = repo_client.seed_analysis_id  # type: ignore[attr-defined]
    response = repo_client.post(
        "/repositories",
        json={
            "url": "https://github.com/acme/demo",
            "analysis_id": str(seeded),
        },
    )
    assert response.status_code == 500
    assert "git clone failed" in response.json()["detail"]


def test_post_repositories_maps_already_exists_to_409(
    repo_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import claim2code_api.repos.router as router_mod

    existing = uuid4()
    monkeypatch.setattr(
        router_mod,
        "run_repo_workflow",
        MagicMock(side_effect=RepoAlreadyExistsError(existing, "done")),
    )
    seeded = repo_client.seed_analysis_id  # type: ignore[attr-defined]
    response = repo_client.post(
        "/repositories",
        json={"url": "https://github.com/acme/demo", "analysis_id": str(seeded)},
    )
    assert response.status_code == 409
    assert str(existing) in response.json()["detail"]


def test_post_repositories_rejects_missing_url(repo_client: TestClient) -> None:
    """Pydantic validation rejects the missing required field with 422."""
    response = repo_client.post(
        "/repositories",
        json={"analysis_id": str(uuid4())},
    )
    assert response.status_code == 422


def test_post_repositories_rejects_missing_analysis_id(repo_client: TestClient) -> None:
    response = repo_client.post(
        "/repositories",
        json={"url": "https://github.com/acme/demo"},
    )
    assert response.status_code == 422


# ============================================================
# Settings sanity
# ============================================================


def test_repo_parse_ignore_dirs_setting_present() -> None:
    s = get_settings()
    assert "node_modules" in s.repo_parse_ignore_dirs
    assert ".venv" in s.repo_parse_ignore_dirs
    # ``.git`` is added unconditionally in ``walk_repo_files`` — not in
    # the settings list — so we don't assert it here.


def test_repo_parse_extensions_setting_present() -> None:
    s = get_settings()
    assert ".py" in s.repo_parse_extensions
    assert ".js" in s.repo_parse_extensions
    assert ".ts" in s.repo_parse_extensions


# (datetime / UTC / UUID are all referenced above in the test bodies —
# no-touch lines needed.)
