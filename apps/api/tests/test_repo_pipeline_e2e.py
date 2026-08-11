"""End-to-end pipeline test for :mod:`claim2code_api.repos.workflow`.

Unlike ``test_repo_workflow.py`` which mocks the cloner, this test
lets the real walk + parse + persist pipeline run against a
hand-built "repo" directory on disk. The git clone step is still
stubbed (no network in unit tests), but every other piece —

  * ``walk_repo_files`` (storage)
  * ``classify_file`` (storage)
  * ``parse_file`` (tree-sitter)
  * ``parse_config_file`` (yaml/json/toml/ini)
  * ``parse_lockfile`` (requirements/poetry/npm)

— runs against real synthetic files. This catches regressions
where one module's output shape no longer matches the next
module's input.

Runs as a unit test (no ``@pytest.mark.integration``) so CI catches
it on every PR.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from claim2code_api.db.models import (
    AnalysisState,
    Base,
    CodeSymbol,
    Repository,
    RepositoryConfig,
    RepositoryDependency,
)
from claim2code_api.repos import cloner
from claim2code_api.repos.workflow import run_repo_workflow


@pytest.fixture
def engine():
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
def synthetic_repo(tmp_path: Path) -> Path:
    """A multi-language synthetic repo exercising every extractor.

    Layout::

        src/
            add.py          — Python function
            widget.ts       — TypeScript class + method
            greet.go        — Go function
        config/
            app.yaml        — YAML config
            settings.json   — JSON config
        lockfile/
            requirements.txt
            package-lock.json
        README.md          — markdown (no symbols expected)
        ignore_me.pyc      — .pyc, should be skipped
        node_modules/
            ignored.py     — must be skipped
    """
    root = tmp_path / "synthetic"
    (root / "src").mkdir(parents=True)
    (root / "config").mkdir(parents=True)
    (root / "lockfile").mkdir(parents=True)

    (root / "src" / "add.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        '    """Add two ints."""\n'
        "    return a + b\n"
    )
    (root / "src" / "widget.ts").write_text(
        "export class Widget {\n"
        "    constructor(public name: string) {}\n"
        "    greet(): string {\n"
        '        return `Hello, ${this.name}!`;\n'
        "    }\n"
        "}\n"
    )
    (root / "src" / "greet.go").write_text(
        "package main\n"
        "\n"
        'import "fmt"\n'
        "\n"
        "func Greet(name string) string {\n"
        '    return fmt.Sprintf("Hello, %s!", name)\n'
        "}\n"
    )

    (root / "config" / "app.yaml").write_text(
        "name: synthetic\nversion: 0.1.0\ndependencies:\n  - foo\n  - bar\n"
    )
    (root / "config" / "settings.json").write_text(
        '{"debug": false, "max_items": 100}\n'
    )

    (root / "lockfile" / "requirements.txt").write_text(
        "requests==2.31.0\n# comment\nnumpy>=1.26\npandas<2.0\n"
    )
    (root / "lockfile" / "package-lock.json").write_text(
        '{"name":"synthetic","version":"1.0.0","lockfileVersion":3,'
        '"packages":{"node_modules/lodash":{"version":"4.17.21"},'
        '"node_modules/react":{"version":"18.2.0"}}}\n'
    )

    (root / "README.md").write_text("# Synthetic repo\n\nFor testing.\n")

    # Ignored files
    (root / "ignore_me.pyc").write_bytes(b"\x00\x00\x00")
    (root / "node_modules").mkdir(parents=True)
    (root / "node_modules" / "ignored.py").write_text("x = 1\n")

    return root


def test_pipeline_walks_parses_persists_every_kind(
    session, synthetic_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: walk + classify + dispatch + persist against a
    multi-language synthetic repo. Every extractor must be
    exercised at least once."""
    analysis = AnalysisState(
        repository_status="pending",
        paper_status="pending",
    )
    session.add(analysis)
    session.commit()

    # Stub the cloner so we never shell out to git.
    fake = MagicMock()
    fake.clone_path = synthetic_repo
    fake.default_branch = "main"
    monkeypatch.setattr(cloner, "clone_repository", lambda url: fake)

    result = run_repo_workflow(
        db=session,
        analysis_id=analysis.analysis_id,
        url="https://github.com/synthetic/repo",
    )

    assert result.index_status == "done"

    # Code symbols: at least one per language we ship.
    symbols = (
        session.query(CodeSymbol)
        .filter(CodeSymbol.repository_id == result.repository_id)
        .all()
    )
    by_lang: dict[str, set[str]] = {}
    for s in symbols:
        by_lang.setdefault(s.language, set()).add(s.name)
    assert "add" in by_lang.get("python", set())
    assert "Widget" in by_lang.get("typescript", set())
    assert "Greet" in by_lang.get("go", set())

    # Configs: both .yaml and .json are persisted.
    configs = (
        session.query(RepositoryConfig)
        .filter(RepositoryConfig.repository_id == result.repository_id)
        .all()
    )
    paths = {c.file_path for c in configs}
    assert "config/app.yaml" in paths
    assert "config/settings.json" in paths

    # Lockfiles: requirements + package-lock both contribute rows.
    deps = (
        session.query(RepositoryDependency)
        .filter(RepositoryDependency.repository_id == result.repository_id)
        .all()
    )
    by_name = {d.name: d for d in deps}
    # requirements.txt contributes pypi rows
    assert "requests" in by_name
    assert by_name["requests"].ecosystem == "pypi"
    # package-lock.json contributes npm rows
    assert "lodash" in by_name
    assert by_name["lodash"].ecosystem == "npm"

    # The ignored files must NOT show up.
    ignored = [d for d in deps if d.name == "ignored"]
    assert ignored == []
    sym_pyc = [s for s in symbols if s.file_path.endswith(".pyc")]
    assert sym_pyc == []


def test_pipeline_handles_empty_repo_without_crashing(
    session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo with zero parseable files completes with
    ``index_status='done'`` and all counts = 0."""
    empty = tmp_path / "empty"
    empty.mkdir()

    analysis = AnalysisState(
        repository_status="pending",
        paper_status="pending",
    )
    session.add(analysis)
    session.commit()

    fake = MagicMock()
    fake.clone_path = empty
    fake.default_branch = "main"
    monkeypatch.setattr(cloner, "clone_repository", lambda url: fake)

    result = run_repo_workflow(
        db=session,
        analysis_id=analysis.analysis_id,
        url="https://github.com/empty/nope",
    )

    assert result.index_status == "done"
    assert result.n_files == 0
    assert result.n_symbols == 0
    assert result.n_configs == 0
    assert result.n_dependencies == 0

    repo = (
        session.query(Repository)
        .filter(Repository.repository_id == result.repository_id)
        .one()
    )
    assert repo.index_status == "done"
    assert repo.n_files == 0