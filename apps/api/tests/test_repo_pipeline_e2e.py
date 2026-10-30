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


@pytest.fixture(autouse=True)
def stub_embedding_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """This suite covers parsing/persistence; embeddings have their own tests."""
    monkeypatch.setattr(
        "claim2code_api.repos.workflow.embed_and_persist_symbols", MagicMock(),
    )


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

# ============================================================
# Method -> class linkage + verifier class context
# ============================================================


def test_methods_link_to_enclosing_class_and_feed_class_context(
    session, tmp_path: Path,
) -> None:
    """Methods get ``parent_symbol_id`` by line containment, which the
    verifier uses to attach the class's setup methods to a candidate."""
    import uuid

    from claim2code_api.repos.storage import WalkedFile
    from claim2code_api.repos.workflow import _extract_symbols
    from claim2code_api.verification.workflow import _class_spans

    src = tmp_path / "layers.py"
    src.write_text(
        "class Linear:\n"
        "    def __init__(self, r):\n"
        "        self.scaling = 1 / r\n"
        "\n"
        "    def reset_parameters(self):\n"
        "        init_zeros(self.B)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return x * self.scaling\n"
        "\n"
        "\n"
        "class Other:\n"
        "    def forward(self, x):\n"
        "        return x\n"
    )
    repo_id = uuid.uuid4()
    wf = WalkedFile(absolute_path=src, relative_path="layers.py", size_bytes=src.stat().st_size)
    _extract_symbols(session, repo_id, wf, {})
    session.flush()

    rows = {(s.name, s.start_line): s for s in session.query(CodeSymbol).all()}
    linear, other = rows[("Linear", 1)], rows[("Other", 12)]
    assert rows[("__init__", 2)].parent_symbol_id == linear.symbol_id
    assert rows[("forward", 8)].parent_symbol_id == linear.symbol_id
    assert rows[("forward", 13)].parent_symbol_id == other.symbol_id

    spans = _class_spans(session, rows[("forward", 8)], tmp_path)
    located = {(s.name, s.file_path, s.start_line, s.end_line): s.source for s in spans}
    assert "self.scaling = 1 / r" in located[("__init__", "layers.py", 2, 3)]
    assert "init_zeros(self.B)" in located[("reset_parameters", "layers.py", 5, 6)]
    assert all("return x * self.scaling" not in s.source for s in spans)  # not the candidate itself
    assert _class_spans(session, rows[("forward", 13)], tmp_path) == []


def test_callee_context_resolves_nearest_definition(
    session, tmp_path: Path,
) -> None:
    """The verifier attaches the source of repo functions a candidate
    calls, picking the definition nearest the caller's file and
    skipping callees that are already candidates."""
    import uuid

    from claim2code_api.repos.storage import WalkedFile
    from claim2code_api.repos.workflow import _extract_symbols
    from claim2code_api.verification.workflow import _callee_spans

    files = {
        "alf/run.py": (
            "def run(env):\n"
            "    step = 0\n"
            "    while step < 49:\n"
            "        if exhausted(env):\n"
            "            return helper(step)\n"
            "        step += 1\n"
        ),
        "alf/hist.py": "def exhausted(env):\n    return env.repeats > 1\n",
        "web/hist.py": "def exhausted(env):\n    return False\n",
        "alf/util.py": "def helper(x):\n    return x\n",
    }
    repo_id = uuid.uuid4()
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        wf = WalkedFile(absolute_path=path, relative_path=rel, size_bytes=path.stat().st_size)
        _extract_symbols(session, repo_id, wf, {})
    session.flush()

    rows = {(s.file_path, s.name): s for s in session.query(CodeSymbol).all()}
    run = rows[("alf/run.py", "run")]
    helper = rows[("alf/util.py", "helper")]

    spans = _callee_spans(session, run, files["alf/run.py"], tmp_path, exclude={helper.symbol_id})
    # Not the farther definition in web/hist.py, nor helper (already a candidate).
    assert [(s.name, s.file_path, s.start_line, s.end_line) for s in spans] == [
        ("exhausted", "alf/hist.py", 1, 2),
    ]
    assert "env.repeats > 1" in spans[0].source

    assert _callee_spans(session, helper, files["alf/util.py"], tmp_path, exclude=set()) == []


def test_callee_context_follows_registry_dispatch(
    session, tmp_path: Path,
) -> None:
    """A call through a module-level registry dict resolves to the
    function the registry names, ahead of the direct callees."""
    import uuid

    from claim2code_api.repos.storage import WalkedFile
    from claim2code_api.repos.workflow import _extract_symbols
    from claim2code_api.verification.workflow import _callee_spans

    files = {
        "runner.py": (
            "def train(config, x):\n"
            "    data = load(x)\n"
            "    return loss_registry[config.kind](data)\n"
        ),
        "losses.py": (
            "def simple_loss(x):\n"
            "    return (x - 1).square().sum()\n"
            "\n"
            "\n"
            "loss_registry = {\n"
            "    'simple': simple_loss,\n"
            "}\n"
        ),
        "data.py": "def load(x):\n    return x\n",
    }
    repo_id = uuid.uuid4()
    for rel, text in files.items():
        path = tmp_path / rel
        path.write_text(text)
        wf = WalkedFile(absolute_path=path, relative_path=rel, size_bytes=path.stat().st_size)
        _extract_symbols(session, repo_id, wf, {})
    session.flush()

    train = next(s for s in session.query(CodeSymbol).all() if s.name == "train")
    spans = _callee_spans(session, train, files["runner.py"], tmp_path, exclude=set())
    assert [(s.name, s.file_path, s.start_line) for s in spans][:2] == [
        ("simple_loss", "losses.py", 1), ("load", "data.py", 1),
    ]
    assert "square().sum()" in spans[0].source
    registry = next(s for s in spans if s.name == "loss_registry")
    assert registry.file_path == "losses.py"
    assert registry.start_line == 5
    assert "'simple': simple_loss" in registry.source


def test_usage_context_shows_where_an_option_is_read(
    session, tmp_path: Path,
) -> None:
    """A command-line option's candidate carries the lines that read it,
    inside their enclosing function, from its file before its siblings."""
    import uuid

    from claim2code_api.repos.storage import WalkedFile
    from claim2code_api.repos.workflow import _extract_symbols
    from claim2code_api.verification.workflow import _usage_spans

    files = {
        "main.py": (
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--batch-size', default=64, type=int)\n"
            "parser.add_argument('--lr', default=5e-4, type=float)\n"
            "\n"
            "\n"
            "def main(args):\n"
            "    model = build()\n"
            "    data = load()\n"
            "    args.lr = args.lr * args.batch_size / 512.0\n"
            "    other.lr_drop = 3\n"
            "    return train(model, data, args)\n"
        ),
        "engine.py": "def step(args):\n    return args.lr\n",
        "sub/engine.py": "def step(args):\n    return args.lr\n",
    }
    repo_id = uuid.uuid4()
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        wf = WalkedFile(absolute_path=path, relative_path=rel, size_bytes=path.stat().st_size)
        _extract_symbols(session, repo_id, wf, {})
    session.flush()

    rows = {(s.file_path, s.name): s for s in session.query(CodeSymbol).all()}
    lr = rows[("main.py", "--lr")]
    spans = _usage_spans(session, lr, tmp_path)
    assert [(s.name, s.file_path, s.start_line, s.end_line) for s in spans] == [
        ("main", "main.py", 8, 12),
    ]
    assert spans[0].symbol_id == rows[("main.py", "main")].symbol_id
    assert "args.lr * args.batch_size / 512.0" in spans[0].source

    # Not read in its own file: the sibling in the same directory, not sub/.
    files["main.py"] = files["main.py"].replace("args.lr = args.lr * ", "scale = ")
    (tmp_path / "main.py").write_text(files["main.py"])
    assert [(s.file_path, s.name) for s in _usage_spans(session, lr, tmp_path)] == [("engine.py", "step")]
