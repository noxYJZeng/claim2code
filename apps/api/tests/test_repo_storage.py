"""Unit tests for :mod:`claim2code_api.repos.storage`."""

from __future__ import annotations

from pathlib import Path

from claim2code_api.repos.storage import (
    classify_file,
    walk_repo_files,
)


def test_classify_source(tmp_path: Path) -> None:
    for ext in (".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs"):
        c = classify_file(tmp_path / f"f{ext}")
        assert c.mode == "source", ext


def test_classify_config(tmp_path: Path) -> None:
    for ext, fmt in [
        (".yaml", "yaml"),
        (".yml", "yaml"),
        (".json", "json"),
        (".toml", "toml"),
        (".ini", "ini"),
        (".cfg", "ini"),
    ]:
        c = classify_file(tmp_path / f"f{ext}")
        assert c.mode == "config", ext
        assert c.format == fmt, ext


def test_classify_lockfile_pypi(tmp_path: Path) -> None:
    for name in ("requirements.txt", "poetry.lock", "Pipfile.lock"):
        c = classify_file(tmp_path / name)
        assert c.mode == "lockfile", name
        assert c.format == "pypi", name


def test_classify_lockfile_npm(tmp_path: Path) -> None:
    for name in ("package-lock.json", "yarn.lock", "pnpm-lock.yaml"):
        c = classify_file(tmp_path / name)
        assert c.mode == "lockfile", name
        assert c.format == "npm", name


def test_classify_readme(tmp_path: Path) -> None:
    for name in ("README.md", "README.rst", "README.txt", "readme"):
        c = classify_file(tmp_path / name)
        assert c.mode == "readme", name


def test_classify_other(tmp_path: Path) -> None:
    assert classify_file(tmp_path / "image.png").mode == "other"


# ============================================================
# walk_repo_files
# ============================================================


def _make_clone(tmp_path: Path, layout: dict[str, str]) -> Path:
    """Helper: create a fake clone directory with given relative paths.

    ``layout`` maps repo-relative POSIX paths to file contents.
    """
    clone = tmp_path / "clone"
    for rel, content in layout.items():
        p = clone / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return clone


def test_walk_returns_all_parseable_files(tmp_path: Path) -> None:
    clone = _make_clone(
        tmp_path,
        {
            "src/main.py": "x = 1",
            "src/util.py": "y = 2",
            "README.md": "# hi",
            "config.yaml": "a: 1",
        },
    )
    files = walk_repo_files(clone)
    rels = {f.relative_path for f in files}
    assert rels == {"src/main.py", "src/util.py", "README.md", "config.yaml"}


def test_walk_skips_ignored_dirs(tmp_path: Path) -> None:
    clone = _make_clone(
        tmp_path,
        {
            "src/main.py": "x",
            "node_modules/lib/index.js": "var x",
            ".git/HEAD": "ref: refs/heads/main",
            "build/output.py": "y",
            "dist/bundle.js": "z",
        },
    )
    files = walk_repo_files(clone, ignore_dirs=["node_modules", "dist", "build"])
    rels = {f.relative_path for f in files}
    assert rels == {"src/main.py"}


def test_walk_skips_exclude_paths(tmp_path: Path) -> None:
    """A directory path excludes its subtree; globs match too."""
    clone = _make_clone(
        tmp_path,
        {
            "loralib/layers.py": "x",
            "examples/NLU/src/transformers/models/bert.py": "y",
            "examples/NLU/run_glue.py": "z",
            "examples/NLG/src/gpt2_beam.py": "w",
        },
    )
    files = walk_repo_files(
        clone,
        exclude_paths=["examples/NLU/src/transformers/", "examples/*/src/gpt2_*.py"],
    )
    rels = {f.relative_path for f in files}
    assert rels == {"loralib/layers.py", "examples/NLU/run_glue.py"}


def test_walk_skips_git_even_if_not_in_ignore(tmp_path: Path) -> None:
    """``.git`` is hard-coded to skip regardless of ignore_dirs."""
    clone = _make_clone(
        tmp_path,
        {"src/main.py": "x", ".git/HEAD": "ref: refs/heads/main"},
    )
    files = walk_repo_files(clone, ignore_dirs=[])
    rels = {f.relative_path for f in files}
    assert ".git/HEAD" not in rels
    assert "src/main.py" in rels


def test_walk_skips_oversized_files(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "small.py").write_text("x")
    (clone / "big.py").write_text("y" * (2 * 1024 * 1024))  # 2 MB
    files = walk_repo_files(clone, max_file_kb=512)
    rels = {f.relative_path for f in files}
    assert "small.py" in rels
    assert "big.py" not in rels


def test_walk_caps_total_count(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    clone.mkdir()
    for i in range(10):
        (clone / f"f{i}.py").write_text("x")
    files = walk_repo_files(clone, max_files=3)
    assert len(files) == 3


def test_walk_missing_dir_returns_empty(tmp_path: Path) -> None:
    files = walk_repo_files(tmp_path / "nope")
    assert files == []


def test_walk_size_bytes_recorded(tmp_path: Path) -> None:
    clone = _make_clone(tmp_path, {"a.py": "hello"})
    files = walk_repo_files(clone)
    assert files[0].size_bytes == len("hello")
