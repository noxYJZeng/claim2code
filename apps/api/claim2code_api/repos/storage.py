"""Disk helpers for the repository layer.

Two responsibilities:

  1. **Walk a clone directory** — :func:`walk_repo_files` returns every
     candidate source file under a clone, applying the
     :attr:`Settings.repo_parse_ignore_dirs` filter (skipping
     ``node_modules``, ``.git``, ``dist``, ``build``, …) and the
     per-file ``max_file_kb`` guard.

  2. **Classify files by extension** — :func:`classify_file` returns
     the parser mode (``source`` / ``config`` / ``lockfile`` / ``readme``
     / ``other``) so the workflow layer can dispatch to the right
     extractor.

We deliberately do not store anything in the DB from this module —
that lives in :mod:`claim2code_api.repos.workflow`. This module only
inspects disk.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

from claim2code_api.app.config import get_settings
from claim2code_api.repos.cloner import data_repos_root

# File extensions that should be parsed for code symbols. Matches the
# v1 grammar list (Python, JS, TS, Java, Go, Rust) plus shell launch
# scripts — C / C++ / PHP / Ruby / Kotlin / Swift / Scala are deferred
# (see phase-3-design.md §3).
SOURCE_EXTENSIONS: set[str] = {
    ".py",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".sh",
    ".bash",
}

# Config file extensions. Each maps to a single format.
CONFIG_EXTENSIONS: dict[str, str] = {
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",  # Setuptools-style configs are INI-shaped
    ".conf": "ini",
}

# Lockfile / dependency-manifest basenames. Order matters for the
# glob match (longer suffixes first).
LOCKFILE_BASENAMES: set[str] = {
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "requirements-prod.txt",
    "poetry.lock",
    "Pipfile.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
}

# README basenames (case-insensitive).
README_BASENAMES: set[str] = {
    "readme.md",
    "readme.rst",
    "readme.txt",
    "readme",
}


@dataclass(frozen=True)
class FileClassification:
    """How the workflow layer should treat a file.

    ``mode`` is one of: ``source``, ``config``, ``lockfile``, ``readme``,
    ``other``. The ``format`` field carries the format tag (``yaml``,
    ``json``, …) for configs and ``pypi`` / ``npm`` for lockfiles.
    """

    mode: str
    format: str = ""
    basename: str = ""


def classify_file(path: Path) -> FileClassification:
    """Return the workflow mode for a single file path.

    Order matters: lockfile basenames (``package-lock.json`` etc.)
    must be checked before the extension-based config match, because
    ``package-lock.json`` also has a ``.json`` extension and would
    otherwise be classified as a config.
    """
    ext = path.suffix.lower()
    name = path.name  # case-sensitive for lockfiles (Pipfile.lock)

    # ---- Lockfile basenames (before extension match) ----
    if name in LOCKFILE_BASENAMES:
        if (
            name.startswith("package-lock")
            or name.startswith("yarn.lock")
            or name.startswith("pnpm-lock")
        ):
            return FileClassification(mode="lockfile", format="npm", basename=name)
        return FileClassification(mode="lockfile", format="pypi", basename=name)

    if ext in SOURCE_EXTENSIONS:
        return FileClassification(mode="source", format=ext.lstrip("."))
    if ext in CONFIG_EXTENSIONS:
        return FileClassification(mode="config", format=CONFIG_EXTENSIONS[ext])
    if name.lower() in README_BASENAMES:
        return FileClassification(mode="readme", basename=name)
    return FileClassification(mode="other")


@dataclass(frozen=True)
class WalkedFile:
    """One file yielded by :func:`walk_repo_files`."""

    absolute_path: Path
    relative_path: str  # repo-relative POSIX path
    size_bytes: int


def walk_repo_files(
    clone_path: Path,
    *,
    ignore_dirs: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    max_file_kb: int | None = None,
    max_files: int | None = None,
) -> list[WalkedFile]:
    """Yield one :class:`WalkedFile` per parseable file.

    Args:
        clone_path: Absolute path to a clone directory (typically
            ``data/repos/<owner>__<repo>/``).
        ignore_dirs: Directory basenames to skip (default:
            :attr:`Settings.repo_parse_ignore_dirs`). A leading
            ``.git`` is always skipped.
        exclude_paths: Repo-relative paths or ``fnmatch`` globs to skip
            (e.g. ``examples/NLU/src/transformers`` for a vendored
            copy of a library). A pattern naming a directory also
            excludes everything under it.
        max_file_kb: Per-file size cap in KiB (default:
            :attr:`Settings.max_file_kb`).
        max_files: Cap on the total number of files returned (default:
            :attr:`Settings.max_repo_files`). When the cap is hit the
            walk stops early.

    Returns:
        A list of :class:`WalkedFile`. Empty if the clone directory
        does not exist or contains no parseable files.
    """
    settings = get_settings()
    ignore = set(ignore_dirs if ignore_dirs is not None else settings.repo_parse_ignore_dirs)
    # Always skip .git regardless of config — touching .git internals
    # would be wasteful and risky.
    ignore.add(".git")
    cap_kb = max_file_kb if max_file_kb is not None else settings.max_file_kb
    cap_files = max_files if max_files is not None else settings.max_repo_files
    cap_bytes = cap_kb * 1024
    exclude = [p.strip("/") for p in (exclude_paths or []) if p.strip("/")]

    if not clone_path.exists():
        return []

    out: list[WalkedFile] = []
    for path in clone_path.rglob("*"):
        if not path.is_file():
            continue
        # Skip if any parent directory is in the ignore list. The
        # parent check matters because ``node_modules`` appears deep
        # in many repos.
        if any(part in ignore for part in path.parts):
            continue
        rel = path.relative_to(clone_path).as_posix()
        if exclude and _is_excluded(rel, exclude):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > cap_bytes:
            continue
        out.append(WalkedFile(absolute_path=path, relative_path=rel, size_bytes=size))
        if len(out) >= cap_files:
            break
    return out


def _is_excluded(rel_path: str, patterns: list[str]) -> bool:
    """True if ``rel_path`` matches a pattern or lies under one."""
    for pat in patterns:
        if fnmatch.fnmatch(rel_path, pat) or fnmatch.fnmatch(rel_path, pat + "/*"):
            return True
    return False


__all__ = [
    "CONFIG_EXTENSIONS",
    "FileClassification",
    "LOCKFILE_BASENAMES",
    "README_BASENAMES",
    "SOURCE_EXTENSIONS",
    "WalkedFile",
    "classify_file",
    "data_repos_root",
    "walk_repo_files",
]
