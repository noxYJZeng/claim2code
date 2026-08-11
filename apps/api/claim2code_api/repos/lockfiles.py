"""Lockfile / dependency-manifest extractors.

Public entry point: :func:`parse_lockfile`. Returns a list of
:class:`RepositoryDependency` rows — one per declared package — with
the ecosystem, name, version spec, and source filename.

Supported files (basename match, case-sensitive):

  * ``requirements.txt`` and the ``-dev`` / ``-test`` / ``-prod``
    variants → ecosystem ``pypi``.
  * ``poetry.lock`` → JSON; ecosystem ``pypi``.
  * ``Pipfile.lock`` → JSON; ecosystem ``pypi``.
  * ``package-lock.json`` → JSON; ecosystem ``npm``.
  * ``yarn.lock`` → text; ecosystem ``npm`` (basic YAML-style parse).
  * ``pnpm-lock.yaml`` → YAML; ecosystem ``npm``.

The version spec is kept verbatim (``==2.1.0``, ``^1.0.0``, etc.) —
normalisation for cross-checking against paper text lands in Phase 6.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from claim2code_api.repos.schemas import DependencyEcosystem, RepositoryDependency

# =========================================================
# Dispatcher
# =========================================================


def parse_lockfile(path: Path, file_path: str) -> list[RepositoryDependency]:
    """Parse a single lockfile / requirements file.

    Args:
        path: Absolute path on disk.
        file_path: Repo-relative POSIX path used as
            :attr:`RepositoryDependency.source_file`.

    Returns:
        A list of :class:`RepositoryDependency` rows. Empty if the
        file is unrecognised or empty. Bad lines are skipped without
        aborting the parse.
    """
    name = path.name
    if name.startswith("requirements"):
        return _parse_requirements_txt(path, file_path)
    if name == "poetry.lock":
        return _parse_poetry_lock(path, file_path)
    if name == "Pipfile.lock":
        return _parse_pipfile_lock(path, file_path)
    if name == "package-lock.json":
        return _parse_package_lock_json(path, file_path)
    if name == "yarn.lock":
        return _parse_yarn_lock(path, file_path)
    if name == "pnpm-lock.yaml":
        return _parse_pnpm_lock(path, file_path)
    return []


# =========================================================
# requirements*.txt
# =========================================================


# A line like ``requests[security]>=2.28 ; python_version >= "3.10"``
# should yield ``requests[security]>=2.28`` plus an env marker. We
# only need the package + version spec; the marker is preserved on
# the package string for now.
_REQUIREMENT_LINE_RE = re.compile(
    r"^\s*(?P<pkg>[A-Za-z0-9._\[\]'+-]+)\s*"
    r"(?P<op>==|>=|<=|~=|!=|>|<|\^)"
    r"\s*"
    r"(?P<ver>[A-Za-z0-9.*+!\-_,]+)"
    r"\s*"
    r"(?:\;.*)?$"
)


def _parse_requirements_txt(path: Path, file_path: str) -> list[RepositoryDependency]:
    rows: list[RepositoryDependency] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        match = _REQUIREMENT_LINE_RE.match(line)
        if match is None:
            continue
        rows.append(
            RepositoryDependency(
                ecosystem=DependencyEcosystem.PYPI,
                name=match.group("pkg"),
                version_spec=f"{match.group('op')}{match.group('ver')}",
                source_file=file_path,
            )
        )
    return rows


# =========================================================
# poetry.lock
# =========================================================


def _parse_poetry_lock(path: Path, file_path: str) -> list[RepositoryDependency]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return []
    rows: list[RepositoryDependency] = []
    for pkg in data.get("package", []) or []:
        name = pkg.get("name")
        version = pkg.get("version", "")
        if not name:
            continue
        rows.append(
            RepositoryDependency(
                ecosystem=DependencyEcosystem.PYPI,
                name=str(name),
                version_spec=f"=={version}" if version else "",
                source_file=file_path,
            )
        )
    return rows


# =========================================================
# Pipfile.lock
# =========================================================


def _parse_pipfile_lock(path: Path, file_path: str) -> list[RepositoryDependency]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return []
    rows: list[RepositoryDependency] = []
    for section in ("default", "develop"):
        block = data.get(section) or {}
        for name, meta in block.items():
            if not isinstance(meta, dict):
                continue
            version = meta.get("version", "")
            rows.append(
                RepositoryDependency(
                    ecosystem=DependencyEcosystem.PYPI,
                    name=str(name),
                    version_spec=str(version) if version else "",
                    source_file=file_path,
                )
            )
    return rows


# =========================================================
# package-lock.json
# =========================================================


def _parse_package_lock_json(path: Path, file_path: str) -> list[RepositoryDependency]:
    """Parse a ``package-lock.json``.

    Supports both lockfileVersion 1 (``dependencies`` map) and 2/3
    (``packages`` map with ``node_modules/<name>`` keys).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return []
    rows: list[RepositoryDependency] = []
    if isinstance(data.get("packages"), dict):
        for key, meta in data["packages"].items():
            if not isinstance(key, str) or not key.startswith("node_modules/"):
                continue
            name = key[len("node_modules/") :]
            if not name:
                continue
            version = (meta or {}).get("version", "") if isinstance(meta, dict) else ""
            rows.append(
                RepositoryDependency(
                    ecosystem=DependencyEcosystem.NPM,
                    name=name,
                    version_spec=str(version) if version else "",
                    source_file=file_path,
                )
            )
    elif isinstance(data.get("dependencies"), dict):
        for name, meta in data["dependencies"].items():
            version = (meta or {}).get("version", "") if isinstance(meta, dict) else ""
            rows.append(
                RepositoryDependency(
                    ecosystem=DependencyEcosystem.NPM,
                    name=str(name),
                    version_spec=str(version) if version else "",
                    source_file=file_path,
                )
            )
    return rows


# =========================================================
# yarn.lock (text format)
# =========================================================


def _parse_yarn_lock(path: Path, file_path: str) -> list[RepositoryDependency]:
    """Best-effort parse of yarn.lock.

    Yarn 1.x is a custom text format; we only extract ``name@^version``
    lines because full resolution is not needed for Phase 3 (we just
    want a flat dep list).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows: list[RepositoryDependency] = []
    for line in text.splitlines():
        if not line or line.startswith("#") or line.startswith(" "):
            continue
        # Lines like ``"package@^1.0.0":`` or ``package@^1.0.0:``
        m = re.match(r'^"?(?P<name>[^@"]+)@(?P<ver>[^":]+)"?:', line)
        if m is None:
            continue
        rows.append(
            RepositoryDependency(
                ecosystem=DependencyEcosystem.NPM,
                name=m.group("name"),
                version_spec=f"@{m.group('ver')}",
                source_file=file_path,
            )
        )
    return rows


# =========================================================
# pnpm-lock.yaml
# =========================================================


def _parse_pnpm_lock(path: Path, file_path: str) -> list[RepositoryDependency]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, yaml.YAMLError):
        return []
    rows: list[RepositoryDependency] = []
    if not isinstance(data, dict):
        return rows
    packages = data.get("packages") or {}
    if not isinstance(packages, dict):
        return rows
    for key, meta in packages.items():
        if not isinstance(key, str) or key.startswith("/"):
            # ``/<name>@<version>`` form; strip the leading slash.
            name = key[1:].split("@", 1)[0]
        else:
            name = key.split("@", 1)[0]
        if not name:
            continue
        version = ""
        if isinstance(meta, dict):
            version = str(meta.get("version", "") or "")
        rows.append(
            RepositoryDependency(
                ecosystem=DependencyEcosystem.NPM,
                name=name,
                version_spec=f"=={version}" if version else "",
                source_file=file_path,
            )
        )
    return rows
