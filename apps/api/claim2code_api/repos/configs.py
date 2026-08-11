"""Config file readers (YAML / JSON / TOML / INI).

Public entry point: :func:`parse_config_file`. Returns a
:class:`RepositoryConfig` with the parsed tree in :attr:`content_json`
(or an error string in :attr:`error` when parsing fails).

Design notes:

  - YAML uses the third-party ``PyYAML`` package (declared in
    ``pyproject.toml``).
  - JSON, TOML, and INI use the Python 3.11 stdlib modules —
    ``json``, ``tomllib``, ``configparser`` respectively.
  - We never raise for parse errors. The frontend needs to see the
    file even if it can't be parsed (so the user knows it exists);
    the error field is set and ``content_json`` becomes ``{}``.
"""

from __future__ import annotations

import configparser
import json
import tomllib
from pathlib import Path

import yaml

from claim2code_api.repos.schemas import ConfigFormat, RepositoryConfig

_YAML_EXTS = {".yaml", ".yml"}
_JSON_EXTS = {".json"}
_TOML_EXTS = {".toml"}
_INI_EXTS = {".ini", ".cfg", ".conf"}


def _parse_ini(text: str) -> dict[str, dict[str, str]]:
    """Parse an INI file, tolerating files without section headers.

    Falls back to a single ``DEFAULT`` section containing the loose
    key/value pairs so the user sees *something* even if the file
    wasn't strict INI.
    """
    parser = configparser.ConfigParser()
    try:
        parser.read_string(text)
    except configparser.MissingSectionHeaderError:
        # Try again with the loose key/value lines under a default
        # section header. configparser refuses to parse a file that
        # has no ``[section]`` headers, so we synthesise one.
        section_text = "[DEFAULT]\n" + text
        parser = configparser.ConfigParser()
        parser.read_string(section_text)
    content = {section: dict(parser.items(section)) for section in parser.sections()}
    if not content:
        content = {"DEFAULT": dict(parser.defaults())} if parser.defaults() else {}
    return content


def detect_format(path: Path) -> ConfigFormat | None:
    """Map a file extension to a :class:`ConfigFormat` enum value."""
    ext = path.suffix.lower()
    if ext in _YAML_EXTS:
        return ConfigFormat.YAML
    if ext in _JSON_EXTS:
        return ConfigFormat.JSON
    if ext in _TOML_EXTS:
        return ConfigFormat.TOML
    if ext in _INI_EXTS:
        return ConfigFormat.INI
    return None


def parse_config_file(path: Path, file_path: str) -> RepositoryConfig:
    """Read ``path`` and return a parsed :class:`RepositoryConfig`.

    Args:
        path: Absolute path to the file on disk.
        file_path: Repo-relative POSIX path used as the row's
            :attr:`RepositoryConfig.file_path`.

    Returns:
        A populated :class:`RepositoryConfig`. On parse failure,
        :attr:`error` is set and :attr:`content_json` is ``{}``.
    """
    fmt = detect_format(path)
    size = path.stat().st_size if path.exists() else 0

    if fmt is None:
        return RepositoryConfig(
            file_path=file_path,
            format=ConfigFormat.INI,  # safe default
            size_bytes=size,
            content_json={},
            error=f"unrecognised config extension {path.suffix!r}",
        )

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return RepositoryConfig(
            file_path=file_path,
            format=fmt,
            size_bytes=size,
            content_json={},
            error=f"read error: {exc}",
        )

    try:
        if fmt is ConfigFormat.YAML:
            content = yaml.safe_load(text) or {}
        elif fmt is ConfigFormat.JSON:
            content = json.loads(text)
            if not isinstance(content, dict):
                content = {"_root": content}
        elif fmt is ConfigFormat.TOML:
            content = tomllib.loads(text)
        else:  # INI
            content = _parse_ini(text)
    except Exception as exc:  # noqa: BLE001 — surface any parse error
        return RepositoryConfig(
            file_path=file_path,
            format=fmt,
            size_bytes=size,
            content_json={},
            error=f"{type(exc).__name__}: {exc}",
        )

    # ``content_json`` is typed ``dict`` — coerce nested non-dict roots
    # (e.g. a YAML list at the top level) into a dict-shaped wrapper.
    if not isinstance(content, dict):
        content = {"_root": content}

    return RepositoryConfig(
        file_path=file_path,
        format=fmt,
        size_bytes=size,
        content_json=content,
    )
